"""OpenAI-compatible streaming chat client. Same code talks to Groq and to the stand-in."""
import asyncio, json, re, time
from dataclasses import dataclass, field
from .config import Endpoint
from .http import post


class RateLimited(Exception):
    def __init__(self, retry_after, scope, message="", headers=None):
        super().__init__(message)
        self.retry_after, self.scope, self.message, self.headers = retry_after, scope, message, headers or {}


class Transient(Exception):
    """5xx, dropped connection, mid-stream failure: worth retrying."""


class LLMTimeout(Exception):
    pass


class Fatal(Exception):
    """4xx that retrying will not fix."""


@dataclass
class LLMResult:
    text: str
    first_token_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    headers: dict = field(default_factory=dict)
    model: str = ""
    usage_estimated: bool = False


_DUR = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)(?![a-z])")


def parse_duration(s: str):
    total = sum(float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[u] for n, u in _DUR.findall(s or ""))
    return total or None


def detect_scope(msg: str):
    m = re.search(r"on (requests|tokens) per (minute|day)", msg or "")
    return {("requests", "minute"): "rpm", ("tokens", "minute"): "tpm",
            ("requests", "day"): "rpd", ("tokens", "day"): "tpd"}.get(m.groups() if m else None, "unknown")


def _rl_headers(h):
    return {k: v for k, v in h.items() if k.startswith("x-ratelimit") or k == "retry-after"}


class LLMClient:
    def __init__(self, ep: Endpoint, connect_timeout=10.0, stall_timeout=12.0, total_timeout=45.0):
        self.ep, self.connect_timeout, self.stall_timeout, self.total_timeout = ep, connect_timeout, stall_timeout, total_timeout

    async def chat(self, model, messages, max_tokens=300, temperature=0.0, json_mode=False,
                   on_token=None, timeout_s=None) -> LLMResult:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "stream_options": {"include_usage": True}}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp, hdrs = None, {}
        try:
            async with asyncio.timeout(min(self.total_timeout, timeout_s) if timeout_s else self.total_timeout):
                resp = await asyncio.wait_for(
                    post(self.ep.base_url + "/chat/completions",
                         {"Authorization": f"Bearer {self.ep.api_key}", "Content-Type": "application/json"},
                         json.dumps(body).encode()), self.connect_timeout)
                hdrs = _rl_headers(resp.headers)
                if resp.status != 200:
                    raw = (await resp.read_all()).decode("utf-8", "replace")
                    try:
                        msg = json.loads(raw).get("error", {}).get("message", raw)
                    except Exception:
                        msg = raw[:300]
                    if resp.status == 429:
                        scope = detect_scope(msg)
                        ra = resp.headers.get("retry-after")
                        ra = float(ra) if ra else (parse_duration(msg) or 5.0)
                        raise RateLimited(ra, scope, msg, hdrs)
                    if resp.status >= 500 or resp.status in (408, 409):
                        raise Transient(f"{resp.status}: {msg[:120]}")
                    raise Fatal(f"{resp.status}: {msg[:200]}")
                text, first, usage, buf = [], None, None, b""
                it = resp.iter_body().__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(it.__anext__(), self.stall_timeout)
                    except StopAsyncIteration:
                        break
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    done = False
                    for ln in lines:
                        ln = ln.strip()
                        if not ln.startswith(b"data:"):
                            continue
                        payload = ln[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                            break
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            continue
                        if obj.get("error"):
                            raise Transient("mid-stream error: " + str(obj["error"])[:120])
                        usage = obj.get("usage") or (obj.get("x_groq") or {}).get("usage") or usage
                        for ch in obj.get("choices", []):
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                if first is None:
                                    first = (time.perf_counter() - t0) * 1000
                                text.append(piece)
                                if on_token:
                                    on_token(piece)
                    if done:
                        break
        except TimeoutError as e:
            raise LLMTimeout(str(e) or "timeout") from e
        except (ConnectionError, OSError) as e:
            raise Transient(f"connection: {e}") from e
        finally:
            if resp:
                resp.close()
        total = (time.perf_counter() - t0) * 1000
        out = "".join(text)
        est = usage is None
        tin = (usage or {}).get("prompt_tokens") or sum(len(m["content"]) for m in messages) // 4
        tout = (usage or {}).get("completion_tokens") or max(1, len(out) // 4)
        return LLMResult(out, first if first is not None else total, total, int(tin), int(tout), hdrs, model, est)
