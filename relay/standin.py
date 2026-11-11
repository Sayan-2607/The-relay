
"""Stand-in model service (OpenAI-compatible /v1/chat/completions, streaming SSE).

Can:
- Be slow (lognormal TTFT + decode speed)
- Have a heavy tail ("succeeds slowly")
- Fail (5xx, mid-stream drops, hangs)
- Refuse with Groq-shaped 429s (RPM/TPM/RPD/TPD, retry-after, x-ratelimit-* headers)
- Slow down under load
- Stream pieces

Replies are canned but shaped like the real thing:
JSON for understand/decide/verify, cited text for draft.

Calibrate with calibrate.py -> standin_config.json.

Run:
    python -m relay.standin --config standin_config.json

Admin:
    POST /admin/reset {"set": {"models/llama-3.1-8b-instant/rpm": 15},
                       "day_used_req": {...}}
    GET /admin/stats
"""

import argparse
import asyncio
import copy
import json
import math
import random
import re
import time
from collections import Counter, defaultdict, deque


DEFAULT = {
    "models": {
        "llama-3.1-8b-instant": {
            "ttft_median_s": 0.25,
            "ttft_sigma": 0.35,
            "tps_median": 450,
            "tps_sigma": 0.2,
            "prefill_ms_per_tok": 0.05,
            "rpm": 30,
            "tpm": 6000,
            "rpd": 14400,
            "tpd": 500000,
        },
        "llama-3.3-70b-versatile": {
            "ttft_median_s": 0.45,
            "ttft_sigma": 0.35,
            "tps_median": 180,
            "tps_sigma": 0.2,
            "prefill_ms_per_tok": 0.15,
            "rpm": 30,
            "tpm": 12000,
            "rpd": 1000,
            "tpd": 100000,
        },
    },
    "max_concurrency": 8,
    "load_knee": 4,
    "load_alpha": 0.5,
    "error_rate": 0.01,
    "midstream_error_share": 0.3,
    "slow_rate": 0.03,
    "slow_factor": 8,
    "hang_rate": 0.002,
    "verify_fail_rate": 0.1,
    "expand_rate": 0.15,
    "draft_tokens_median": 110,
    "day_reset_s": 10800,
    "seed": 1,
}

OFFTOPIC = (
    "world cup",
    "recipe",
    "weather",
    "stock price",
    "football",
    "movie",
)

STOPW = set(
    "a an the of in on to and or is are was were be by for with as at "
    "from that this it its what how why do does did".split()
)


def words(s):
    return [
        w
        for w in re.findall(r"[a-z0-9]+", s.lower())
        if w not in STOPW
    ]


def est_tokens(s):
    return max(1, len(s) // 4)


def dur(s):
    s = int(math.ceil(s))
    return f"{s // 60}m{s % 60}s" if s >= 60 else f"{s}s"


class StandIn:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed", 1))
        self.sem = asyncio.Semaphore(cfg["max_concurrency"])
        self.active = 0
        self.reset_state()

    def reset_state(self):
        self.win = defaultdict(deque)
        self.day_req = Counter()
        self.day_tok = Counter()
        self.day_start = time.monotonic()
        self.stats = Counter()
        self.peak_active = 0

    def mcfg(self, model):
        return self.cfg["models"].get(model) or next(
            iter(self.cfg["models"].values())
        )

    def day_left(self):
        return max(
            1.0,
            self.cfg["day_reset_s"] - (time.monotonic() - self.day_start),
        )

    def check_limits(self, model, tin, max_out):
        m = self.mcfg(model)
        now = time.monotonic()
        w = self.win[model]

        while w and w[0][0] < now - 60:
            w.popleft()

        req_est = tin + max_out

        if self.day_req[model] >= m["rpd"]:
            return (
                "requests",
                "day",
                m["rpd"],
                self.day_req[model],
                1,
                self.day_left(),
            )

        if self.day_tok[model] + req_est > m["tpd"]:
            return (
                "tokens",
                "day",
                m["tpd"],
                self.day_tok[model],
                req_est,
                self.day_left(),
            )

        if len(w) + 1 > m["rpm"]:
            return (
                "requests",
                "minute",
                m["rpm"],
                len(w),
                1,
                w[0][0] + 60 - now,
            )

        used = sum(t for _, t in w)

        if used + req_est > m["tpm"]:
            acc = 0
            for t, tok in w:
                acc += tok
                if used - acc + req_est <= m["tpm"]:
                    return (
                        "tokens",
                        "minute",
                        m["tpm"],
                        used,
                        req_est,
                        t + 60 - now,
                    )

            return (
                "tokens",
                "minute",
                m["tpm"],
                used,
                req_est,
                60.0,
            )

        return None

    def rl_headers(self, model):
        m = self.mcfg(model)
        now = time.monotonic()
        w = self.win[model]

        used = sum(
            t for t0, t in w
            if t0 >= now - 60
        )

        return {
            "x-ratelimit-limit-requests": int(m["rpd"]),
            "x-ratelimit-limit-tokens": int(m["tpm"]),
            "x-ratelimit-remaining-requests": max(
                0, int(m["rpd"] - self.day_req[model])
            ),
            "x-ratelimit-remaining-tokens": max(
                0, int(m["tpm"] - used)
            ),
            "x-ratelimit-reset-requests": dur(self.day_left()),
            "x-ratelimit-reset-tokens": dur(2),
        }

    # ---------------------------------------------------------
    # Canned model responses
    # ---------------------------------------------------------

    def content(self, model, messages, max_tokens):
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "",
        )
        user = messages[-1]["content"]

        task = (
            re.match(r"TASK:(\w+)", system) or [None, "draft"]
        )[1]

        c = self.cfg

        if task == "understand":
            low = user.lower()

            return json.dumps({
                "intent": (
                    "off_topic"
                    if any(k in low for k in OFFTOPIC)
                    else "papers"
                ),
                "query": " ".join(words(user)[:12]),
                "papers": [],
            })

        passages = re.findall(
            r"^\[(attn|moshi|hnet) p(\d+)\][^\n]*?\)\s*(.*)$",
            user,
            re.M,
        )

        q = words(
            (re.search(r"QUESTION: (.*)", user) or [None, ""])[1]
        )

        if task == "decide":
            ptxt = set(words(" ".join(p[2] for p in passages)))
            cov = (
                len(set(q) & ptxt) / max(1, len(set(q)))
            )

            if cov < 0.3:
                act = "refuse"
            elif self.rng.random() < c["expand_rate"]:
                act = "expand"
            else:
                act = "answer"

            return json.dumps({
                "action": act,
                "reason": "stub",
            })

        if task == "verify":
            ok = self.rng.random() > c["verify_fail_rate"]

            return json.dumps({
                "supported": ok,
                "unsupported": (
                    [] if ok else ["stub unsupported claim"]
                ),
            })

        # Draft: quote the first sentences of the top passages
        # and attach page citations.
        if not passages:
            return "NOT_IN_PAPERS"

        target = int(
            c["draft_tokens_median"]
            * math.exp(self.rng.gauss(0, 0.3))
        )

        out = []

        for p, pg, txt in passages:
            for s in re.split(r"(?<=[.!?])\s+", txt):
                if len(s.split()) >= 5:
                    out.append(
                        s.rstrip(".") + f" [{p} p{pg}]."
                    )

                if est_tokens(" ".join(out)) >= min(
                    target, max_tokens
                ):
                    break

            if est_tokens(" ".join(out)) >= min(
                target, max_tokens
            ):
                break

        return (
            " ".join(out)[: max_tokens * 4]
            or "NOT_IN_PAPERS"
        )

    # ---------------------------------------------------------
    # HTTP request handling
    # ---------------------------------------------------------

    async def handle(self, reader, writer):
        try:
            line = await reader.readline()

            if not line:
                return

            try:
                method, path, _ = line.decode().split(" ", 2)
            except (UnicodeDecodeError, ValueError):
                self.send(
                    writer,
                    400,
                    {"error": "invalid HTTP request line"},
                )
                await writer.drain()
                return

            hdrs = {}

            while True:
                l = await reader.readline()

                if l in (b"\r\n", b""):
                    break

                k, _, v = l.decode().partition(":")
                hdrs[k.strip().lower()] = v.strip()

            try:
                content_length = int(
                    hdrs.get("content-length", "0")
                )
            except ValueError:
                self.send(
                    writer,
                    400,
                    {"error": "invalid content-length"},
                )
                await writer.drain()
                return

            if content_length < 0:
                self.send(
                    writer,
                    400,
                    {"error": "invalid content-length"},
                )
                await writer.drain()
                return

            body = (
                await reader.readexactly(content_length)
                if content_length
                else b""
            )

            # GET /admin/stats does not need a JSON body.
            if path.startswith("/admin/stats"):
                return await self.admin(
                    method, path, body, writer
                )

            if not body:
                self.send(
                    writer,
                    400,
                    {"error": "empty request body"},
                )
                await writer.drain()
                return

            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send(
                    writer,
                    400,
                    {"error": "invalid JSON request body"},
                )
                await writer.drain()
                return

            if path.startswith("/admin"):
                return await self.admin(
                    method, path, body, writer
                )

            if (
                method != "POST"
                or path != "/v1/chat/completions"
            ):
                self.send(
                    writer,
                    404,
                    {"error": "not found"},
                )
                await writer.drain()
                return

            if not isinstance(payload, dict):
                self.send(
                    writer,
                    400,
                    {"error": "JSON body must be an object"},
                )
                await writer.drain()
                return

            await self.chat(payload, writer)

        except (ConnectionError, asyncio.IncompleteReadError):
            pass

        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError, RuntimeError):
                pass

    def send(self, writer, status, obj, extra=None):
        data = json.dumps(obj).encode()

        h = {
            "Content-Type": "application/json",
            "Content-Length": len(data),
            "Connection": "close",
            **(extra or {}),
        }

        writer.write(
            f"HTTP/1.1 {status} X\r\n".encode()
            + "".join(
                f"{k}: {v}\r\n" for k, v in h.items()
            ).encode()
            + b"\r\n"
            + data
        )

    async def admin(self, method, path, body, writer):
        if path.startswith("/admin/stats"):
            self.send(
                writer,
                200,
                {
                    "stats": dict(self.stats),
                    "peak_active": self.peak_active,
                    "day_req": dict(self.day_req),
                    "config": self.cfg,
                },
            )
            await writer.drain()
            return

        try:
            req = json.loads(body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send(
                writer,
                400,
                {"error": "invalid admin JSON"},
            )
            await writer.drain()
            return

        if not isinstance(req, dict):
            self.send(
                writer,
                400,
                {"error": "admin JSON must be an object"},
            )
            await writer.drain()
            return

        for dotted, val in (req.get("set") or {}).items():
            d = self.cfg
            parts = dotted.split("/")

            for p in parts[:-1]:
                d = d[p]

            d[parts[-1]] = val

        if req.get("reset", True):
            self.reset_state()

        for m, n in (req.get("day_used_req") or {}).items():
            self.day_req[m] = int(n)

        self.send(writer, 200, {"ok": True})
        await writer.drain()

    # ---------------------------------------------------------
    # Chat completions
    # ---------------------------------------------------------

    async def chat(self, req, writer):
        c = self.cfg
        model = req.get("model", "")
        m = self.mcfg(model)

        messages = req["messages"]
        max_out = int(req.get("max_tokens", 300))
        stream = req.get("stream", False)

        tin = sum(
            est_tokens(x["content"])
            for x in messages
        )

        self.stats["requests"] += 1

        lim = self.check_limits(model, tin, max_out)

        if lim:
            kind, unit, limit, used, asked, wait = lim

            abbr = (
                {"requests": "R", "tokens": "T"}[kind]
                + "P"
                + unit[0].upper()
            )

            msg = (
                f"Rate limit reached for model `{model}` "
                f"in organization `org_standin` "
                f"service tier `on_demand` on "
                f"{kind} per {unit} ({abbr}): "
                f"Limit {limit}, Used {used}, "
                f"Requested {asked}. "
                f"Please try again in {dur(wait)}."
            )

            self.stats[f"429_{abbr.lower()}"] += 1

            return self.send(
                writer,
                429,
                {
                    "error": {
                        "message": msg,
                        "type": kind,
                        "code": "rate_limit_exceeded",
                    }
                },
                {
                    "retry-after": int(math.ceil(wait)),
                    **self.rl_headers(model),
                },
            )

        self.win[model].append(
            (time.monotonic(), tin + max_out)
        )

        self.day_req[model] += 1
        self.day_tok[model] += tin + max_out

        rng = self.rng

        if rng.random() < c["hang_rate"]:
            self.stats["hang"] += 1
            await asyncio.sleep(120)

        async with self.sem:
            self.active += 1
            self.peak_active = max(
                self.peak_active,
                self.active,
            )

            try:
                mult = (
                    1
                    + c["load_alpha"]
                    * max(0, self.active - c["load_knee"])
                    / max(1, c["load_knee"])
                )

                slow = rng.random() < c["slow_rate"]

                if slow:
                    mult *= c["slow_factor"]
                    self.stats["slow"] += 1

                ttft = (
                    m["ttft_median_s"]
                    * math.exp(
                        rng.gauss(0, m["ttft_sigma"])
                    )
                    + tin * m["prefill_ms_per_tok"] / 1000
                )

                tps = (
                    m["tps_median"]
                    * math.exp(
                        rng.gauss(0, m["tps_sigma"])
                    )
                )

                err = rng.random() < c["error_rate"]

                if (
                    err
                    and rng.random() > c["midstream_error_share"]
                ):
                    await asyncio.sleep(ttft * 0.3)
                    self.stats["error_5xx"] += 1

                    return self.send(
                        writer,
                        503,
                        {"error": {"message": "stub overloaded"}},
                        self.rl_headers(model),
                    )

                text = self.content(
                    model, messages, max_out
                )

                tout = est_tokens(text)

                await asyncio.sleep(ttft * mult)

                if not stream:
                    await asyncio.sleep(
                        tout / tps * mult
                    )

                    return self.send(
                        writer,
                        200,
                        {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": text,
                                    }
                                }
                            ],
                            "usage": {
                                "prompt_tokens": tin,
                                "completion_tokens": tout,
                            },
                        },
                        self.rl_headers(model),
                    )

                head = {
                    "Content-Type": "text/event-stream",
                    "Transfer-Encoding": "chunked",
                    "Connection": "close",
                    **self.rl_headers(model),
                }

                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    + "".join(
                        f"{k}: {v}\r\n"
                        for k, v in head.items()
                    ).encode()
                    + b"\r\n"
                )

                def chunk(obj):
                    b = (
                        b"data: "
                        + (
                            obj
                            if isinstance(obj, bytes)
                            else json.dumps(obj).encode()
                        )
                        + b"\n\n"
                    )

                    writer.write(
                        f"{len(b):x}\r\n".encode()
                        + b
                        + b"\r\n"
                    )

                pieces = re.findall(r"\S+\s*", text)
                step = 4
                i = 0
                drop_at = (
                    len(pieces) // 2
                    if err
                    else None
                )

                while i < len(pieces):
                    part = "".join(
                        pieces[i:i + step]
                    )

                    chunk({
                        "choices": [
                            {"delta": {"content": part}}
                        ]
                    })

                    await writer.drain()
                    i += step

                    if (
                        drop_at is not None
                        and i >= drop_at
                    ):
                        self.stats["error_midstream"] += 1
                        return

                    await asyncio.sleep(
                        est_tokens(part) / tps * mult
                    )

                chunk({
                    "choices": [],
                    "usage": {
                        "prompt_tokens": tin,
                        "completion_tokens": tout,
                    },
                })

                chunk(b"[DONE]")

                writer.write(b"0\r\n\r\n")
                await writer.drain()

            finally:
                self.active -= 1


async def serve(cfg, host="127.0.0.1", port=8099):
    s = StandIn(cfg)

    server = await asyncio.start_server(
        s.handle,
        host,
        port,
        backlog=1024,
    )

    return s, server


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="standin_config.json",
    )

    ap.add_argument(
        "--port",
        type=int,
        default=8099,
    )

    a = ap.parse_args()

    cfg = copy.deepcopy(DEFAULT)

    try:
        with open(a.config, encoding="utf-8") as f:
            user = json.load(f)

        for k, v in user.items():
            if k == "models":
                for mk, mv in v.items():
                    cfg["models"].setdefault(
                        mk, {}
                    ).update(mv)
            else:
                cfg[k] = v

        print(f"loaded calibration from {a.config}")

    except FileNotFoundError:
        print(
            f"{a.config} not found: "
            "using UNCALIBRATED defaults "
            "(run probe.py + calibrate.py)"
        )

    async def run():
        _, server = await serve(
            cfg,
            port=a.port,
        )

        print(
            f"stand-in listening on "
            f"http://127.0.0.1:{a.port}/v1"
        )

        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
