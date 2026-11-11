"""understand -> retrieve -> decide (+tool loop) -> draft -> verify (+redraft loop) -> deliver.
THE decision that shapes everything: the draft is buffered and only streamed after it passes verification
(cfg.verify_before_stream=True). Cost: first token ~= end of pipeline, not the first model token. Benefit: no wrong
or unverified text ever reaches a person. Set it False to measure what streaming-first would buy and risk."""
import asyncio, json, re, time
from dataclasses import dataclass, field
from . import retry as R
from .config import FAST_MODEL, STRONG_MODEL
from .governor import Shed
from .llm import Fatal, LLMTimeout, RateLimited
from .retrieval import CITE, PAPERS, SETTINGS, Hit, RetrievalSetting, coverage, grounding_check
from .tools import ToolError
from .trace import Trace


@dataclass
class PipelineConfig:
    setting: RetrievalSetting = SETTINGS["w250_k4"]
    verify_before_stream: bool = True
    llm_verify: bool = True
    max_redrafts: int = 1
    max_expansions: int = 1
    patience_s: float = 25.0
    soft_ratio: float = 0.8        # after this fraction of patience, stop calling the model and answer extractively
    draft_max_tokens: int = 260
    fast_model: str = FAST_MODEL
    strong_model: str = STRONG_MODEL


@dataclass
class Answer:
    text: str = ""
    status: str = "ok"          # ok | rate_limited (degraded because of limits) | timeout | error
    mode: str = "normal"        # normal | lean | extractive
    refused: bool = False
    grounding: str = "none"     # llm+det | det | none
    redrafts: int = 0
    retracted: bool = False
    reason: str = ""
    cited: list = field(default_factory=list)
    trace: Trace = None


SYS = {
    "understand": ("TASK:understand\nYou route questions for an assistant that answers ONLY from three papers: "
                   "attn (Attention Is All You Need), moshi (Moshi: speech-text dialogue model), hnet (Dynamic Chunking "
                   "for End-to-End Hierarchical Sequence Modeling / H-Net). Reply with JSON only: "
                   '{"intent":"papers"|"off_topic","query":"<search keywords>","papers":["attn"|"moshi"|"hnet"]}. '
                   "Use off_topic only if the question has nothing to do with the subject matter of those papers. "
                   "A question the papers might not answer is still intent papers. Use papers [] if unsure."),
    "decide": ("TASK:decide\nGiven a QUESTION and numbered PASSAGES, reply with JSON only: "
               '{"action":"answer"|"expand"|"refuse","reason":"<short>"}. answer: the passages contain the answer. '
               "expand: passages look cut off and the answer is probably in the adjacent text. "
               "refuse: the passages do not contain the answer."),
    "draft": ("TASK:draft\nAnswer the QUESTION using ONLY the PASSAGES. Every sentence must carry a citation copied from "
              "a passage label, placed before the final full stop, like: The base model uses 6 layers [attn p3]. "
              "Never state a fact or number that is not in the cited passage. If the passages do not contain the answer, "
              "reply exactly NOT_IN_PAPERS and nothing else. Max 110 words."),
    "verify": ("TASK:verify\nYou check an ANSWER against PASSAGES. Reply with JSON only: "
               '{"supported":true|false,"unsupported":["<claim>"]}. supported=true only if every claim in ANSWER is '
               "stated in the passage it cites."),
}


def fmt_passages(hits, max_chars=1400):
    return "\n".join(f"[{h.chunk.paper} p{h.chunk.page}] (§{h.chunk.section or '?'}) {h.chunk.text[:max_chars]}" for h in hits)


def parse_json(s, default):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return default


class Pipeline:
    def __init__(self, cfg, indexes, governor, client, tools, retry_budget, policy=None):
        self.cfg, self.indexes, self.gov, self.client, self.tools = cfg, indexes, governor, client, tools
        self.budget, self.policy = retry_budget, policy or R.RetryPolicy()

    # ---- helpers ----
    async def _call(self, step, model, system, user, tr, deadline, max_tokens, json_mode=False, on_token=None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return await R.call(self.gov, self.client, model, msgs, step=step, trace=tr, deadline=deadline,
                            policy=self.policy, budget=self.budget, max_tokens=max_tokens, json_mode=json_mode,
                            on_token=on_token)

    def _deliver(self, text, tr, emit):
        for piece in re.findall(r"\S+\s*", text):
            tr.mark_first_token()
            if emit:
                emit(piece)

    def _extractive(self, hits, question, why):
        if not hits:
            return ("I could not look this up right now and I won't guess. Please try again shortly. "
                    f"({why})")
        parts = []
        for h in hits[:2]:
            sent = re.split(r"(?<=[.!?])\s+", h.chunk.text)[:2]
            parts.append(" ".join(sent)[:380] + f" [{h.chunk.paper} p{h.chunk.page}]")
        return ("The full assistant is at capacity, so this is a passage-only reply, not a written answer. "
                "Closest passages from the papers:\n" + "\n".join("- " + p for p in parts) + f"\n({why})")

    def _refusal(self, hits):
        near = ", ".join(sorted({f"[{h.chunk.paper} p{h.chunk.page}]" for h in hits[:3]}))
        return ("The three papers I can use do not say. I won't guess."
                + (f" Closest related passages: {near}." if near else ""))

    # ---- main ----
    async def answer(self, rid, question, emit=None, trace=None) -> Answer:
        cfg = self.cfg
        tr = trace or Trace(rid)
        t0 = time.monotonic()
        hard, soft = t0 + cfg.patience_s, t0 + cfg.patience_s * cfg.soft_ratio
        ans = Answer(trace=tr)
        models = [cfg.fast_model, cfg.strong_model]
        mode = self.gov.mode(models) if self.gov else "normal"
        ans.mode = mode
        index = self.indexes(cfg.setting)
        hits, query, papers = [], question, None
        try:
            if mode == "extractive":
                raise Shed("daily_budget")
            # 1 understand (skipped when lean)
            if mode == "normal":
                r = await self._call("understand", cfg.fast_model, SYS["understand"], question, tr, soft, 80, True)
                u = parse_json(r.text, {})
                if u.get("intent") == "off_topic":
                    ans.refused, ans.text, ans.reason = True, self._refusal([]).replace(
                        "do not say", "do not cover that topic"), "off_topic"
                    self._deliver(ans.text, tr, emit)
                    return ans
                query = u.get("query") or question
                papers = [p for p in (u.get("papers") or []) if p in PAPERS] or None
            # 2 retrieve (local)
            ts = time.monotonic()
            hits = index.search(query, cfg.setting, papers)
            if not hits and papers:
                hits = index.search(query, cfg.setting, None)
            tr.span("work", "retrieve", ts, time.monotonic(), f"{len(hits)} hits")
            # 3 decide, with a tool-call loop
            expansions = 0
            while True:
                if mode == "lean":
                    action = "answer" if coverage(query, hits) >= 0.5 else "refuse"
                else:
                    r = await self._call("decide", cfg.fast_model, SYS["decide"],
                                         f"QUESTION: {question}\n\nPASSAGES:\n{fmt_passages(hits, 700)}", tr, soft, 60, True)
                    action = parse_json(r.text, {}).get("action", "answer")
                if action == "expand" and expansions < cfg.max_expansions and hits:
                    expansions += 1
                    ts = time.monotonic()
                    try:
                        extra = await self.tools.neighbors(index, hits[0].chunk)
                        seen = {h.chunk.id for h in hits}
                        hits += [Hit(c, 0.0) for c in extra if c.id not in seen]
                        tr.span("work", "tool:neighbors", ts, time.monotonic(), f"+{len(extra)}")
                    except ToolError as e:
                        tr.span("retry", "tool:neighbors", ts, time.monotonic(), str(e))
                        tr.notes.append("tool_failed_continue_without")
                        action = "answer"
                        break
                    continue
                break
            if action == "refuse":
                ans.refused, ans.reason = True, "decide_refuse"
                ans.text = self._refusal(hits)
                self._deliver(ans.text, tr, emit)
                return ans
            # 4 draft + 5 verify loop
            feedback, streamed = "", False
            for attempt in range(cfg.max_redrafts + 1):
                user = f"QUESTION: {question}\n\nPASSAGES:\n{fmt_passages(hits)}"
                if feedback:
                    user += f"\n\nYour previous draft was rejected: {feedback}. Fix it using only the passages."
                cb = None
                if not cfg.verify_before_stream and attempt == 0:
                    def cb(piece):
                        nonlocal streamed
                        streamed = True
                        tr.mark_first_token()
                        if emit:
                            emit(piece)
                d = await self._call("draft", cfg.strong_model, SYS["draft"], user, tr, soft, cfg.draft_max_tokens,
                                     False, cb)
                text = d.text.strip()
                if "NOT_IN_PAPERS" in text:
                    ans.refused, ans.reason, ans.text = True, "draft_not_in_papers", self._refusal(hits)
                    if not streamed:
                        self._deliver(ans.text, tr, emit)
                    return ans
                ok_det, problems = grounding_check(text, hits)
                ok, how = ok_det, "det"
                if ok_det and mode == "normal" and cfg.llm_verify:
                    v = await self._call("verify", cfg.fast_model, SYS["verify"],
                                         f"ANSWER: {text}\n\nPASSAGES:\n{fmt_passages(hits, 700)}", tr, soft, 80, True)
                    vj = parse_json(v.text, {"supported": True})
                    ok, how = bool(vj.get("supported", True)), "llm+det"
                    if not ok:
                        problems = vj.get("unsupported") or ["unsupported claim"]
                if ok:
                    ans.text, ans.grounding = text, how
                    ans.cited = sorted({f"{p} p{pg}" for p, pg in CITE.findall(text)})
                    if not streamed:
                        self._deliver(text, tr, emit)
                    return ans
                ans.redrafts += 1
                feedback = "; ".join(map(str, problems))[:300]
                if streamed:
                    ans.retracted = True
            # unverifiable after the redraft: never ship it
            ans.mode, ans.reason = "extractive", "verification_failed"
            ans.text = self._extractive(hits, question, "I could not verify a written answer")
            if ans.retracted and emit:
                emit("\n[Correction: the text above could not be verified. Use the passages below instead.]\n")
            self._deliver(ans.text, tr, emit)
            return ans
        except Shed as e:
            ans.mode, ans.status, ans.reason = "extractive", "rate_limited", f"shed:{e.reason}"
        except (R.RetriesExhausted, R.GivenUp) as e:
            # classify by cause: limits -> rate_limited, slow/timed-out calls -> timeout, anything else -> error
            kind = "gave_up_before_retry" if isinstance(e, R.GivenUp) else "retries_exhausted"
            if isinstance(e.last, RateLimited):
                st = "rate_limited"
            elif isinstance(e.last, LLMTimeout):
                st = "timeout"
            else:
                st = "error"
            ans.mode, ans.status, ans.reason = "extractive", st, f"{kind}:{type(e.last).__name__}"
        except TimeoutError:
            ans.mode, ans.status, ans.reason = "extractive", "timeout", "deadline"
        except Fatal as e:
            ans.mode, ans.status, ans.reason = "extractive", "error", f"fatal:{str(e)[:60]}"
        # graceful degradation: the person still gets something true, with citations, and is told what it is
        if not hits:
            ts = time.monotonic()
            try:
                hits = index.search(question, cfg.setting, None, k=3)
            except Exception:
                hits = []
            tr.span("work", "retrieve_fallback", ts, time.monotonic())
        ans.text = self._extractive(hits, question, {"daily_budget": "daily model budget used up; full answers resume after reset"}.get(
            ans.reason.split(":")[-1], "model limit reached"))
        self._deliver(ans.text, tr, emit)
        return ans
