"""Shared plumbing: build the system, run one request, write results.csv rows + a detail file."""
import asyncio, csv, json, os, time
from datetime import datetime, timezone
from . import retry as R
from .config import FAST_MODEL, STRONG_MODEL, endpoint, load_limits
from .governor import Governor
from .llm import LLMClient
from .pipeline import Pipeline, PipelineConfig
from .retrieval import SETTINGS, Index, load_papers, make_chunks
from .tools import ToolService
from .trace import Trace

CSV_FIELDS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
              "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]


class System:
    def __init__(self, source, setting="w250_k4", papers_dir="data/papers", limits_path="limits.json", patience_s=25.0,
                 naive_retry=False, verify_before_stream=True, tool_kwargs=None, use_governor=True):
        self.source = source
        self.papers = load_papers(papers_dir)
        self._idx = {}
        self.cfg = PipelineConfig(setting=SETTINGS[setting], patience_s=patience_s, verify_before_stream=verify_before_stream)
        self.client = LLMClient(endpoint(source))
        self.gov = Governor(load_limits(limits_path)) if use_governor else None
        self.tools = ToolService(**(tool_kwargs or {}))
        policy = R.RetryPolicy(naive=naive_retry)
        self.pipe = Pipeline(self.cfg, self.index_for, self.gov, self.client, self.tools, R.RetryBudget(policy.budget_ratio), policy)

    def index_for(self, setting):
        key = (setting.chunk_words, setting.overlap)
        if key not in self._idx:
            self._idx[key] = Index(make_chunks(self.papers, *key))
        return self._idx[key]

    def set_setting(self, name):
        self.cfg.setting = SETTINGS[name]

    async def start(self):
        if self.gov:
            await self.gov.start()

    async def stop(self):
        if self.gov:
            await self.gov.stop()


async def run_one(system: System, rid, scenario, question, grader_q=None, writer=None, detail=None):
    tr = Trace(rid)
    start_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    status, ans = "ok", None
    try:
        ans = await asyncio.wait_for(system.pipe.answer(rid, question, trace=tr), system.cfg.patience_s)
        status = ans.status
    except TimeoutError:
        status = "timeout"          # the person gave up before any answer reached them
    except Exception as e:           # noqa: BLE001  (record, never crash the load run)
        status, tr.notes = "error", tr.notes + [f"{type(e).__name__}: {e}"]
    total = tr.now_ms()
    first = tr.first_token_ms
    grade = ""
    if grader_q is not None and ans is not None:
        from .grader import grade as g
        grade = g(ans, grader_q)
    row = {"request_id": rid, "scenario": scenario, "source": system.source, "start_time": start_iso,
           "first_token_ms": "" if first is None else round(first, 1), "total_ms": round(total, 1), "status": status,
           "retries": tr.retries, "tokens_in": tr.tokens_in, "tokens_out": tr.tokens_out,
           "retrieval_setting": system.cfg.setting.name, "answer_grade": grade}
    if writer:
        writer.writerow(row)
    if detail is not None:
        detail.write(json.dumps({
            "request_id": rid, "scenario": scenario, "question": question, "mode": getattr(ans, "mode", None),
            "reason": getattr(ans, "reason", ""), "refused": getattr(ans, "refused", None),
            "grounding": getattr(ans, "grounding", ""), "redrafts": getattr(ans, "redrafts", 0),
            "retracted": getattr(ans, "retracted", False), "answer": getattr(ans, "text", ""),
            "cited": getattr(ans, "cited", []), "split_ms": tr.split(), "first_token_ms": first,
            "spans": tr.spans, "calls": tr.calls, "notes": tr.notes}) + "\n")
        detail.flush()
    return row, ans


def open_outputs(outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "results.csv")
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    f = open(p, "a", newline="")
    w = csv.DictWriter(f, CSV_FIELDS)
    if new:
        w.writeheader()
    d = open(os.path.join(outdir, "results_detail.jsonl"), "a")
    return f, w, d
