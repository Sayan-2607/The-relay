"""Per-request timeline. Every span is wait (queued in the governor), work (a call that succeeded
or local compute) or retry (failed attempt time + backoff sleep). Section 4 of the write-up uses this."""
import time


class Trace:
    def __init__(self, rid):
        self.rid, self.t0 = rid, time.perf_counter()
        self.spans, self.first_token_ms = [], None
        self.retries = self.calls = self.tokens_in = self.tokens_out = 0
        self.notes = []

    def now_ms(self):
        return (time.perf_counter() - self.t0) * 1000

    def span(self, kind, step, t_start, t_end, note=""):
        self.spans.append({"kind": kind, "step": step, "start_ms": round((t_start - self.t0) * 1000, 1),
                           "end_ms": round((t_end - self.t0) * 1000, 1), "note": note})

    def mark_first_token(self):
        if self.first_token_ms is None:
            self.first_token_ms = self.now_ms()

    def split(self):
        s = {"wait": 0.0, "work": 0.0, "retry": 0.0}
        for sp in self.spans:
            s[sp["kind"]] += sp["end_ms"] - sp["start_ms"]
        s["total"] = self.now_ms()
        s["other"] = s["total"] - s["wait"] - s["work"] - s["retry"]
        return {k: round(v, 1) for k, v in s.items()}
