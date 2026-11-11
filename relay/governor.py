"""Client-side rate governor: paces calls so we (mostly) never hit the provider's 429, orders waiting
calls earliest-deadline-first, sheds calls that cannot finish in time, learns the real limits
(headers + 429s), and tells the pipeline which quality mode it can afford (normal / lean / extractive)."""
import asyncio, time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass


@dataclass
class Limits:
    rpm: float = 30
    tpm: float = 6000
    rpd: float = 14400
    tpd: float = 500000
    max_inflight: int = 4


class Shed(Exception):
    """Governor refuses the call: reason is daily_budget | deadline | would_miss_deadline."""
    def __init__(self, reason, wait=None):
        super().__init__(reason)
        self.reason, self.wait = reason, wait


class Permit:
    __slots__ = ("model", "entry", "est")

    def __init__(self, model, entry, est):
        self.model, self.entry, self.est = model, entry, est


class Governor:
    def __init__(self, limits, safety=0.9, lean_queue=12, day_reserve=0.10):
        self.limits, self.safety, self.lean_queue, self.day_reserve = limits, safety, lean_queue, day_reserve
        self.cap_rpm = {m: float(l.rpm) for m, l in limits.items()}   # AIMD-adjusted per-minute request cap
        self.win = defaultdict(deque)          # model -> deque[[t, tokens]] over last 60 s
        self.day_req, self.day_tok, self.inflight = Counter(), Counter(), Counter()
        self.blocked, self.day_blocked = {}, {}
        self.last429, self.last_inc = {}, {}
        self.queue, self.seq, self.peak_queue = [], 0, 0
        self.events = Counter()               # shed reasons, 429 scopes etc. for the write-up
        self._wake, self._task = None, None

    async def start(self):
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()

    # ---------- public API ----------
    async def acquire(self, model, est_tokens, deadline, min_service_s=0.5) -> Permit:
        lim = self.limits[model]
        est = min(est_tokens, lim.tpm * self.safety * 0.9)
        fut = asyncio.get_running_loop().create_future()
        self.seq += 1
        self.queue.append((deadline, self.seq, model, est, min_service_s, fut))
        self.peak_queue = max(self.peak_queue, len(self.queue))
        self._wake.set()
        try:
            return await fut
        except asyncio.CancelledError:
            fut.cancel()
            self._wake.set()
            raise

    def release(self, permit, actual_tokens=None, rejected=False):
        self.inflight[permit.model] -= 1
        if rejected:   # provider did not count it
            try:
                self.win[permit.model].remove(permit.entry)
            except ValueError:
                pass
            self.day_req[permit.model] -= 1
            self.day_tok[permit.model] -= permit.est
        elif actual_tokens is not None:
            self.day_tok[permit.model] += actual_tokens - permit.est
            permit.entry[1] = actual_tokens
        self._wake.set()

    def note_429(self, model, retry_after, scope):
        now = time.monotonic()
        self.events[f"429_{scope}"] += 1
        self.last429[model] = now
        if scope in ("rpd", "tpd") or retry_after >= 120:
            self.day_blocked[model] = now + retry_after
        else:
            self.blocked[model] = max(self.blocked.get(model, 0), now + retry_after)
            if scope == "rpm":   # multiplicative decrease of our belief about RPM
                n = len(self._window(model, now))
                self.cap_rpm[model] = max(1.0, min(self.cap_rpm[model], max(n, 1)) * 0.8)
        self._wake.set()

    def sync_headers(self, model, h):
        """Groq reports daily request limit in x-ratelimit-limit-requests and the per-minute token limit in
        x-ratelimit-limit-tokens (verify with probe.py!). Remaining-requests is authoritative and includes
        usage by other processes sharing the same key, which is why we trust it over our own counter."""
        lim = self.limits[model]
        try:
            if "x-ratelimit-limit-tokens" in h:
                lim.tpm = float(h["x-ratelimit-limit-tokens"])
            if "x-ratelimit-limit-requests" in h:
                lim.rpd = float(h["x-ratelimit-limit-requests"])
            if "x-ratelimit-remaining-requests" in h:
                self.day_req[model] = max(0, int(lim.rpd - float(h["x-ratelimit-remaining-requests"])))
        except ValueError:
            pass

    def mode(self, models):
        now = time.monotonic()
        worst = "normal"
        for m in models:
            l = self.limits[m]
            if self.day_blocked.get(m, 0) > now or self.day_req[m] >= l.rpd or self.day_tok[m] >= l.tpd * 0.98:
                return "extractive"
            if 1 - max(self.day_req[m] / l.rpd, self.day_tok[m] / l.tpd) < self.day_reserve:
                worst = "lean"
        if len(self.queue) > self.lean_queue:
            worst = "lean"
        return worst

    # ---------- internals ----------
    def _window(self, model, now):
        w = self.win[model]
        while w and w[0][0] < now - 60:
            w.popleft()
        return w

    def _wait_needed(self, model, est, now):
        """Seconds to wait before this call fits; None if the daily budget is gone."""
        l = self.limits[model]
        if self.day_blocked.get(model, 0) > now or self.day_req[model] >= l.rpd or self.day_tok[model] + est > l.tpd:
            return None
        w = max(0.0, self.blocked.get(model, 0) - now)
        if self.inflight[model] >= l.max_inflight:
            w = max(w, 0.05)
        win = self._window(model, now)
        cap = max(1, int(self.cap_rpm[model] * self.safety))
        k = len(win) - cap + 1
        if k > 0:
            w = max(w, win[k - 1][0] + 60 - now)
        budget, used = l.tpm * self.safety, sum(t for _, t in win)
        if used + est > budget:
            acc, freed_at = 0, None
            for t, tok in win:
                acc += tok
                if used - acc + est <= budget:
                    freed_at = t
                    break
            w = max(w, (freed_at + 60 - now) if freed_at else 60.0)
        return w

    def _grant(self, model, est, now):
        entry = [now, est]
        self.win[model].append(entry)
        self.day_req[model] += 1
        self.day_tok[model] += est
        self.inflight[model] += 1
        if now - self.last429.get(model, -1e9) > 30 and now - self.last_inc.get(model, -1e9) > 10 \
                and self.cap_rpm[model] < self.limits[model].rpm:
            self.cap_rpm[model] += 1
            self.last_inc[model] = now
        return Permit(model, entry, est)

    def _scan(self):
        now, waits, keep = time.monotonic(), [], []
        for item in sorted(self.queue, key=lambda x: (x[0], x[1])):
            deadline, _, model, est, min_s, fut = item
            if fut.done():
                continue
            if deadline - now < min_s:
                self.events["shed_deadline"] += 1
                fut.set_exception(Shed("deadline"))
                continue
            w = self._wait_needed(model, est, now)
            if w is None:
                self.events["shed_daily_budget"] += 1
                fut.set_exception(Shed("daily_budget"))
                continue
            if w > 0:
                if now + w + min_s > deadline:
                    self.events["shed_would_miss_deadline"] += 1
                    fut.set_exception(Shed("would_miss_deadline", w))
                    continue
                keep.append(item)
                waits.append(min(w, deadline - min_s - now))
                continue
            fut.set_result(self._grant(model, est, now))
        self.queue = keep
        return max(0.01, min(waits)) if waits else None

    async def _run(self):
        while True:
            self._wake.clear()
            nxt = self._scan()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=nxt)
            except TimeoutError:
                pass
