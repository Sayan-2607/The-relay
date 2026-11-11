"""Retry policy. Default ("careful"): honour retry-after, full-jitter exponential backoff, never retry past the
deadline, never retry a daily-limit 429, and cap retries with a retry budget (<= ratio of first attempts) so a
brown-out cannot turn into a retry storm. policy.naive=True is the deliberately bad version used to show
retrying making things worse: immediate retries, no governor, ignores retry-after, no budget."""
import asyncio, random, time
from dataclasses import dataclass
from .governor import Shed
from .llm import RateLimited, Transient, LLMTimeout, Fatal


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_s: float = 0.5
    cap_s: float = 8.0
    budget_ratio: float = 0.2
    naive: bool = False


class RetryBudget:
    def __init__(self, ratio=0.2, cap=10.0):
        self.ratio, self.cap, self.tokens = ratio, cap, cap

    def first_attempt(self):
        self.tokens = min(self.cap, self.tokens + self.ratio)

    def take(self):
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class GivenUp(Exception):
    def __init__(self, last=None):
        super().__init__("deadline before retry")
        self.last = last


class RetriesExhausted(Exception):
    def __init__(self, last):
        super().__init__(str(last))
        self.last = last


async def call(gov, client, model, messages, *, step, trace, deadline, policy, budget,
               max_tokens=300, json_mode=False, on_token=None):
    est = sum(len(m["content"]) for m in messages) // 4 + max_tokens
    budget.first_attempt()
    attempts = 5 if policy.naive else policy.max_attempts
    last = None
    for attempt in range(attempts):
        permit = None
        t_wait = time.monotonic()
        if gov is not None and not policy.naive:
            permit = await gov.acquire(model, est, deadline)      # may raise Shed
        t_start = time.monotonic()
        if t_start - t_wait > 0.005:
            trace.span("wait", step, t_wait, t_start)
        trace.calls += 1
        try:
            res = await client.chat(model, messages, max_tokens=max_tokens, json_mode=json_mode, on_token=on_token,
                                    timeout_s=max(0.5, deadline - t_start))
            t_end = time.monotonic()
            trace.span("work", step, t_start, t_end, f"{res.tokens_in}+{res.tokens_out}tok")
            if gov is not None:
                if permit:
                    gov.release(permit, res.tokens_in + res.tokens_out)
                gov.sync_headers(model, res.headers)
            trace.tokens_in += res.tokens_in
            trace.tokens_out += res.tokens_out
            return res
        except RateLimited as e:
            t_end = time.monotonic()
            trace.span("retry", step, t_start, t_end, f"429 {e.scope}")
            if gov is not None:
                if permit:
                    gov.release(permit, rejected=True)
                gov.note_429(model, e.retry_after, e.scope)
            if not policy.naive and (e.scope in ("rpd", "tpd") or e.retry_after >= 120):
                raise Shed("daily_budget")
            last, delay = e, (0.0 if policy.naive else e.retry_after + random.uniform(0, 0.25))
        except (Transient, LLMTimeout) as e:
            t_end = time.monotonic()
            trace.span("retry", step, t_start, t_end, f"{type(e).__name__}: {str(e)[:60]}")
            if gov is not None and permit:
                gov.release(permit, 0)
            last = e
            delay = 0.0 if policy.naive else random.uniform(0, min(policy.cap_s, policy.base_s * 2 ** attempt))
        except Fatal:
            if gov is not None and permit:
                gov.release(permit, 0)
            raise
        except BaseException:
            if gov is not None and permit:
                gov.release(permit, 0)
            raise
        if attempt == attempts - 1:
            break
        if not policy.naive and not budget.take():
            trace.notes.append("retry_budget_exhausted")
            raise RetriesExhausted(last)
        if time.monotonic() + delay > deadline:
            raise GivenUp(last)
        trace.retries += 1
        t_s = time.monotonic()
        await asyncio.sleep(delay)
        trace.span("retry", step, t_s, time.monotonic(), "backoff")
    raise RetriesExhausted(last)
