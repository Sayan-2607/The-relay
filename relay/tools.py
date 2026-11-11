"""The pipeline's tool: a (simulated) remote document-store call that returns the chunks adjacent to a hit.
It has its own latency tail, failures and timeout, so tool slowness is exercised separately from the LLM."""
import asyncio, math, random


class ToolError(Exception):
    pass


class ToolService:
    def __init__(self, median_ms=60, sigma=0.6, fail_rate=0.01, slow_rate=0.03, slow_factor=15, timeout_s=1.5, seed=7):
        self.median_ms, self.sigma, self.fail_rate = median_ms, sigma, fail_rate
        self.slow_rate, self.slow_factor, self.timeout_s = slow_rate, slow_factor, timeout_s
        self.rng = random.Random(seed)
        self.calls = self.timeouts = self.failures = 0

    async def neighbors(self, index, chunk):
        self.calls += 1
        d = self.median_ms * math.exp(self.rng.gauss(0, self.sigma)) / 1000
        if self.rng.random() < self.slow_rate:
            d *= self.slow_factor
        fail = self.rng.random() < self.fail_rate
        try:
            async with asyncio.timeout(self.timeout_s):
                await asyncio.sleep(d)
        except TimeoutError:
            self.timeouts += 1
            raise ToolError("tool timeout")
        if fail:
            self.failures += 1
            raise ToolError("tool failed")
        return index.neighbors(chunk)
