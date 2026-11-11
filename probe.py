"""Probe the REAL service (Groq). Spends real quota, so it has a hard request cap. Raw output: results/probe_raw.jsonl
  export GROQ_API_KEY=...        (never commit it)
  python probe.py --model llama-3.1-8b-instant --phases latency_short latency_long concurrency rpm_burst --max-requests 150
Phases:
  latency_short  N sequential tiny requests               -> TTFT / decode-speed distribution, tail
  latency_long   N sequential ~1.5k-token prompts         -> prefill cost per input token (429s here teach you TPM)
  concurrency    c = 1,2,4,8 simultaneous requests        -> does latency rise with concurrency / is there an in-flight cap?
  rpm_burst      fire requests as fast as possible        -> first 429: which limit, what message/headers, how long to recover
Everything (headers, 429 bodies, retry-after) is logged so Table 1 can be filled from evidence, not memory."""
import argparse, asyncio, json, os, time
from relay.config import endpoint
from relay.llm import LLMClient, RateLimited, Transient, LLMTimeout, Fatal

LOREM = ("Transformers use attention to relate positions of a sequence. Neural audio codecs compress speech into discrete tokens. "
         "Hierarchical models learn to chunk raw bytes into units. ") * 40


class Probe:
    def __init__(self, a):
        self.a, self.client, self.n = a, LLMClient(endpoint("real")), 0
        os.makedirs(a.out, exist_ok=True)
        self.f = open(os.path.join(a.out, "probe_raw.jsonl"), "a")

    async def one(self, phase, prompt, max_tokens, **extra):
        if self.n >= self.a.max_requests:
            raise SystemExit(f"request cap {self.a.max_requests} reached; stopping to protect your daily budget")
        self.n += 1
        rec = {"phase": phase, "model": self.a.model, "t": time.time(), **extra}
        try:
            r = await self.client.chat(self.a.model, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
            rec.update(status=200, ttft_ms=r.first_token_ms, total_ms=r.total_ms, tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                       usage_estimated=r.usage_estimated, headers=r.headers)
        except RateLimited as e:
            rec.update(status=429, scope=e.scope, retry_after=e.retry_after, message=e.message, headers=e.headers)
        except (Transient, LLMTimeout, Fatal) as e:
            rec.update(status=-1, error=f"{type(e).__name__}: {e}")
        self.f.write(json.dumps(rec) + "\n"); self.f.flush()
        return rec

    async def paced(self, rec_fn):
        """Run one request; on 429 sleep retry-after (the sleep itself is data: how long recovery takes)."""
        rec = await rec_fn()
        while rec["status"] == 429:
            wait = min(rec["retry_after"], 90) + 0.5
            print(f"  429 ({rec['scope']}), retry-after={rec['retry_after']}s; sleeping {wait:.0f}s")
            if rec["scope"] in ("rpd", "tpd"):
                raise SystemExit("daily limit hit: stop. Record this in Table 1 and resume tomorrow.")
            await asyncio.sleep(wait)
            rec = await rec_fn()
        return rec

    async def latency_short(self):
        for i in range(self.a.n):
            await self.paced(lambda: self.one("latency_short", f"Say the word 'ok' and one fact about attention. #{i}", 60, i=i))
            await asyncio.sleep(0.3)

    async def latency_long(self):
        for i in range(max(8, self.a.n // 2)):
            await self.paced(lambda: self.one("latency_long", LOREM[: 5000 + 400 * (i % 5)] + "\nSummarize in one sentence.", 100, i=i))

    async def concurrency(self):
        for c in (1, 2, 4, 8):
            for rep in range(3):
                recs = await asyncio.gather(*[self.one("concurrency", f"Name a color. #{k}", 40, c=c, rep=rep) for k in range(c)])
                print(f"  c={c}: statuses {[r['status'] for r in recs]}")
                await asyncio.sleep(2)

    async def rpm_burst(self):
        print("  burst: firing tiny requests until the first 429 (cap", self.a.burst, ")")
        first = None
        for batch in range(0, self.a.burst, 10):
            recs = await asyncio.gather(*[self.one("rpm_burst", "hi", 1, k=batch + k) for k in range(10)])
            hit = [r for r in recs if r["status"] == 429]
            if hit:
                first = hit[0]
                break
        if not first:
            print("  no 429 within burst cap: RPM is above", self.a.burst, "per minute or not enforced for this model")
            return
        print("  first 429:", first["scope"], first["message"][:160], first["headers"])
        t0 = time.time()
        while time.time() - t0 < 120:
            await asyncio.sleep(1.0)
            r = await self.one("rpm_recovery", "hi", 1, since_429_s=round(time.time() - t0, 1))
            if r["status"] == 200:
                print(f"  recovered after {time.time() - t0:.1f}s (retry-after said {first['retry_after']}s)")
                break


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--phases", nargs="+", default=["latency_short", "concurrency", "latency_long", "rpm_burst"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--burst", type=int, default=60)
    ap.add_argument("--max-requests", type=int, default=150)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    p = Probe(a)
    for ph in a.phases:
        print("phase", ph)
        await getattr(p, ph)()
    print("requests used:", p.n, "-> now read results/probe_raw.jsonl headers (x-ratelimit-*) and run calibrate.py")

asyncio.run(main())
