"""Load scenarios against the real service or the stand-in. Appends to results/results.csv (+ detail jsonl).
  python loadtest.py --scenario steady   --source standin --rate 20
  python loadtest.py --scenario burst    --source standin --rate 20
  python loadtest.py --scenario overload --source standin --rate 20
  python loadtest.py --scenario rpm_halved      --source standin --rate 20   # stand-in RPM halved, governor NOT told
  python loadtest.py --scenario daily_exhausted --source standin --rate 20   # daily budget runs out mid-run
  python loadtest.py --scenario steady --source real --rate 6 --duration 60 --max-requests 25   # protect your daily budget
Naive-retry demo: add --naive-retry --no-governor (and compare to the same scenario without them)."""
import argparse, asyncio, json, os, random, time, urllib.request
from relay.config import FAST_MODEL, STRONG_MODEL
from relay.runner import System, open_outputs, run_one

SCHEDULES = {   # list of (seconds, requests_per_minute) given base rate r
    "steady": lambda r: [(60, r)],
    "burst": lambda r: [(15, r * 0.5), (10, r * 6), (35, r * 0.5)],
    "overload": lambda r: [(60, r * 4)],
    "rpm_halved": lambda r: [(60, r)],
    "daily_exhausted": lambda r: [(60, r)],
    "tool_slow": lambda r: [(60, r)],
}


def admin(url, payload):
    base = url.replace("/v1", "")
    req = urllib.request.Request(base + "/admin/reset", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=10))


def admin_stats(url):
    return json.load(urllib.request.urlopen(url.replace("/v1", "") + "/admin/stats", timeout=10))


def prepare_standin(a, url):
    stats = admin_stats(url)["config"]["models"]
    a._orig = {f"models/{m}/{k}": c[k] for m, c in stats.items() for k in ("rpm", "rpd")}
    payload = {"set": {}, "reset": True}
    if a.scenario == "rpm_halved":
        for m, c in stats.items():
            payload["set"][f"models/{m}/rpm"] = max(1, int(c["rpm"]) // 2)
    if a.scenario == "daily_exhausted":
        # start healthy, flip to exhausted part-way through: leave ~20 requests of daily budget
        payload["day_used_req"] = {m: int(c["rpd"]) - a.daily_left for m, c in stats.items()}
    for kv in a.set or []:
        k, v = kv.split("=")
        payload["set"][k] = json.loads(v)
    admin(url, payload)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCHEDULES))
    ap.add_argument("--source", choices=["real", "standin"], required=True)
    ap.add_argument("--rate", type=float, default=20, help="base offered rate, requests/min")
    ap.add_argument("--duration", type=float, default=None, help="scale schedule to this many seconds")
    ap.add_argument("--setting", default="w250_k4")
    ap.add_argument("--patience", type=float, default=25.0)
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-requests", type=int, default=100000, help="hard cap: protects the real daily budget")
    ap.add_argument("--daily-left", type=int, default=12)
    ap.add_argument("--set", nargs="*", help="stand-in overrides, e.g. models/llama-3.1-8b-instant/rpm=15")
    ap.add_argument("--naive-retry", action="store_true")
    ap.add_argument("--no-governor", action="store_true")
    ap.add_argument("--stream-first", action="store_true", help="stream the draft before verification (unsafe variant)")
    ap.add_argument("--tag", default="", help="suffix for the scenario name in results.csv")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    url = os.environ.get("STANDIN_URL", "http://127.0.0.1:8099/v1")
    if a.source == "standin":
        prepare_standin(a, url)
    tool_kw = {"slow_rate": 0.3, "median_ms": 400} if a.scenario == "tool_slow" else {}
    system = System(a.source, a.setting, patience_s=a.patience, naive_retry=a.naive_retry, tool_kwargs=tool_kw,
                    use_governor=not a.no_governor, verify_before_stream=not a.stream_first)
    await system.start()
    questions = [q["q"] for q in json.load(open(a.questions))]
    f, w, d = open_outputs(a.out)
    name = a.scenario + (f"-{a.tag}" if a.tag else "")
    sched = SCHEDULES[a.scenario](a.rate)
    if a.duration:
        k = a.duration / sum(s for s, _ in sched)
        sched = [(s * k, r) for s, r in sched]
    tasks, t_start, i = [], time.monotonic(), 0
    peak = 0
    for seg_s, rate in sched:
        seg_end = time.monotonic() + seg_s
        while time.monotonic() < seg_end and i < a.max_requests:
            tasks.append(asyncio.create_task(run_one(system, f"{name}-{i:05d}", name, questions[i % len(questions)], writer=w, detail=d)))
            i += 1
            await asyncio.sleep(rng.expovariate(rate / 60.0))
    sent_for = time.monotonic() - t_start
    done = await asyncio.gather(*tasks)
    f.flush()
    peak = system.gov.peak_queue if system.gov else 0
    meta_p = os.path.join(a.out, "scenario_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta[name] = {"source": a.source, "requests": len(done), "send_window_s": round(sent_for, 1), "peak_queue": peak,
                  "governor_events": dict(system.gov.events) if system.gov else {}, "retry_budget_left": system.pipe.budget.tokens,
                  "tool_calls": system.tools.calls, "tool_timeouts": system.tools.timeouts, "args": {k: v for k, v in vars(a).items() if not k.startswith("_")}}
    json.dump(meta, open(meta_p, "w"), indent=2)
    from collections import Counter
    print(name, a.source, "sent", len(done), "in", round(sent_for, 1), "s; statuses", dict(Counter(r["status"] for r, _ in done)),
          "peak queue", peak, "events", meta[name]["governor_events"])
    await system.stop()
    f.close(); d.close()
    if a.source == "standin" and getattr(a, "_orig", None):
        admin(url, {"set": a._orig, "reset": True})     # restore the stand-in's calibrated limits

asyncio.run(main())
