"""Compute every number in the write-up tables from results/results.csv (+ probe_raw.jsonl for Table 1 evidence).
Percentiles: nearest-rank, over requests with status == ok (first-token and total). Percentages use ALL requests sent.
  python analyze.py                      # Tables 2a, 2b, 3
  python analyze.py --probe              # evidence for Table 1
  python analyze.py --slowest steady     # latency split (wait/work/retry) of the slowest request in a scenario
  python analyze.py --regrade-from results/grading_review.csv   # apply your hand grades to results.csv"""
import argparse, csv, json, math, os
from collections import defaultdict
from datetime import datetime
from relay.config import PRICES, STRONG_MODEL


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


def trio(xs):
    return " / ".join(f1(pct(xs, p)) for p in (50, 95, 99))


def duo(xs):
    return " / ".join(f1(pct(xs, p)) for p in (50, 95))


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def load(path):
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--slowest")
    ap.add_argument("--regrade-from")
    a = ap.parse_args()
    d = os.path.dirname(a.csv)
    if a.probe:
        return probe(os.path.join(d, "probe_raw.jsonl"))
    if a.regrade_from:
        return regrade(a.csv, a.regrade_from)
    if a.slowest:
        return slowest(os.path.join(d, "results_detail.jsonl"), a.slowest)
    rows = load(a.csv)
    meta = json.load(open(os.path.join(d, "scenario_meta.json"))) if os.path.exists(os.path.join(d, "scenario_meta.json")) else {}
    pin, pout = PRICES.get(STRONG_MODEL, (0, 0))
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    load_sc = {k: v for k, v in by.items() if not k.startswith("quality-")}
    print("\nTable 2a: latency (ms). p50 / p95 / p99 over status=ok requests\n")
    print("| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total p50 / p95 / p99 (ms) |")
    print("|---|---|---|---|---|---|---|")
    for k, v in load_sc.items():
        starts = [ts(r["start_time"]) for r in v]
        dur = max(s + float(r["total_ms"]) / 1000 for s, r in zip(starts, v)) - min(starts)
        span = max(starts) - min(starts)
        rate = len(v) / span * 60 if span > 0 else float("nan")
        ok = [r for r in v if r["status"] == "ok"]
        ft = [float(r["first_token_ms"]) for r in ok if r["first_token_ms"]]
        tt = [float(r["total_ms"]) for r in ok]
        print(f"| {k} | {v[0]['source']} | {len(v)} | {dur:.1f} | {rate:.1f} | {trio(ft)} (n={len(ft)}) | {trio(tt)} (n={len(tt)}) |")
    print("\nTable 2b: outcomes. ok = full-quality answer; rate_limited = served the degraded passage-only reply because of limits;"
          " timeout = person gave up; error = other failure\n")
    print("| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Error % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers (USD, list price upper bound) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in load_sc.items():
        n = len(v)
        c = lambda s: 100 * sum(r["status"] == s for r in v) / n
        toks = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in v if int(r["tokens_in"]) + int(r["tokens_out"]) > 0]
        cost = [(int(r["tokens_in"]) * pin + int(r["tokens_out"]) * pout) / 1e6 for r in v]
        print(f"| {k} | {c('ok'):.1f} | {c('rate_limited'):.1f} | {c('timeout'):.1f} | {c('error'):.1f} | "
              f"{sum(int(r['retries']) for r in v) / n:.2f} | {meta.get(k, {}).get('peak_queue', 'not measured')} | "
              f"{(sum(toks) / len(toks)) if toks else 0:.0f} | {1000 * sum(cost) / n:.2f} |")
    q = {k: v for k, v in by.items() if k.startswith("quality-")}
    if q:
        print("\nTable 3: retrieval settings (quality scenarios)\n")
        print("| Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Wrongly refused % | Tokens per answer | First token p50 / p95 (ms) | Total p50 / p95 (ms) | Source |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for k, v in q.items():
            n = len(v)
            g = lambda s: 100 * sum(r["answer_grade"] == s for r in v) / n
            ok = [r for r in v if r["status"] == "ok"]
            toks = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in v]
            print(f"| {k[8:]} | {n} | {g('correct'):.1f} | {g('wrong'):.1f} | {g('refused_correctly'):.1f} | {g('refused_wrongly'):.1f} | "
                  f"{sum(toks) / n:.0f} | {duo([float(r['first_token_ms']) for r in ok if r['first_token_ms']])} | "
                  f"{duo([float(r['total_ms']) for r in ok])} | {v[0]['source']} |")
    print("\nPeak queue / governor events come from scenario_meta.json (not derivable from the CSV alone); say so in the write-up.")


def probe(path):
    rows = [json.loads(l) for l in open(path)]
    seen, firsts = {}, []
    for r in rows:
        for k, v in (r.get("headers") or {}).items():
            seen.setdefault(k, set()).add(v)
        if r.get("status") == 429:
            firsts.append(r)
    print("Rate-limit headers seen (value samples):")
    for k, v in sorted(seen.items()):
        print(f"  {k}: {sorted(v)[:4]}")
    print(f"\n429 responses: {len(firsts)}")
    for r in firsts[:5]:
        print(f"  [{r['phase']}] scope={r['scope']} retry_after={r['retry_after']}  {r['message'][:200]}")
    rec = [r for r in rows if r["phase"] == "rpm_recovery" and r["status"] == 200]
    if rec:
        print("\nfirst success after burst-429 at", rec[0]["since_429_s"], "s")


def regrade(csv_path, review):
    human = {r["request_id"]: r["human_grade"] for r in csv.DictReader(open(review)) if r["human_grade"].strip()}
    rows = load(csv_path)
    for r in rows:
        if r["request_id"] in human:
            r["answer_grade"] = human[r["request_id"]]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("applied", len(human), "hand grades")


def slowest(path, scenario):
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r["scenario"].startswith(scenario) and r["split_ms"]["total"]]
    r = max(recs, key=lambda r: r["split_ms"]["total"])
    print(f"slowest request in {scenario}: {r['request_id']} mode={r['mode']} reason={r['reason']} first_token_ms={r['first_token_ms']}")
    print("split (ms):", r["split_ms"])
    ft = r["first_token_ms"] or r["split_ms"]["total"]
    for sp in r["spans"]:
        side = "BEFORE first token" if sp["end_ms"] <= ft + 1 else "after first token"
        print(f"  {sp['kind']:5} {sp['step']:18} {sp['start_ms']:9.1f} -> {sp['end_ms']:9.1f}  {side}  {sp['note']}")
    print("notes:", r["notes"])


main()
