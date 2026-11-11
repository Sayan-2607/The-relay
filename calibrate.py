"""Fit stand-in parameters from results/probe_raw.jsonl -> standin_config.json (+ limits.json, calibration_report.md).
Run once per model you probed. Anything it cannot estimate keeps the default and is listed as UNCALIBRATED."""
import json, math, statistics as st, sys
from relay.standin import DEFAULT
import copy

rows = [json.loads(l) for l in open("results/probe_raw.jsonl")]
cfg, report, limits = copy.deepcopy(DEFAULT), [], {}
models = sorted({r["model"] for r in rows})


def lognorm(xs):
    ls = [math.log(max(x, 1e-6)) for x in xs]
    return math.exp(st.median(ls)), (st.pstdev(ls) if len(ls) > 1 else 0.3)


for m in models:
    ok = [r for r in rows if r["model"] == m and r.get("status") == 200]
    short = [r for r in ok if r["phase"] == "latency_short"]
    longr = [r for r in ok if r["phase"] == "latency_long"]
    mc = cfg["models"].setdefault(m, copy.deepcopy(next(iter(DEFAULT["models"].values()))))
    if len(short) >= 8:
        med, sig = lognorm([r["ttft_ms"] / 1000 for r in short])
        mc["ttft_median_s"], mc["ttft_sigma"] = round(med, 4), round(min(sig, 1.0), 3)
        tps = [r["tokens_out"] / max(1e-3, (r["total_ms"] - r["ttft_ms"]) / 1000) for r in short if r["tokens_out"] > 8 and r["total_ms"] > r["ttft_ms"] + 20]
        if tps:
            tm, ts = lognorm(tps)
            mc["tps_median"], mc["tps_sigma"] = round(tm, 1), round(min(ts, 0.8), 3)
        decode_note = f"{mc['tps_median']} tok/s" if tps else "UNCALIBRATED (no measurable post-first-token interval)"
        report.append(f"- {m}: TTFT median {med*1000:.0f} ms, log-sigma {sig:.2f} (n={len(short)}); decode {decode_note}")
    else:
        report.append(f"- {m}: UNCALIBRATED latency (need >=8 latency_short samples)")
    pts = [(r["tokens_in"], r["ttft_ms"]) for r in short + longr if r["tokens_in"]]
    if len(pts) >= 6 and len({p[0] for p in pts}) > 2:
        mx, my = st.mean(p[0] for p in pts), st.mean(p[1] for p in pts)
        var = sum((p[0] - mx) ** 2 for p in pts)
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / var if var else 0
        mc["prefill_ms_per_tok"] = round(max(0.0, slope), 4)
        report.append(f"  prefill slope {slope:.3f} ms per input token (n={len(pts)})")
    # tail: succeeds-slowly fraction
    tot = [r["total_ms"] for r in short + longr]
    if len(tot) >= 10:
        med = st.median(tot)
        slow = [t for t in tot if t > 3 * med]
        cfg["slow_rate"] = round(len(slow) / len(tot), 4)
        cfg["slow_factor"] = round(st.median(slow) / med, 1) if slow else cfg["slow_factor"]
        report.append(f"  tail: {len(slow)}/{len(tot)} requests > 3x median total ({med:.0f} ms)")
    # limits from headers
    hs = [r["headers"] for r in ok if r.get("headers")]
    if hs:
        h = hs[0]
        if "x-ratelimit-limit-requests" in h:
            mc["rpd"] = float(h["x-ratelimit-limit-requests"])
        if "x-ratelimit-limit-tokens" in h:
            mc["tpm"] = float(h["x-ratelimit-limit-tokens"])
        report.append(f"  headers seen: {h}")
    burst = [r for r in rows if r["model"] == m and r["phase"] == "rpm_burst"]
    n429 = next((i for i, r in enumerate(burst) if r["status"] == 429), None)
    if n429 is not None and burst[n429].get("scope") == "rpm":
        mc["rpm"] = n429
        report.append(f"  RPM: first 429 after {n429} requests in burst")
    for r in rows:
        if r["model"] == m and r.get("status") == 429 and r.get("scope") in ("tpd", "rpd"):
            report.append(f"  daily 429 seen: {r['message'][:140]}")
    limits[m] = {k: mc[k] for k in ("rpm", "tpm", "rpd", "tpd")} | {"max_inflight": 4}
    conc = {}
    for r in rows:
        if r["model"] == m and r["phase"] == "concurrency" and r.get("status") == 200:
            conc.setdefault(r["c"], []).append(r["total_ms"])
    if 1 in conc and len(conc) > 2:
        base = st.median(conc[1])
        rat = {c: st.median(v) / base for c, v in conc.items()}
        report.append(f"  median latency vs concurrency (ratio to c=1): {rat}")
        knee = max([c for c, x in rat.items() if x < 1.25] or [1])
        over = [(c, x) for c, x in rat.items() if c > knee]
        if over:
            cfg["load_knee"] = knee
            cfg["load_alpha"] = round(st.mean((x - 1) * knee / (c - knee) for c, x in over), 3)
errs = [r for r in rows if r.get("status") == -1 or (isinstance(r.get("status"), int) and r["status"] >= 500)]
if errs:
    cfg["error_rate"] = round(len(errs) / len(rows), 4)
    report.append(f"- error rate {len(errs)}/{len(rows)}")
else:
    report.append("- error rate: UNCALIBRATED (calibration input contains only successful samples; stand-in default retained)")
json.dump(cfg, open("standin_config.json", "w"), indent=2)
json.dump(limits, open("limits.json", "w"), indent=2)
open("results/calibration_report.md", "w").write("# Calibration\n" + "\n".join(report) + "\n\nNot calibrated (defaults kept): hang_rate, verify_fail_rate, expand_rate, content, "
    "day_reset_s, mid-stream error share, TLS/connection cost, queueing inside Groq. Say so in section 2.\n")
print("\n".join(report))
print("wrote standin_config.json, limits.json, results/calibration_report.md")
