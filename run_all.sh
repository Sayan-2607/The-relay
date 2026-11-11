#!/usr/bin/env bash
# Suggested order. Read the comments: the real-service steps spend your shared daily quota.
set -e
export GROQ_API_KEY="${GROQ_API_KEY:?set your own free-tier key; never commit it}"

python fetch_papers.py                                    # 1. papers at the exact versions (needs internet)
python tests/smoke.py                                     # 2. sanity check, no key needed
python probe.py --max-requests 150                        # 3. REAL: latency, limits, 429 shape  -> results/probe_raw.jsonl
python analyze.py --probe                                 #    evidence for Table 1
python calibrate.py                                       # 4. fit stand-in + limits.json from the probe

python -m relay.standin --port 8099 &                     # 5. stand-in (calibrated)
sleep 1
for sc in steady burst overload rpm_halved daily_exhausted tool_slow; do
  python loadtest.py --scenario $sc --source standin --rate 20
done
python loadtest.py --scenario steady --source standin --rate 20 --naive-retry --no-governor --tag naive   # retries making it worse
python loadtest.py --scenario overload --source standin --rate 20 --stream-first --tag streamfirst         # cost of verify-before-stream

python loadtest.py --scenario steady --source real --rate 6 --duration 60 --max-requests 20              # 6. REAL, small, capped
python quality.py --source real --settings w120_k6 w250_k4 w250_k4_rr w400_k3 --concurrency 2            # 7. REAL quality (4 settings x 20 q x ~4 calls)
python analyze.py                                         # 8. tables from results/results.csv
python analyze.py --slowest overload                      #    section 4 split
kill %1
