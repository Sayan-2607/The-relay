# The Relay: write-up (2 pages max: keep every section tight)

## 1 What I built and why
(Design in 5 lines. The one decision that shaped it most: probably verify-before-stream. Say what it costs, with your measured TTFT difference from the `--stream-first` run.)

## 2 The real service and my stand-in
Table 1 (fill from `python analyze.py --probe` + Groq docs; give confidence per row)
| Limit | Value you found | How you found it | Confidence |
|---|---|---|---|
| Requests per minute | | probe rpm_burst: first 429 after N requests | |
| Tokens per minute | | x-ratelimit-limit-tokens / 429 message | |
| Requests per day | | x-ratelimit-limit-requests | |
| Tokens per day | | docs / 429 message / not measured | |
| Requests in flight at once | | probe concurrency phase | |
| What it returns when a limit is hit | | 429 body, retry-after, headers | |
| Time until it recovers | | probe rpm_recovery | |

Stand-in: behaviours, what `calibrate.py` fitted (`results/calibration_report.md`), where it differs most (canned content, linear load model, no TLS cost, no real queueing).

## 3 Results
Real vs stand-in per row. Paste `python analyze.py` output (Tables 2a, 2b, 3). Add: one number you could not have guessed (candidate: how few answers per minute the TPM limit allows; check your own data). Quality: n questions, tokens per answer.

## 4 Where the time goes
`python analyze.py --slowest <scenario>`: paste split (wait/work/retry), say before/after first token, and why you know (spans).

## 5 When a limit is hit
Per-minute: governor paces, EDF queue, shed if it would miss the deadline, degrade to passage-only. Per-day: lean mode below 10%, extractive at 0. Who is delayed / dropped / weaker, and why. Use `scenario_meta.json` governor_events.

## Retrieval: quality versus cost
Table 3 (>= 3 settings). Choice + reason. What retrieval does when the limit is close (lean mode: no understand call, heuristic decide).

## How this library will mislead a naive retriever
1. ... example question that goes wrong
2. ...
3. ...

## 6 Retries
Policy. One case where retrying made it worse (naive run vs careful run, with numbers).

## 7 What the person sees
Quote an actual extractive reply from `results_detail.jsonl` (daily_exhausted scenario). Say what it promises and what it doesn't.

## 8 What I'm unsure about, what I left out, what I learned
Least certain assumption + what breaks. Two things not built (e.g. embeddings, hedged requests, keep-alive pooling). One new thing, in your own words.

## Your feedback on Vaani
(Visit https://www.vaani.dev/ yourself: what stood out, what confused you, one change.)

## How I worked with the assistant
Attach the real conversation export. Then: 3 most important prompts and why; one thing it got wrong that YOU caught and how (only write what actually happened to you); one decision where you overruled it; what is distinctly yours.
