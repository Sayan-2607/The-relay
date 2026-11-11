# The Relay: write-up DRAFT
> Prose below describes what the code does. Anything marked [FILL] or "not measured" needs YOUR runs. Trim to 2 pages before submitting. Do not keep a sentence you cannot defend live.

## 1 What I built and why
A five-step pipeline in stdlib Python: understand (fast model, JSON), retrieve (BM25 over per-page chunks), decide (fast model, may call a neighbour-passage tool and loop once), draft (strong model, cited), verify (deterministic grounding check, then fast-model check, redraft once). A governor sits between every step and the provider: it paces against per-minute budgets, queues earliest-deadline-first, sheds calls that cannot finish in time, and learns the real limits from `x-ratelimit-*` headers and 429s.

The decision that shaped it most: **the draft is buffered and streamed only after verification passes**. The brief says a wrong answer is worse than a slow one, so first token is roughly the end of the pipeline, not the first model token. Cost: [FILL: TTFT p50 default vs `--stream-first` run]. Everything else (degrade instead of drop, lean mode, soft deadline) follows from "slow is acceptable, wrong is not, and nobody is turned away".

## 2 The real service and my stand-in
| Limit | Value you found | How you found it | Confidence |
|---|---|---|---|
| Requests per minute | [FILL] | first 429 in `probe.py` rpm_burst, message text | [FILL] |
| Tokens per minute | [FILL] | `x-ratelimit-limit-tokens`, 429 message | [FILL] |
| Requests per day | [FILL] | `x-ratelimit-limit-requests` | [FILL] |
| Tokens per day | [FILL or "not measured"] | docs / 429 message; I did not burn the quota to find it | [FILL] |
| Requests in flight at once | [FILL] | probe concurrency phase (c=1,2,4,8) | [FILL] |
| What it returns when a limit is hit | HTTP 429, JSON error naming the limit, `retry-after` header [FILL: confirm] | probe 429 records | [FILL] |
| Time until it recovers | [FILL] | `rpm_recovery` rows | [FILL] |

**Stand-in.** An asyncio HTTP server speaking the same streaming API. It models lognormal time-to-first-token and decode speed, a heavy tail where calls succeed slowly, 5xx errors, mid-stream drops, hangs, Groq-shaped 429s (RPM, TPM, RPD, TPD with `retry-after` and headers), a service-side concurrency cap, and a linear slowdown above a load knee. `calibrate.py` fits TTFT, decode speed, prefill cost per input token, tail rate, limits and the concurrency slope from `probe_raw.jsonl` [FILL: paste key fitted values from `results/calibration_report.md`].
**Least like the real thing:** replies are canned, so it says nothing about answer quality; the load slowdown is a straight line while real queueing inside the provider is unknown; hang and mid-stream-error rates and the day-reset time are guesses; my client opens a new TLS connection per request, which inflates real latency by one handshake.

## 3 Results
Real vs stand-in per row. Paste output of `python analyze.py` here, unchanged:
- Table 2a (latency): [FILL from analyze.py]
- Table 2b (outcomes): [FILL from analyze.py]
- Table 3 (retrieval settings): [FILL from analyze.py after `quality.py --source real`]

Status meanings: ok = full-quality answer; rate_limited = person got the passage-only reply because of limits; timeout = gave up or a call timed out; error = other failure. Percentiles are nearest-rank over ok requests; n is shown beside each. Peak queue comes from `scenario_meta.json`, not the CSV.
Number I could not have guessed: [FILL: pick from your data. Strong candidate to check: answers per minute the token-per-minute limit allows once each answer costs about four calls plus reserved `max_tokens`.]
Quality: [FILL: n questions, correct/wrong/refused after your hand corrections in `grading_review.csv`], tokens per answer [FILL].

## 4 Where the time goes
[FILL: run `python analyze.py --slowest <scenario>` and paste the split.] How to read it: spans are sequential, so wait + work + retry should sum to total within a few ms ("other"). Because the answer is released only after verification, every span lands **before** the first token by design; the evidence is the span end times against `first_token_ms`. State which of wait, work or retry dominated in your slow case.

## 5 When a limit is hit
**Per-minute.** The governor holds calls until request and token windows have room (90% safety margin, learned RPM via AIMD). Waiting calls are ordered by deadline. A call that cannot fit before its deadline is shed, and that request is answered passage-only. A 429 that still slips through blocks that model for `retry-after`. *Delayed:* everyone in the queue, oldest deadline first. *Dropped:* nobody, unless they gave up. *Weaker answer:* requests the governor sheds.
**Per-day.** Below 10% of daily budget the system goes *lean* (skip understand and the LLM verify, heuristic decide; about two calls per answer instead of four; deterministic citation check still runs). At zero it goes *extractive*: no model calls, cited passages only, nothing spent. Why: stretching the remaining budget across more people beats giving a few people the best answer. Cost: lean mode refuses more [FILL: measured].

## Retrieval: quality versus cost
Settings tried: w120_k6, w250_k4, w250_k4_rr (lexical re-rank from 12 candidates), w400_k3. [FILL: Table 3.] Choice: [FILL: one setting + reason from your numbers]. When the limit is close: lean mode drops the understand call (no paper filter or query rewrite) and uses a coverage heuristic instead of the decide call, so retrieval stays local and costs zero tokens.

## How this library will mislead a naive retriever
Verify each with a question of your own before keeping it.
1. **Same word, different meaning.** "Chunk" in the dynamic-chunking paper is a learned boundary over bytes, not a text passage. *Example:* "How does the model chunk its input?" retrieves chunking passages but may mix them with other papers' sense of the word [FILL: test].
2. **Shared vocabulary across papers.** "Attention", "transformer", "codebook" appear in all three. *Example:* "How many layers does the Transformer have?" can return Moshi's or H-Net's layer counts instead of the 6-layer base model [FILL: test].
3. **PDF extraction damage.** Two-column layout, tables and equations extract in the wrong order or split across pages, and my chunks never cross a page. *Example:* a BLEU or hyperparameter value in a table whose row labels land in a different chunk [FILL: test with a real table question].

## 6 Retries
Policy: up to 3 attempts; honour `retry-after` plus up to 250 ms jitter; full-jitter exponential backoff for 5xx and timeouts; never retry a daily-limit 429; never retry past the deadline; retries capped at 20% of first attempts. Case where retrying made it worse: [FILL: compare `--naive-retry --no-governor --tag naive` with the careful run: retries per request, rate-limited %, p95]. Mechanism to explain: immediate retries add load while the provider is already refusing, so each refusal creates more refused traffic.

## 7 What the person sees
At 3pm with the daily budget gone, the reply is generated with zero model calls:
> The full assistant is at capacity, so this is a passage-only reply, not a written answer. Closest passages from the papers:
> - [two cited passages, each trimmed to its first two sentences, with `[paper pN]`]
> (daily model budget used up; full answers resume after reset)

It arrives in milliseconds. It is true to the papers, cited, and says what it is. It does **not** answer in prose, does not give a reset time (the code knows the `retry-after` but does not show it; showing it is a small change), and does not queue the question for later.

## 8 What I'm unsure about, what I left out, what I learned
**Least certain assumption:** that Groq counts reserved `max_tokens` toward the per-minute token limit and that the header meanings are as I assumed. If wrong, the governor is too cautious (wasting capacity) or too optimistic (more 429s). [FILL: what the probe showed.]
**Not built:** embedding retrieval or a cross-encoder re-ranker (BM25 plus a lexical re-rank only); hedged requests and connection pooling. [FILL: change if you built others.]
**New to me:** [FILL in your own words. Write about something you actually learned, e.g. how streaming makes first-token and total time different quantities, or how a retry budget stops retry storms.]

## Your feedback on Vaani
[FILL: visit https://www.vaani.dev/ yourself. A few honest lines: what stood out, what confused you, one change.]

## How I worked with the assistant
[FILL: attach your real, uncleaned conversation export.]
Three prompts that mattered most and why: [FILL]. One thing it got wrong that I caught, and how: [FILL, only what really happened]. One decision where I overruled it: [FILL]. Distinctly mine: [FILL: your measurements, the settings you chose and why, the questions you wrote].
