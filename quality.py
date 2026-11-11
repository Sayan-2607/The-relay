"""Answer-quality + retrieval-setting comparison (Table 3). Low concurrency, so limits do not distort quality.
  python quality.py --source real --settings w120_k6 w250_k4 w250_k4_rr w400_k3 --concurrency 2
Writes rows to results.csv with scenario quality-<setting> and results/grading_review.csv for hand-checking."""
import argparse, asyncio, csv, json, os
from relay.runner import System, open_outputs, run_one


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "standin"], required=True)
    ap.add_argument("--settings", nargs="+", default=["w250_k4"])
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only first N questions (budget!)")
    a = ap.parse_args()
    qs = json.load(open(a.questions))[: a.limit]
    f, w, d = open_outputs(a.out)
    rev_path = os.path.join(a.out, "grading_review.csv")
    new = not os.path.exists(rev_path)
    rev = open(rev_path, "a", newline="")
    rw = csv.writer(rev)
    if new:
        rw.writerow(["request_id", "setting", "qid", "type", "question", "auto_grade", "human_grade", "answer"])
    for name in a.settings:
        system = System(a.source, name)
        await system.start()
        sem = asyncio.Semaphore(a.concurrency)

        async def one(i, q):
            async with sem:
                rid = f"quality-{name}-{q['id']}"
                row, ans = await run_one(system, rid, f"quality-{name}", q["q"], grader_q=q, writer=w, detail=d)
                rw.writerow([rid, name, q["id"], q["type"], q["q"], row["answer_grade"], "", (ans.text if ans else "")[:600]])
                return row
        rows = await asyncio.gather(*[one(i, q) for i, q in enumerate(qs)])
        from collections import Counter
        print(name, dict(Counter(r["answer_grade"] for r in rows)), "tokens/answer",
              round(sum(r["tokens_in"] + r["tokens_out"] for r in rows) / len(rows)))
        await system.stop()
    f.close(); d.close(); rev.close()

asyncio.run(main())
