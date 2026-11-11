"""End-to-end smoke test on tiny fixtures against the in-process stand-in. No network, no key needed."""
import asyncio, copy, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["STANDIN_URL"] = "http://127.0.0.1:18099/v1"
from relay import standin
from relay.runner import System, run_one, open_outputs
from relay.llm import parse_duration, detect_scope


async def main():
    assert parse_duration("try again in 14m20.5s") == 14 * 60 + 20.5 and parse_duration("120ms") == 0.12
    assert detect_scope("on tokens per day (TPD): Limit") == "tpd"
    cfg = copy.deepcopy(standin.DEFAULT)
    cfg.update(error_rate=0.0, hang_rate=0.0, slow_rate=0.0)
    s, server = await standin.serve(cfg, port=18099)
    sysm = System("standin", setting="w120_k6", papers_dir="tests/fixtures", limits_path="none.json")
    await sysm.start()
    out = tempfile.mkdtemp()
    f, w, d = open_outputs(out)
    qs = ["How many layers does the encoder of the Transformer have?",
          "What frame rate does Mimi operate at?",
          "Who won the world cup?",
          "What is the carbon footprint of training Moshi?"]
    rows = await asyncio.gather(*[run_one(sysm, f"t{i}", "smoke", q, writer=w, detail=d) for i, q in enumerate(qs)])
    for (row, ans), q in zip(rows, qs):
        print(q, "->", row["status"], row["first_token_ms"], row["total_ms"], "refused" if ans.refused else ans.mode, "|", ans.text[:110].replace("\n", " "))
    assert all(r["status"] == "ok" for r, _ in rows)
    assert rows[2][1].refused
    # daily budget gone -> extractive, still a reply
    await asyncio.get_running_loop().run_in_executor(None, lambda: None)
    sysm.gov.day_req[sysm.cfg.fast_model] = 10 ** 9
    row, ans = await run_one(sysm, "t9", "smoke-daily", qs[0], writer=w, detail=d)
    print("daily-exhausted ->", row["status"], ans.mode, ans.text[:90].replace("\n", " "))
    assert ans.mode == "extractive" and row["status"] == "rate_limited"
    f.close(); d.close()
    await sysm.stop(); server.close()
    print("SMOKE OK", out)

asyncio.run(main())
