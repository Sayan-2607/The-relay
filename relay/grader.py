"""Auto-grader for the question set. A first pass only: review quality/grading_review.csv by hand and fix grades
(edit the CSV, then run analyze.py --regrade-from results/grading_review.csv)."""
import re


def grade(ans, q):
    """q: {type: answerable|unanswerable|offtopic, paper, gold_regex:[...]}"""
    if ans.status in ("timeout", "error"):
        return "refused_wrongly" if q["type"] == "answerable" else "wrong"
    if q["type"] in ("unanswerable", "offtopic"):
        return "refused_correctly" if ans.refused else "wrong"
    if ans.refused or ans.mode == "extractive":
        return "refused_wrongly"
    text = ans.text
    has_gold = all(re.search(rx, text, re.I) for rx in q.get("gold_regex", []))
    cites_right_paper = any(c.startswith(q["paper"] + " ") for c in ans.cited)
    return "correct" if has_gold and cites_right_paper else "wrong"
