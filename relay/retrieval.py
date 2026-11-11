"""Chunking + BM25 (stdlib only) + optional lexical re-rank. Chunks never cross a page, so every citation
is an exact page ([attn p5]). Cost of that choice: a fact split across a page break is split in two chunks."""
import math, os, re
from collections import Counter, defaultdict
from dataclasses import dataclass

PAPERS = {
    "attn": "Attention Is All You Need (arXiv 1706.03762v7)",
    "moshi": "Moshi (arXiv 2410.00037v1)",
    "hnet": "Dynamic Chunking / H-Net (arXiv 2507.07955v2)",
}
STOP = set("a an the of in on to and or is are was were be by for with as at from that this it its which what how why "
           "do does did we our can used use paper papers model models report reported".split())
SEC = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9 ,:\-\(\)/]{2,80})$")
CITE = re.compile(r"\[(attn|moshi|hnet)\s+p(\d+)[^\]]*\]")


def toks(s):
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", s.lower())]


def content(s):
    return {w for w in toks(s) if w not in STOP}


@dataclass
class Chunk:
    id: int
    paper: str
    page: int
    section: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    chunk_words: int
    overlap: int
    k: int
    rerank: str = "none"        # none | lexical
    candidates: int = 0         # fetched before re-rank (0 -> 3*k)


SETTINGS = {s.name: s for s in [
    RetrievalSetting("w120_k6", 120, 20, 6),
    RetrievalSetting("w250_k4", 250, 40, 4),
    RetrievalSetting("w250_k4_rr", 250, 40, 4, "lexical", 12),
    RetrievalSetting("w400_k3", 400, 60, 3),
]}


def load_papers(papers_dir):
    out = {}
    for key in PAPERS:
        path = os.path.join(papers_dir, f"{key}.txt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run: python fetch_papers.py")
        out[key] = open(path, encoding="utf-8").read().split("\f")
    return out


def make_chunks(papers, size, overlap):
    chunks, cid = [], 0
    for key, pages in papers.items():
        section = ""
        for pno, page in enumerate(pages, 1):
            words = []
            for line in page.splitlines():
                m = SEC.match(line.strip())
                if m:
                    section = f"{m[1]} {m[2]}"
                for w in line.split():
                    words.append((w, section))
            step = max(1, size - overlap)
            for i in range(0, max(1, len(words)), step):
                win = words[i:i + size]
                if len(win) < 8:
                    continue
                chunks.append(Chunk(cid, key, pno, win[0][1], " ".join(w for w, _ in win)))
                cid += 1
                if i + size >= len(words):
                    break
    return chunks


class Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.post, self.dl = defaultdict(list), []
        for c in chunks:
            tf = Counter(toks(c.text))
            self.dl.append(sum(tf.values()))
            for t, n in tf.items():
                self.post[t].append((c.id, n))
        self.avg = sum(self.dl) / max(1, len(self.dl))
        self.by_id = {c.id: c for c in chunks}

    def bm25(self, query, papers=None):
        scores, N = defaultdict(float), len(self.chunks)
        for t in content(query):
            plist = self.post.get(t, [])
            idf = math.log(1 + (N - len(plist) + .5) / (len(plist) + .5))
            for cid, tf in plist:
                if papers and self.by_id[cid].paper not in papers:
                    continue
                d = self.dl[cid]
                scores[cid] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * d / self.avg))
        return scores

    def search(self, query, setting, papers=None, k=None):
        k = k or setting.k
        scores = self.bm25(query, papers)
        n = (setting.candidates or 3 * k) if setting.rerank != "none" else k
        top = sorted(scores.items(), key=lambda x: -x[1])[:n]
        hits = [Hit(self.by_id[cid], s) for cid, s in top]
        if setting.rerank == "lexical" and hits:
            hits = self._lexical_rerank(query, hits)
        return hits[:k]

    def _lexical_rerank(self, query, hits):
        qt = [t for t in toks(query) if t not in STOP]
        bigrams = {(a, b) for a, b in zip(qt, qt[1:])}
        mx = max(h.score for h in hits) or 1.0
        out = []
        for h in hits:
            ct = toks(h.chunk.text)
            cb = set(zip(ct, ct[1:]))
            bi = len(bigrams & cb) / max(1, len(bigrams))
            sec = len(set(qt) & content(h.chunk.section)) / max(1, len(set(qt)))
            out.append(Hit(h.chunk, h.score / mx + 0.5 * bi + 0.3 * sec))
        return sorted(out, key=lambda h: -h.score)

    def neighbors(self, chunk):
        return [self.by_id[i] for i in (chunk.id - 1, chunk.id + 1)
                if i in self.by_id and self.by_id[i].paper == chunk.paper]


def coverage(query, hits):
    q = content(query)
    if not q:
        return 0.0
    seen = set()
    for h in hits:
        seen |= content(h.chunk.text)
    return len(q & seen) / len(q)


def grounding_check(text, hits):
    """Deterministic, zero-token check: every sentence cites a retrieved page, every number appears in the cited
    text, and enough of the sentence's content words appear in it. Returns (ok, problems)."""
    valid = {}
    for h in hits:
        valid[(h.chunk.paper, h.chunk.page)] = valid.get((h.chunk.paper, h.chunk.page), "") + " " + h.chunk.text
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    merged = []
    for s in sents:
        if merged and re.fullmatch(r"\s*(\[[^\]]+\]\s*)+\.?\s*", s):
            merged[-1] += " " + s
        else:
            merged.append(s)
    problems = []
    for s in merged:
        cites = CITE.findall(s)
        if not cites:
            problems.append(f"no citation: {s[:60]}")
            continue
        src = " ".join(valid.get((p, int(pg)), "") for p, pg in cites)
        if not src.strip():
            problems.append(f"citation not in retrieved passages: {s[:60]}")
            continue
        body = CITE.sub("", s)
        for n in set(re.findall(r"\d+(?:\.\d+)?", body)):
            if n not in src:
                problems.append(f"number {n} not in cited passage")
        w = content(body)
        if w and len(w & content(src)) / len(w) < 0.5:
            problems.append(f"low overlap with cited passage: {s[:60]}")
    return (not problems), problems
