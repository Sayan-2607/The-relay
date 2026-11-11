"""Fetch the three arXiv PDFs at the EXACT versions and extract per-page text (pages separated by form feed).
Requires: pip install pypdf   (run this on your machine; it needs internet)"""
import io, os, urllib.request
from pypdf import PdfReader

PAPERS = {"attn": "1706.03762v7", "moshi": "2410.00037v1", "hnet": "2507.07955v2"}
os.makedirs("data/papers", exist_ok=True)
for key, ver in PAPERS.items():
    pdf_path = f"data/papers/{ver}.pdf"
    if not os.path.exists(pdf_path):
        req = urllib.request.Request(f"https://arxiv.org/pdf/{ver}", headers={"User-Agent": "relay-assignment/1.0"})
        open(pdf_path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    reader = PdfReader(pdf_path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    open(f"data/papers/{key}.txt", "w", encoding="utf-8").write("\f".join(pages))
    print(key, ver, len(pages), "pages", sum(len(p.split()) for p in pages), "words")
print("Open data/papers/*.txt and eyeball them: two-column PDFs, tables and equations extract badly (a finding for the write-up).")
