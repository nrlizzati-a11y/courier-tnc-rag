import json
import re
from datetime import date
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

HEADERS = {"User-Agent": "Mozilla/5.0 (RAG portfolio demo)"}


def clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def html_to_text(data: bytes) -> str:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def load_source(src: dict) -> str:
    if "path" in src:
        data = Path(src["path"]).read_bytes()
        kind = src.get("type") or ("pdf" if src["path"].lower().endswith(".pdf") else "html")
    else:
        resp = requests.get(src["url"], headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.content
        is_pdf = "pdf" in resp.headers.get("Content-Type", "").lower() or src["url"].lower().endswith(".pdf")
        kind = src.get("type") or ("pdf" if is_pdf else "html")
    return clean(pdf_to_text(data) if kind == "pdf" else html_to_text(data))


def main():
    sources = json.loads(Path("sources.json").read_text())
    docs = []
    for src in sources:
        name = f"{src['carrier']} - {src['title']}"
        try:
            text = load_source(src)
        except Exception as e:
            print(f"FAILED  {name}: {e}")
            continue
        status = "OK    " if len(text) > 5000 else "SHORT "
        print(f"{status} {name}: {len(text):,} characters")
        docs.append({
            "carrier": src["carrier"],
            "title": src["title"],
            "url": src.get("url", src.get("path", "")),
            "retrieved": date.today().isoformat(),
            "text": text,
        })
    Path("data").mkdir(exist_ok=True)
    Path("data/docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(docs)} documents to data/docs.json")


if __name__ == "__main__":
    main()