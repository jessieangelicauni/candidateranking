from pathlib import Path

import pdfplumber


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_resume_text(path: str | Path) -> str:
    with pdfplumber.open(Path(path)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()
