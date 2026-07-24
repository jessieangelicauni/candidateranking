from pathlib import Path

from fpdf import FPDF

from evidencerank.io import load_resume_text, load_text_file


def _make_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_load_resume_text_extracts_pdf_content(tmp_path):
    pdf_path = tmp_path / "resume.pdf"
    _make_pdf(pdf_path, "Jane Example\nPython, SQL, Docker")

    text = load_resume_text(pdf_path)

    assert "Jane Example" in text
    assert "Python, SQL, Docker" in text


def test_load_text_file_reads_plain_text(tmp_path):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")

    assert load_text_file(jd_path) == "Machine Learning Engineer\nPython required"
