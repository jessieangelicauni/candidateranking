import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from fpdf import FPDF

from evidencerank.cli import rank
from evidencerank.models import CalibratedResult, JDRequirements, Tier


def _make_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_rank_command_writes_json_and_markdown_reports(tmp_path, monkeypatch):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    _make_pdf(resumes_dir / "candidate1.pdf", "Candidate One\nPython, PyTorch")

    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_final_state = {
        "jd": fake_jd,
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [
            CalibratedResult(
                candidate_id="candidate1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only candidate",
            )
        ],
        "hallucination_reports": {},
    }
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final_state
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "candidate1"
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert "candidate1" in invoked_state["raw_resumes"]
