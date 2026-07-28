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
            "--prefilter-threshold", "0.7",
            "--hallucination-threshold", "90.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "candidate1"
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert "candidate1" in invoked_state["raw_resumes"]
    assert invoked_state["prefilter_threshold"] == 0.7
    assert invoked_state["hallucination_threshold"] == 90.0


def test_rank_command_with_eval_report_flag_writes_eval_report(tmp_path, monkeypatch):
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
    out_eval_report = tmp_path / "eval_report.md"
    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
            "--with-eval-report",
            "--out-eval-report", str(out_eval_report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_eval_report.exists()
    content = out_eval_report.read_text(encoding="utf-8")
    assert "## Pipeline Stats" in content
    assert "## GEval Metrics" in content


def test_rank_command_without_eval_report_flag_skips_eval_report(tmp_path, monkeypatch):
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
    out_eval_report = tmp_path / "eval_report.md"

    called = []
    monkeypatch.setattr(
        "evaluation.report.write_eval_markdown_report",
        lambda *args, **kwargs: called.append(args),
    )

    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
            "--out-eval-report", str(out_eval_report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert not out_eval_report.exists()
    assert called == []


def test_rank_command_passes_llm_concurrency_through_to_graph_state(tmp_path, monkeypatch):
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

    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(tmp_path / "out.json"),
            "--out-md", str(tmp_path / "out.md"),
            "--llm-concurrency", "8",
        ],
    )

    assert result.exit_code == 0, result.output
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 8


def test_rank_command_defaults_llm_concurrency_to_four(tmp_path, monkeypatch):
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

    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(tmp_path / "out.json"),
            "--out-md", str(tmp_path / "out.md"),
        ],
    )

    assert result.exit_code == 0, result.output
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 4
