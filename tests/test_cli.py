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


def _fake_final_state(fake_jd: JDRequirements) -> dict:
    return {
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


def test_rank_command_writes_json_and_markdown_reports(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    called = []
    monkeypatch.setattr(
        "evaluation.report.write_eval_markdown_report",
        lambda *args, **kwargs: called.append(args),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("jd.txt").write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
        Path("resumes").mkdir()
        _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython, PyTorch")

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output
        assert Path("report.json").exists()
        assert Path("report.md").exists()
        data = json.loads(Path("report.json").read_text(encoding="utf-8"))
        assert data["calibrated_results"][0]["candidate_id"] == "candidate1"

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert "candidate1" in invoked_state["raw_resumes"]
    assert invoked_state["prefilter_threshold"] == 0.7
    assert invoked_state["hallucination_threshold"] == 85.0


def test_rank_command_always_writes_eval_report(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("jd.txt").write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
        Path("resumes").mkdir()
        _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython, PyTorch")

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output
        assert Path("evaluation-metric.md").exists()
        content = Path("evaluation-metric.md").read_text(encoding="utf-8")
        assert "## Pipeline Stats" in content
        assert "## GEval Metrics" in content


def test_rank_command_passes_llm_concurrency_through_to_graph_state(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    called = []
    monkeypatch.setattr(
        "evaluation.report.write_eval_markdown_report",
        lambda *args, **kwargs: called.append(args),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("jd.txt").write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
        Path("resumes").mkdir()
        _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython, PyTorch")

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--llm-concurrency", "8"],
        )

        assert result.exit_code == 0, result.output

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 8


def test_rank_command_defaults_llm_concurrency_to_four(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    called = []
    monkeypatch.setattr(
        "evaluation.report.write_eval_markdown_report",
        lambda *args, **kwargs: called.append(args),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("jd.txt").write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
        Path("resumes").mkdir()
        _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython, PyTorch")

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 4


def test_rank_command_rejects_non_positive_llm_concurrency(monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("jd.txt").write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
        Path("resumes").mkdir()
        _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython, PyTorch")

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--llm-concurrency", "0"],
        )

        assert result.exit_code != 0
        assert "llm-concurrency" in result.output.lower() or "llm_concurrency" in result.output.lower()
