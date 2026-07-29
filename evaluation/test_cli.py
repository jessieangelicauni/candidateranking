import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from fpdf import FPDF

from evidencerank.models import CalibratedResult, JDRequirements, Tier

from evaluation.cli import rank_stability, report


def _write_minimal_report(path: Path) -> None:
    data = {
        "jd": {
            "title": "ML Engineer",
            "required_skills": ["Python"],
            "nice_to_have_skills": [],
            "min_experience_years": 0,
            "education": "",
            "responsibilities": [],
        },
        "profiles": {"alice": {"raw_cv_text": "alice cv"}},
        "prefilter_results": {},
        "dropped": [],
        "judge_results": {"alice": {"tier": "Strong Fit", "rating": 9, "evidence": []}},
        "calibrated_results": [
            {"candidate_id": "alice", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        "hallucination_reports": {"alice": {"candidate_id": "alice", "unverified_quotes": []}},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_report_cli_writes_output_file(tmp_path):
    report_path = tmp_path / "report.json"
    _write_minimal_report(report_path)
    out_path = tmp_path / "report.md"

    runner = CliRunner()
    result = runner.invoke(
        report, ["--reports", str(report_path), "--out", str(out_path)]
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert str(out_path) in result.output


def _make_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def _write_jd_and_resume():
    Path("jd.txt").write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
    Path("resumes").mkdir()
    _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython")


def _fake_final_state(fake_jd: JDRequirements) -> dict:
    # Two candidates (not one) - rank_stability() requires at least 2
    # candidates common to every run to compute a correlation at all.
    return {
        "jd": fake_jd,
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [
            CalibratedResult(
                candidate_id="candidate1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="First",
            ),
            CalibratedResult(
                candidate_id="candidate2", final_rank=2, tier=Tier.MODERATE_FIT,
                rating=6, calibration_notes="Second",
            ),
        ],
        "hallucination_reports": {},
    }


def test_rank_stability_runs_pipeline_n_times_and_writes_run_reports(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank_stability,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "3"],
        )

        assert result.exit_code == 0, result.output
        assert Path("run1.json").exists()
        assert Path("run2.json").exists()
        assert Path("run3.json").exists()
        assert Path("report.md").exists()

    assert fake_graph.invoke.call_count == 3


def test_rank_stability_includes_rank_stability_section(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank_stability,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "2"],
        )

        assert result.exit_code == 0, result.output
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Rank Stability" in content
        assert "1.000" in content  # identical rankings every run -> perfect correlation


def test_rank_stability_defaults_runs_to_three(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(rank_stability, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output

    assert fake_graph.invoke.call_count == 3


def test_rank_stability_rejects_runs_below_two():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank_stability,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "1"],
        )

        assert result.exit_code != 0


def test_rank_stability_passes_llm_concurrency_through_to_each_run(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank_stability,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "2", "--llm-concurrency", "8"],
        )

        assert result.exit_code == 0, result.output

    for call in fake_graph.invoke.call_args_list:
        assert call[0][0]["max_concurrency"] == 8
