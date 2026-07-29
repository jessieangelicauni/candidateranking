import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from fpdf import FPDF

from evidencerank.cli import cli, rank
from evidencerank.models import CalibratedResult, JDRequirements, Tier


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
    _make_pdf(Path("resumes/candidate1.pdf"), "Candidate One\nPython, PyTorch")


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


def _fake_final_state_two_candidates(fake_jd: JDRequirements) -> dict:
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


def test_rank_single_run_writes_json_and_markdown_reports(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output
        assert Path("report.json").exists()
        assert Path("report.md").exists()
        data = json.loads(Path("report.json").read_text(encoding="utf-8"))
        assert data["calibrated_results"][0]["candidate_id"] == "candidate1"
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Rankings" in content
        assert "candidate1" in content

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert "candidate1" in invoked_state["raw_resumes"]
    assert invoked_state["prefilter_threshold"] == 0.9
    assert invoked_state["hallucination_threshold"] == 85.0


def test_rank_single_run_report_md_includes_pipeline_stats(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Pipeline Stats" in content


def test_rank_single_run_passes_llm_concurrency_through_to_graph_state(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--llm-concurrency", "8"],
        )

        assert result.exit_code == 0, result.output

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 8


def test_rank_single_run_defaults_llm_concurrency_to_twelve(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 12


def test_rank_rejects_non_positive_llm_concurrency(monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--llm-concurrency", "0"],
        )

        assert result.exit_code != 0
        assert "llm-concurrency" in result.output.lower() or "llm_concurrency" in result.output.lower()


def test_rank_requires_jd_and_resumes_dir():
    runner = CliRunner()
    result = runner.invoke(rank, [])

    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_rank_folds_in_extra_reports_alongside_fresh_run(monkeypatch, tmp_path):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state_two_candidates(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    extra_report = tmp_path / "past_run.json"
    extra_report.write_text(json.dumps({
        "jd": fake_jd.model_dump(),
        "profiles": {"candidate1": {"raw_cv_text": "x"}, "candidate2": {"raw_cv_text": "y"}},
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [
            {"candidate_id": "candidate1", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""},
            {"candidate_id": "candidate2", "final_rank": 2, "tier": "Moderate Fit", "rating": 6, "calibration_notes": ""},
        ],
        "hallucination_reports": {},
    }), encoding="utf-8")

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--reports", str(extra_report)],
        )

        assert result.exit_code == 0, result.output
        assert Path("report.json").exists()
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Rank Stability" in content


def test_rank_multi_run_runs_pipeline_n_times_and_writes_run_reports(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state_two_candidates(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "3"],
        )

        assert result.exit_code == 0, result.output
        assert Path("run1.json").exists()
        assert Path("run2.json").exists()
        assert Path("run3.json").exists()
        assert not Path("report.json").exists()
        assert Path("report.md").exists()

    assert fake_graph.invoke.call_count == 3


def test_rank_multi_run_includes_rank_stability_section(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state_two_candidates(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "2"],
        )

        assert result.exit_code == 0, result.output
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Rank Stability" in content
        assert "1.000" in content


def test_rank_defaults_runs_to_one(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(rank, ["--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output
        assert Path("report.json").exists()

    assert fake_graph.invoke.call_count == 1


def test_rank_rejects_runs_below_one():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "0"],
        )

        assert result.exit_code != 0


def test_rank_multi_run_passes_llm_concurrency_through_to_each_run(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state_two_candidates(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(
            rank,
            ["--jd", "jd.txt", "--resumes-dir", "resumes", "--runs", "2", "--llm-concurrency", "8"],
        )

        assert result.exit_code == 0, result.output

    for call in fake_graph.invoke.call_args_list:
        assert call[0][0]["max_concurrency"] == 8


def test_cli_group_dispatches_to_rank_subcommand(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_jd_and_resume()

        result = runner.invoke(cli, ["rank", "--jd", "jd.txt", "--resumes-dir", "resumes"])

        assert result.exit_code == 0, result.output
        assert Path("report.json").exists()
