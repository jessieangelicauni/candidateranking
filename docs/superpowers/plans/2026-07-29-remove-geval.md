# Remove GEval from the Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two DeepEval `GEval` LLM-as-judge metrics (`RecruiterAlignment`, `EvidenceRelevancy`) from the evaluation harness, leaving only deterministic signals (Pipeline Stats, Hallucination Rate, Stage Timings, Rank Stability).

**Architecture:** `evaluation/report.py` currently calls into `evaluation/metrics.py` (DeepEval `GEval` + a local Ollama judge model) to score each candidate, then renders a "GEval Metrics" Markdown section. This plan strips that call chain out module-by-module, working from the innermost module (`evaluation/report.py`) outward to its callers (`evaluation/cli.py`, `src/evidencerank/cli.py`), then deletes the now-unreferenced `evaluation/metrics.py` module and its dependency (`deepeval`), then updates docs.

**Tech Stack:** Python 3.11, click, pytest, uv.

## Global Constraints

- No replacement ground-truth metric in this pass — that is a separate, future project (per spec).
- No changes to production ranking logic, `report.json`, or `report.md` — only the evaluation harness (`evaluation/`) and the eval-report wiring in `src/evidencerank/cli.py`.
- Historical spec docs (`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md`, `docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md`) are not edited — only superseded via cross-reference in the new spec (already done).
- `scipy` and `numpy` dependencies stay in `pyproject.toml` — both are used outside the code being removed (`evaluation/rank_stability.py` uses `scipy`; `src/evidencerank/agents/prefilter.py` uses `numpy` directly).

---

### Task 1: Remove GEval scoring from `evaluation/report.py`

**Files:**
- Modify: `evaluation/report.py`
- Test: `evaluation/test_report.py`

**Interfaces:**
- Produces: `compute_pipeline_stats(report_path: str | Path) -> dict` (unchanged signature/behavior). `build_eval_markdown_report(report_paths: list[str | Path]) -> str` (drops the `max_concurrency` parameter it had). `write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None` (drops the `max_concurrency` parameter it had). These three are what Task 2 and Task 3's callers must match.
- Consumes: `evaluation.rank_stability.rank_stability` (unchanged, already exists).

- [ ] **Step 1: Replace `evaluation/report.py` with the GEval-free version**

Replace the entire file content with:

```python
import json
from pathlib import Path

from evaluation.rank_stability import rank_stability


def compute_pipeline_stats(report_path: str | Path) -> dict:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))

    total_candidates = len(data["profiles"])
    dropped_prefilter = len(data["dropped"])
    evaluated_by_judge = len(data["judge_results"])
    hallucination_flagged = sum(
        1
        for report in data["hallucination_reports"].values()
        if report["unverified_quotes"]
    )
    hallucination_rate = hallucination_flagged / evaluated_by_judge if evaluated_by_judge else 0.0

    return {
        "total_candidates": total_candidates,
        "passed_prefilter": total_candidates - dropped_prefilter,
        "dropped_prefilter": dropped_prefilter,
        "evaluated_by_judge": evaluated_by_judge,
        "hallucination_rate": hallucination_rate,
    }


def build_eval_markdown_report(report_paths: list[str | Path]) -> str:
    primary = report_paths[0]
    data = json.loads(Path(primary).read_text(encoding="utf-8"))
    stats = compute_pipeline_stats(primary)

    lines = [
        "# Evaluation Metric Report",
        "",
        f"**JD:** {data['jd']['title']}",
        f"**Primary report:** {primary}",
    ]
    if len(report_paths) > 1:
        extra = ", ".join(str(path) for path in report_paths[1:])
        lines.append(f"**Additional reports (rank stability):** {extra}")
    lines += [
        "",
        "## Pipeline Stats",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total candidates | {stats['total_candidates']} |",
        f"| Passed pre-filter | {stats['passed_prefilter']} |",
        f"| Dropped by pre-filter | {stats['dropped_prefilter']} |",
        f"| Evaluated by Judge | {stats['evaluated_by_judge']} |",
        f"| Hallucination Rate | {stats['hallucination_rate']:.1%} |",
    ]

    stage_timings = data.get("stage_timings") or {}
    if stage_timings:
        lines += [
            "",
            "## Stage Timings",
            "",
            "| Stage | Seconds |",
            "|---|---|",
        ]
        for stage_name, seconds in stage_timings.items():
            lines.append(f"| {stage_name} | {seconds:.3f} |")

    if len(report_paths) >= 2:
        stability = rank_stability([str(path) for path in report_paths])
        lines += [
            "",
            "## Rank Stability",
            "",
            "| Runs | Mean Spearman | Mean Kendall Tau |",
            "|---|---|---|",
            f"| {stability['n_runs']} | {stability['mean_spearman']:.3f} "
            f"| {stability['mean_kendall_tau']:.3f} |",
        ]

    return "\n".join(lines)


def write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None:
    Path(path).write_text(build_eval_markdown_report(report_paths), encoding="utf-8")
```

- [ ] **Step 2: Run the (still old) test file to see it fail**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: FAIL at collection — `ImportError: cannot import name 'compute_geval_scores' from 'evaluation.report'` (the old test file still imports it, and it no longer exists).

- [ ] **Step 3: Replace `evaluation/test_report.py` with the GEval-free version**

Replace the entire file content with:

```python
import json
from pathlib import Path

from evaluation.report import compute_pipeline_stats


def _write_report(path: Path, **overrides) -> None:
    base = {
        "jd": {
            "title": "ML Engineer",
            "required_skills": ["Python", "PyTorch"],
            "nice_to_have_skills": ["Docker"],
            "min_experience_years": 2,
            "education": "",
            "responsibilities": ["Build models"],
        },
        "profiles": {},
        "prefilter_results": {},
        "dropped": [],
        "judge_results": {},
        "calibrated_results": [],
        "hallucination_reports": {},
    }
    base.update(overrides)
    path.write_text(json.dumps(base), encoding="utf-8")


def test_compute_pipeline_stats_counts_candidates(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={
            "alice": {"raw_cv_text": "alice cv"},
            "bob": {"raw_cv_text": "bob cv"},
            "carol": {"raw_cv_text": "carol cv"},
        },
        dropped=[{"candidate_id": "carol", "reason": "pre-filter: no relevant skill overlap"}],
        judge_results={
            "alice": {"tier": "Strong Fit", "rating": 9, "evidence": []},
            "bob": {"tier": "Weak Fit", "rating": 3, "evidence": []},
        },
        hallucination_reports={
            "alice": {"candidate_id": "alice", "unverified_quotes": []},
            "bob": {"candidate_id": "bob", "unverified_quotes": ["some quote"]},
        },
    )

    stats = compute_pipeline_stats(report_path)

    assert stats == {
        "total_candidates": 3,
        "passed_prefilter": 2,
        "dropped_prefilter": 1,
        "evaluated_by_judge": 2,
        "hallucination_rate": 0.5,
    }


def test_compute_pipeline_stats_hallucination_rate_is_zero_when_no_one_judged(tmp_path):
    # Guards against a ZeroDivisionError when every candidate is dropped at
    # pre-filter and no one reaches the Judge.
    report_path = tmp_path / "report.json"
    _write_report(report_path, profiles={}, judge_results={}, hallucination_reports={})

    stats = compute_pipeline_stats(report_path)

    assert stats["hallucination_rate"] == 0.0


def _write_calibrated_report(path: Path, ranks: dict[str, int]) -> None:
    _write_report(
        path,
        profiles={candidate_id: {"raw_cv_text": f"{candidate_id} cv"} for candidate_id in ranks},
        judge_results={
            candidate_id: {"tier": "Strong Fit", "rating": 8, "evidence": []}
            for candidate_id in ranks
        },
        calibrated_results=[
            {
                "candidate_id": candidate_id,
                "final_rank": final_rank,
                "tier": "Strong Fit",
                "rating": 8,
                "calibration_notes": "",
            }
            for candidate_id, final_rank in ranks.items()
        ],
    )


def test_build_eval_markdown_report_single_run_omits_rank_stability(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    markdown = build_eval_markdown_report([report_path])

    assert "## Pipeline Stats" in markdown
    assert "## GEval Metrics" not in markdown
    assert "## Rank Stability" not in markdown


def test_build_eval_markdown_report_multi_run_includes_rank_stability(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})

    markdown = build_eval_markdown_report([report_a, report_b])

    assert "## Rank Stability" in markdown
    assert "1.000" in markdown  # identical rankings -> spearman/kendall == 1.0


def test_write_eval_markdown_report_writes_file(tmp_path):
    from evaluation.report import write_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    out_path = tmp_path / "evaluation-metric.md"

    write_eval_markdown_report([report_path], out_path)

    assert out_path.exists()
    assert "## Pipeline Stats" in out_path.read_text(encoding="utf-8")


def test_build_eval_markdown_report_includes_stage_timings_when_present(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}
    report_path.write_text(json.dumps(data), encoding="utf-8")

    markdown = build_eval_markdown_report([report_path])

    assert "## Stage Timings" in markdown
    assert "| extract_profiles | 1.500 |" in markdown
    assert "| judge | 3.250 |" in markdown


def test_build_eval_markdown_report_omits_stage_timings_when_absent(tmp_path):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    # NOTE: intentionally does NOT add "stage_timings" — simulates an older
    # report.json written before this key existed, to guard against a
    # KeyError regression.

    markdown = build_eval_markdown_report([report_path])

    assert "## Stage Timings" not in markdown
```

- [ ] **Step 4: Run the updated test file to confirm it passes**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: PASS — all 7 tests green.

- [ ] **Step 5: Commit**

```bash
git add evaluation/report.py evaluation/test_report.py
git commit -m "refactor: remove GEval scoring from evaluation/report.py"
```

---

### Task 2: Remove GEval concurrency plumbing from `evaluation/cli.py`

**Files:**
- Modify: `evaluation/cli.py`
- Test: `evaluation/test_cli.py`

**Interfaces:**
- Consumes: `write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None` from Task 1 (no `max_concurrency` param).
- Produces: `eval_report` click command with options `--reports` (required, multiple) and `--out` only (no `--llm-concurrency`). `rank_stability` click command unchanged (`--jd`, `--resumes-dir`, `--runs`, `--llm-concurrency`, `--out`) — its `--llm-concurrency` still bounds `run_pipeline`'s LLM concurrency, just no longer threads into `write_eval_markdown_report`.

- [ ] **Step 1: Edit `evaluation/cli.py`**

Replace the whole file with:

```python
import click

from evidencerank.cli import run_pipeline
from evidencerank.report import write_json_report

from evaluation.report import write_eval_markdown_report


@click.command()
@click.option(
    "--reports",
    "report_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
)
@click.option("--out", default="evaluation-metric.md", type=click.Path())
def eval_report(report_paths, out):
    """Build an evaluation metric report from one or more report.json files.

    Pass --reports once per report.json path. One path gives pipeline stats
    only; repeat --reports for each additional run to also include rank
    stability across runs, e.g.:

        evidencerank-eval-report --reports a.json --reports b.json --out evaluation-metric.md
    """
    write_eval_markdown_report(list(report_paths), out)
    click.echo(f"Wrote {out}")


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--runs", default=3, type=click.IntRange(min=2))
@click.option("--llm-concurrency", default=4, type=click.IntRange(min=1))
@click.option("--out", default="evaluation-metric.md", type=click.Path())
def rank_stability(jd_path, resumes_dir, runs, llm_concurrency, out):
    """Run the pipeline RUNS times on the same JD/resumes and build an
    evaluation report that includes rank stability across the runs.

    Writes run1.json, run2.json, ... (one full report.json per run, never
    overwritten) alongside OUT, so each run stays available for inspection -
    not just the aggregated evaluation-metric.md.
    """
    report_paths = []
    for i in range(1, runs + 1):
        click.echo(f"Run {i}/{runs}...")
        final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)
        path = f"run{i}.json"
        write_json_report(final_state, path)
        report_paths.append(path)
        click.echo(f"Wrote {path}")

    write_eval_markdown_report(report_paths, out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    eval_report()
```

- [ ] **Step 2: Run the (still old) test file to see it fail**

Run: `uv run pytest evaluation/test_cli.py -v`
Expected: FAIL — `test_eval_report_cli_passes_llm_concurrency_through` and `test_eval_report_cli_defaults_llm_concurrency_to_four` both fail with `result.exit_code != 0` (click rejects the now-removed `--llm-concurrency` option on `eval_report` with a "no such option" usage error).

- [ ] **Step 3: Edit `evaluation/test_cli.py`**

Replace the whole file with:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from fpdf import FPDF

from evidencerank.models import CalibratedResult, JDRequirements, Tier

from evaluation.cli import eval_report, rank_stability


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


def test_eval_report_cli_writes_output_file(tmp_path):
    report_path = tmp_path / "report.json"
    _write_minimal_report(report_path)
    out_path = tmp_path / "evaluation-metric.md"

    runner = CliRunner()
    result = runner.invoke(
        eval_report, ["--reports", str(report_path), "--out", str(out_path)]
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
        assert Path("evaluation-metric.md").exists()

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
        content = Path("evaluation-metric.md").read_text(encoding="utf-8")
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
```

- [ ] **Step 4: Run the updated test file to confirm it passes**

Run: `uv run pytest evaluation/test_cli.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add evaluation/cli.py evaluation/test_cli.py
git commit -m "refactor: drop GEval concurrency option from eval_report CLI"
```

---

### Task 3: Remove GEval concurrency plumbing from `src/evidencerank/cli.py`

**Files:**
- Modify: `src/evidencerank/cli.py:59-62`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None` from Task 1 (no `max_concurrency` param).

- [ ] **Step 1: Edit `src/evidencerank/cli.py`**

In the `rank` command, change:

```python
    from evaluation.report import write_eval_markdown_report

    write_eval_markdown_report([OUT_JSON], OUT_EVAL_REPORT, max_concurrency=llm_concurrency)
    click.echo(f"Wrote {OUT_EVAL_REPORT}")
```

to:

```python
    from evaluation.report import write_eval_markdown_report

    write_eval_markdown_report([OUT_JSON], OUT_EVAL_REPORT)
    click.echo(f"Wrote {OUT_EVAL_REPORT}")
```

- [ ] **Step 2: Run the (still old) test file to see it fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL —
- `test_rank_command_always_writes_eval_report` fails with `AssertionError` on `assert "## GEval Metrics" in content` (the section no longer exists, since Task 1 removed it from `evaluation/report.py`).
- `test_rank_command_passes_llm_concurrency_through_to_graph_state` fails with `KeyError: 'max_concurrency'` when it reads `eval_report_kwargs["max_concurrency"]` (no longer passed, since this task's Step 1 edit stopped passing it).

- [ ] **Step 3: Edit `tests/test_cli.py`**

Change the assertion in `test_rank_command_always_writes_eval_report` from:

```python
        assert "## Pipeline Stats" in content
        assert "## GEval Metrics" in content
```

to:

```python
        assert "## Pipeline Stats" in content
        assert "## GEval Metrics" not in content
```

Replace `test_rank_command_passes_llm_concurrency_through_to_graph_state` in full, from:

```python
def test_rank_command_passes_llm_concurrency_through_to_graph_state(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    eval_report_calls = []
    monkeypatch.setattr(
        "evaluation.report.write_eval_markdown_report",
        lambda *args, **kwargs: eval_report_calls.append((args, kwargs)),
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

    # The same --llm-concurrency value also bounds the eval report's GEval
    # concurrency, not just the production judge/extractor calls.
    _, eval_report_kwargs = eval_report_calls[0]
    assert eval_report_kwargs["max_concurrency"] == 8
```

to:

```python
def test_rank_command_passes_llm_concurrency_through_to_graph_state(monkeypatch):
    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_final_state(fake_jd)
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    monkeypatch.setattr(
        "evaluation.report.write_eval_markdown_report",
        lambda *args, **kwargs: None,
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
```

- [ ] **Step 4: Run the updated test file to confirm it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/cli.py tests/test_cli.py
git commit -m "refactor: stop passing GEval concurrency into eval report from rank command"
```

---

### Task 4: Delete `evaluation/metrics.py`, drop the `deepeval` dependency

**Files:**
- Delete: `evaluation/metrics.py`
- Delete: `evaluation/test_metrics.py`
- Modify: `pyproject.toml:15`
- Modify: `.env.example`
- Delete: `.deepeval/.deepeval_telemetry.txt` (tracked artifact produced by the `deepeval` package at import time; vestigial once the dependency is gone)
- Modify: `.gitignore`

**Interfaces:** None — by this point (after Tasks 1-3), nothing in the repo imports `evaluation.metrics` or `deepeval`.

- [ ] **Step 1: Confirm nothing still references `evaluation.metrics` or `deepeval`**

Run: `grep -rn "evaluation.metrics\|deepeval" --include="*.py" src evaluation tests`
Expected: no output (only `evaluation/metrics.py` and `evaluation/test_metrics.py` themselves would match, and they're deleted in this task).

- [ ] **Step 2: Delete the metrics module and its test file**

```bash
git rm evaluation/metrics.py evaluation/test_metrics.py
```

- [ ] **Step 3: Remove the `deepeval` dependency from `pyproject.toml`**

In the `dependencies` list, change:

```toml
    "click>=8.1",
    "deepeval>=1.1",
    "scipy>=1.13",
```

to:

```toml
    "click>=8.1",
    "scipy>=1.13",
```

- [ ] **Step 4: Remove `EVIDENCERANK_EVAL_MODEL` from `.env.example`**

Change the file from:

```
# Copy this file to .env and fill in real values. .env is gitignored.

# Only needed if you hit Hugging Face Hub rate limits/auth requirements when
# downloading the bge-small-en-v1.5 embedding model used by the pre-filter stage.
HF_TOKEN=

# Optional: Ollama model used as the GEval judge in evaluation/metrics.py.
# Defaults to qwen2.5:14b-instruct if unset.
EVIDENCERANK_EVAL_MODEL=
```

to:

```
# Copy this file to .env and fill in real values. .env is gitignored.

# Only needed if you hit Hugging Face Hub rate limits/auth requirements when
# downloading the bge-small-en-v1.5 embedding model used by the pre-filter stage.
HF_TOKEN=
```

- [ ] **Step 5: Remove the now-vestigial `.deepeval` telemetry artifact and gitignore it**

```bash
git rm .deepeval/.deepeval_telemetry.txt
```

Add `.deepeval/` to `.gitignore` — append it to the end of the file so it reads:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
report.json
report.md
*.egg-info/
.env
.claude/
.superpowers/
.cache/
.deepeval/
```

- [ ] **Step 6: Sync the environment and lockfile**

Run: `uv sync`
Expected: exits 0, `uv.lock` updates to drop `deepeval` and its transitive-only dependencies, `deepeval` is uninstalled from `.venv`.

- [ ] **Step 7: Run the full test suite to confirm nothing broke**

Run: `uv run pytest`
Expected: PASS — full suite green, no `ModuleNotFoundError` for `deepeval` anywhere (confirms no other file silently depended on it).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore
git commit -m "chore: drop deepeval dependency now that GEval scoring is removed"
```

---

### Task 5: Update `README.md`

**Files:**
- Modify: `README.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Remove the `qwen3:14b` pull step from Setup**

Change:

```markdown
2. Pull the default models:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull qwen2.5:14b-instruct
   ollama pull qwen3:14b  # GEval judge for the evaluation harness (evaluation-metric.md)
   ```
```

to:

```markdown
2. Pull the default models:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull qwen2.5:14b-instruct
   ```
```

- [ ] **Step 2: Update the eval-report note in "Running the pipeline"**

Change:

```markdown
Every run also generates the evaluation metric report (`evaluation-metric.md`) — see
[Evaluation metric report](#evaluation-metric-report) below for what it contains. This
requires `ollama serve` running locally with the eval judge model available (same as the
standalone `evidencerank-eval-report` command).
```

to:

```markdown
Every run also generates the evaluation metric report (`evaluation-metric.md`) — see
[Evaluation metric report](#evaluation-metric-report) below for what it contains. This is
pure computation over the run's `report.json` (no LLM calls, no extra setup beyond what
`rank` already requires).
```

- [ ] **Step 3: Update the "Environment variables" section**

Change:

```markdown
`.env` is loaded automatically (via `python-dotenv`) whenever you run `uv run evidencerank`
or import `evaluation.metrics` — no manual `export` needed. `.env` is gitignored — never
commit real tokens.

| Variable | Required for | Notes |
| --- | --- | --- |
| `HF_TOKEN` | Downloading the `BAAI/bge-small-en-v1.5` embedding model used by the pre-filter stage | Only needed if you hit Hugging Face Hub rate limits/auth requirements on first download; the model is cached locally afterward. Get a token at https://huggingface.co/settings/tokens. |
| `EVIDENCERANK_EVAL_MODEL` | `evaluation/metrics.py` GEval metrics | Optional. Ollama model used as the GEval judge; defaults to `qwen3:14b` (`ollama pull qwen3:14b`). Requires `ollama serve` running locally, same as the production pipeline. |
```

to:

```markdown
`.env` is loaded automatically (via `python-dotenv`) whenever you run `uv run evidencerank`
— no manual `export` needed. `.env` is gitignored — never commit real tokens.

| Variable | Required for | Notes |
| --- | --- | --- |
| `HF_TOKEN` | Downloading the `BAAI/bge-small-en-v1.5` embedding model used by the pre-filter stage | Only needed if you hit Hugging Face Hub rate limits/auth requirements on first download; the model is cached locally afterward. Get a token at https://huggingface.co/settings/tokens. |
```

- [ ] **Step 4: Rewrite the "Research evaluation harness" intro**

Change:

```markdown
## Research evaluation harness

The `evaluation/` package is separate from the production pipeline (`src/evidencerank/`):

- `evaluation/metrics.py` — DeepEval `GEval` metrics (RecruiterAlignment, EvidenceRelevancy)
  to run against pipeline output. There is deliberately no GEval "Groundedness" metric —
  quote authenticity is already measured deterministically by the production pipeline's
  hallucination checker (see Hallucination Rate below), which is strictly more reliable
  for that question than an LLM re-judging it (see the caveat below).
- `evaluation/rank_stability.py` — computes Spearman/Kendall-tau rank correlation across
  repeated runs on the same input, to report LLM judgment consistency.
- `evaluation/report.py` — aggregates the above (plus pipeline stats: candidates
  submitted, pre-filter pass/drop, hallucination rate) into a single Markdown
  evaluation report, suitable for a paper appendix.

The two `GEval` metrics in `evaluation/metrics.py` use a local Ollama model as the
judge (`EVIDENCERANK_EVAL_MODEL`, see [Environment variables](#environment-variables)) —
no external API key required, same as the production pipeline, though a different model
by default (`qwen3:14b`, a reasoning model, rather than the production judge's
`qwen2.5:14b-instruct`). A non-reasoning model of the same size was measurably less
reliable as an evaluator: it sometimes misjudged its own side-by-side text comparison
(e.g. claiming a quote didn't match the resume when it was in fact an exact match),
dragging down GEval scores with the eval judge's own errors rather than real
production-pipeline defects.
```

to:

```markdown
## Research evaluation harness

The `evaluation/` package is separate from the production pipeline (`src/evidencerank/`):

- `evaluation/rank_stability.py` — computes Spearman/Kendall-tau rank correlation across
  repeated runs on the same input, to report ranking consistency.
- `evaluation/report.py` — aggregates the above (plus pipeline stats: candidates
  submitted, pre-filter pass/drop, hallucination rate) into a single Markdown
  evaluation report, suitable for a paper appendix.

Both signals are deterministic — fuzzy string-matching for the hallucination rate,
rank-correlation statistics for rank stability — with no LLM judging another LLM's
output involved, and no extra model or `ollama serve` requirement beyond what the
production pipeline itself already needs.
```

- [ ] **Step 5: Update the rank-stability command description**

Change:

```markdown
This runs the pipeline `--runs` times (default `3`, minimum `2`), writes each run's full
report as `run1.json`, `run2.json`, ... (never overwritten, so every run stays available
for inspection), and builds `evaluation-metric.md` from all of them — GEval scores and
pipeline stats from `run1.json`, rank stability (Spearman/Kendall-tau) across all of
them. `--llm-concurrency` and `--out` work the same as the other commands.
```

to:

```markdown
This runs the pipeline `--runs` times (default `3`, minimum `2`), writes each run's full
report as `run1.json`, `run2.json`, ... (never overwritten, so every run stays available
for inspection), and builds `evaluation-metric.md` from all of them — pipeline stats
from `run1.json`, rank stability (Spearman/Kendall-tau) across all of them.
`--llm-concurrency` and `--out` work the same as the other commands.
```

- [ ] **Step 6: Rewrite the "Evaluation metric report" section**

Change:

```markdown
`uv run evidencerank-eval-report` builds a Markdown report combining GEval metric
aggregates, pipeline stats, and (when 2+ runs are given) rank stability, from one or
more existing `report.json` files:
```

to:

```markdown
`uv run evidencerank-eval-report` builds a Markdown report combining pipeline stats and
(when 2+ runs are given) rank stability, from one or more existing `report.json` files:
```

Change:

```markdown
GEval scores and pipeline stats are always computed from the first `--reports` path
given; every path is used for rank stability. This requires `ollama serve` running
locally (same GEval judge model as above) — the GEval calls are not mocked outside
of tests.

When the underlying `report.json` includes per-stage timing (`stage_timings`,
added by the production pipeline), the report also includes a "Stage Timings"
table showing wall-clock seconds per stage — absent for older `report.json`
files that predate this field.

Quote authenticity itself isn't a GEval metric (see Research evaluation harness above) —
it's already measured deterministically via Hallucination Rate in Pipeline Stats, since
the hallucination checker strips unverified evidence before calibration. RecruiterAlignment
and EvidenceRelevancy are the GEval signals for judge quality, since calibration and
claim-quote relevance have no deterministic equivalent.
```

to:

```markdown
Pipeline stats are always computed from the first `--reports` path given; every path is
used for rank stability. Both are pure computation over `report.json` — no `ollama serve`
or model required to build this report.

When the underlying `report.json` includes per-stage timing (`stage_timings`,
added by the production pipeline), the report also includes a "Stage Timings"
table showing wall-clock seconds per stage — absent for older `report.json`
files that predate this field.

Quote authenticity is measured deterministically via Hallucination Rate in Pipeline
Stats — the hallucination checker strips unverified evidence before calibration, so
this is a direct count from that check, not a judgment call by any model.
```

- [ ] **Step 7: Verify no stray references remain**

Run: `grep -n "GEval\|deepeval\|qwen3\|EVIDENCERANK_EVAL_MODEL" README.md`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: update README for GEval removal from the evaluation harness"
```

---

### Task 6: Final verification

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: PASS — every test in `tests/` and `evaluation/` green.

- [ ] **Step 2: Grep the whole repo (excluding historical specs and `.venv`) for leftover references**

Run: `grep -rn "GEval\|deepeval\|EVIDENCERANK_EVAL_MODEL" --include="*.py" --include="*.md" --include="*.toml" --include="*.example" . | grep -v ".venv" | grep -v "docs/superpowers/specs/2026-07-27-eval-metric-report-design.md" | grep -v "docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md" | grep -v "docs/superpowers/specs/2026-07-24-evidencerank-design.md" | grep -v "docs/superpowers/specs/2026-07-29-remove-geval-design.md" | grep -v "tests/agents/test_judge.py"`
Expected: no output. (The excluded files are: the three historical spec docs, which intentionally stay as a record of past decisions per the spec's "Out of Scope" section; the new spec doc itself, which references GEval by name in its Purpose section describing what was removed; and `tests/agents/test_judge.py`, which has code comments referencing "GEval" only to explain why a regression test exists — no import or code dependency.)

- [ ] **Step 3: Smoke-test the CLI end-to-end against real resumes (manual, requires `ollama serve` running)**

Run: `uv run evidencerank --jd machine_learning_engineer.txt --resumes-dir resumes --llm-concurrency 4`
Expected: exits 0, writes `report.json`, `report.md`, and `evaluation-metric.md`. Open `evaluation-metric.md` and confirm it has "Pipeline Stats" (and "Stage Timings", and "Rank Stability" if you ran `evidencerank-rank-stability` instead) but no "GEval Metrics" section.
