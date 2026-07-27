# Evaluation Metric Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a report generator that consumes one or more existing `report.json` pipeline outputs and produces a single Markdown "evaluation metric report" (GEval metric aggregates, pipeline stats, and rank stability across runs) suitable for a Q1-Scopus paper appendix.

**Architecture:** New module `evaluation/report.py` with three pure/near-pure functions (`compute_pipeline_stats`, `compute_geval_scores`, `build_eval_markdown_report`) plus a writer (`write_eval_markdown_report`), and a new `evaluation/cli.py` exposing a `click` command registered as `evidencerank-eval-report`. `compute_geval_scores` is the only function that calls out to the local Ollama judge model (via the existing `evaluation/metrics.py` GEval objects); everything else is pure JSON aggregation.

**Tech Stack:** Python 3.11, `deepeval` (GEval), `click`, `statistics` (stdlib), existing `evaluation/rank_stability.py`.

## Global Constraints

- No new dependencies — `deepeval`, `click`, `scipy`, `numpy` are already in `pyproject.toml`.
- Markdown output only (per spec — no LaTeX/CSV/JSON output in this build).
- Aggregates only — no per-candidate score breakdown table.
- GEval sample stdev uses `statistics.stdev` (`ddof=1`); `std` is `None` when `n < 2` (mathematically undefined for n=0 and n=1).
- `evaluation/` stays fully decoupled from `src/evidencerank/` — this module reads `report.json` from disk only, never live pipeline `state`.
- Tests must not require a live Ollama server — mock/monkeypatch GEval metric `.measure()` calls, consistent with `evaluation/test_metrics.py`'s existing pattern of exercising the metric objects without invoking them.

---

### Task 1: `compute_pipeline_stats` — pure JSON aggregation, no LLM

**Files:**
- Create: `evaluation/report.py`
- Test: `evaluation/test_report.py`

**Interfaces:**
- Produces: `compute_pipeline_stats(report_path: str | Path) -> dict` returning
  `{"total_candidates": int, "passed_prefilter": int, "dropped_prefilter": int, "evaluated_by_judge": int, "hallucination_flagged": int}`.

Real `report.json` shape (confirmed against this repo's own output):
```json
{
  "jd": {"title": "...", "required_skills": [...], "nice_to_have_skills": [...], "min_experience_years": 0, "education": "", "responsibilities": [...]},
  "profiles": {"<candidate_id>": {"contact": {...}, "skills": [...], "work_history": [...], "education": [...], "projects": [...], "candidate_id": "...", "raw_cv_text": "..."}},
  "prefilter_results": {"<candidate_id>": {"candidate_id": "...", "similarity": 0.0, "passed": true}},
  "dropped": [{"candidate_id": "...", "reason": "pre-filter: no relevant skill overlap"}],
  "judge_results": {"<candidate_id>": {"tier": "Strong Fit", "rating": 9, "evidence": [{"claim": "...", "quote": "..."}]}},
  "calibrated_results": [{"candidate_id": "...", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": "..."}],
  "hallucination_reports": {"<candidate_id>": {"candidate_id": "...", "unverified_quotes": []}}
}
```
Note: `profiles` contains every submitted candidate (extraction runs before the pre-filter), so `total_candidates = len(profiles)`.

- [ ] **Step 1: Write the failing test**

Create `evaluation/test_report.py`:

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
        "hallucination_flagged": 1,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.report'` (or `ImportError: cannot import name 'compute_pipeline_stats'`)

- [ ] **Step 3: Write minimal implementation**

Create `evaluation/report.py`:

```python
import json
from pathlib import Path


def compute_pipeline_stats(report_path: str | Path) -> dict:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))

    total_candidates = len(data["profiles"])
    dropped_prefilter = len(data["dropped"])
    hallucination_flagged = sum(
        1
        for report in data["hallucination_reports"].values()
        if report["unverified_quotes"]
    )

    return {
        "total_candidates": total_candidates,
        "passed_prefilter": total_candidates - dropped_prefilter,
        "dropped_prefilter": dropped_prefilter,
        "evaluated_by_judge": len(data["judge_results"]),
        "hallucination_flagged": hallucination_flagged,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/report.py evaluation/test_report.py
git commit -m "feat: add pipeline stats aggregation for evaluation report"
```

---

### Task 2: `compute_geval_scores` — GEval aggregation over Judge results

**Files:**
- Modify: `evaluation/report.py`
- Test: `evaluation/test_report.py`

**Interfaces:**
- Consumes: `groundedness_metric`, `recruiter_alignment_metric`, `evidence_relevancy_metric`, `build_test_case` from `evaluation/metrics.py` (existing — see `evaluation/metrics.py:13-56`). Each metric object has `.name: str`, `.threshold: float`, and `.measure(test_case) -> float`.
- Produces: `compute_geval_scores(report_path: str | Path) -> dict[str, dict]`, keyed by each metric's `.name` (`"Groundedness"`, `"RecruiterAlignment"`, `"EvidenceRelevancy"`), each value `{"n": int, "mean": float | None, "std": float | None, "pass_rate": float | None}`.

- [ ] **Step 1: Write the failing tests**

Add to `evaluation/test_report.py`:

```python
from unittest.mock import Mock

from evaluation.metrics import (
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)
from evaluation.report import compute_geval_scores


def _write_geval_report(path: Path, judge_results: dict, profiles: dict | None = None) -> None:
    _write_report(
        path,
        profiles=profiles
        or {candidate_id: {"raw_cv_text": f"{candidate_id} cv"} for candidate_id in judge_results},
        judge_results=judge_results,
    )


def test_compute_geval_scores_aggregates_two_candidates(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_geval_report(
        report_path,
        judge_results={
            "alice": {"tier": "Strong Fit", "rating": 9, "evidence": [{"claim": "c1", "quote": "q1"}]},
            "bob": {"tier": "Weak Fit", "rating": 3, "evidence": [{"claim": "c2", "quote": "q2"}]},
        },
    )

    monkeypatch.setattr(groundedness_metric, "measure", Mock(side_effect=[0.9, 0.5]))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(side_effect=[0.8, 0.4]))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(side_effect=[1.0, 0.6]))

    scores = compute_geval_scores(report_path)

    assert scores["Groundedness"]["n"] == 2
    assert round(scores["Groundedness"]["mean"], 4) == 0.7
    assert round(scores["Groundedness"]["std"], 4) == 0.2828  # stdev([0.9, 0.5])
    assert scores["Groundedness"]["pass_rate"] == 0.5  # only 0.9 >= 0.7 threshold

    assert round(scores["RecruiterAlignment"]["mean"], 4) == 0.6
    assert scores["EvidenceRelevancy"]["pass_rate"] == 0.5  # only 1.0 >= 0.7 threshold


def test_compute_geval_scores_empty_judge_results_returns_none_fields(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_geval_report(report_path, judge_results={})

    mock = Mock()
    monkeypatch.setattr(groundedness_metric, "measure", mock)
    monkeypatch.setattr(recruiter_alignment_metric, "measure", mock)
    monkeypatch.setattr(evidence_relevancy_metric, "measure", mock)

    scores = compute_geval_scores(report_path)

    assert scores["Groundedness"] == {"n": 0, "mean": None, "std": None, "pass_rate": None}
    mock.assert_not_called()


def test_compute_geval_scores_single_candidate_std_is_none(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_geval_report(
        report_path,
        judge_results={"alice": {"tier": "Weak Fit", "rating": 4, "evidence": []}},
    )

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.6))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.6))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.6))

    scores = compute_geval_scores(report_path)

    assert scores["Groundedness"]["n"] == 1
    assert scores["Groundedness"]["mean"] == 0.6
    assert scores["Groundedness"]["std"] is None
    assert scores["Groundedness"]["pass_rate"] == 0.0  # 0.6 < 0.7 threshold
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest evaluation/test_report.py -v -k compute_geval_scores`
Expected: FAIL with `ImportError: cannot import name 'compute_geval_scores'`

- [ ] **Step 3: Write minimal implementation**

Add to `evaluation/report.py` (append imports at top, functions at bottom):

```python
import statistics

from deepeval.test_case import LLMTestCase

from evaluation.metrics import (
    build_test_case,
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)

_GEVAL_METRICS = [groundedness_metric, recruiter_alignment_metric, evidence_relevancy_metric]


def _format_jd_text(jd: dict) -> str:
    return "\n".join(
        [
            f"Title: {jd['title']}",
            f"Required skills: {', '.join(jd['required_skills'])}",
            f"Nice-to-have skills: {', '.join(jd['nice_to_have_skills'])}",
            f"Minimum experience years: {jd['min_experience_years']}",
            f"Education: {jd['education']}",
            f"Responsibilities: {', '.join(jd['responsibilities'])}",
        ]
    )


def _format_judge_result_text(judge_result: dict) -> str:
    lines = [f"Tier: {judge_result['tier']}", f"Rating: {judge_result['rating']}"]
    for claim in judge_result["evidence"]:
        lines.append(f'- {claim["claim"]}: "{claim["quote"]}"')
    return "\n".join(lines)


def _aggregate_scores(scores: list[float], threshold: float) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "pass_rate": None}
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if n >= 2 else None
    pass_rate = sum(1 for score in scores if score >= threshold) / n
    return {"n": n, "mean": mean, "std": std, "pass_rate": pass_rate}


def compute_geval_scores(report_path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    jd_text = _format_jd_text(data["jd"])

    test_cases: list[LLMTestCase] = []
    for candidate_id, judge_result in data["judge_results"].items():
        cv_text = data["profiles"][candidate_id]["raw_cv_text"]
        judge_text = _format_judge_result_text(judge_result)
        test_cases.append(build_test_case(jd_text, judge_text, cv_text))

    results: dict[str, dict] = {}
    for metric in _GEVAL_METRICS:
        scores = [metric.measure(test_case) for test_case in test_cases]
        results[metric.name] = _aggregate_scores(scores, metric.threshold)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: PASS (all tests from Task 1 and Task 2)

- [ ] **Step 5: Commit**

```bash
git add evaluation/report.py evaluation/test_report.py
git commit -m "feat: add GEval score aggregation for evaluation report"
```

---

### Task 3: `build_eval_markdown_report` / `write_eval_markdown_report`

**Files:**
- Modify: `evaluation/report.py`
- Test: `evaluation/test_report.py`

**Interfaces:**
- Consumes: `compute_pipeline_stats` (Task 1), `compute_geval_scores` (Task 2), `rank_stability` from `evaluation/rank_stability.py:13` (existing — takes `list[str]` of report paths, returns `{"mean_spearman": float, "mean_kendall_tau": float, "n_runs": int}`).
- Produces: `build_eval_markdown_report(report_paths: list[str | Path]) -> str`, `write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None`.

- [ ] **Step 1: Write the failing tests**

Add to `evaluation/test_report.py`:

```python
from evaluation.report import build_eval_markdown_report, write_eval_markdown_report


def _write_calibrated_report(path: Path, ranks: dict[str, int]) -> None:
    _write_geval_report(
        path,
        judge_results={
            candidate_id: {"tier": "Strong Fit", "rating": 8, "evidence": []}
            for candidate_id in ranks
        },
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["calibrated_results"] = [
        {
            "candidate_id": candidate_id,
            "final_rank": final_rank,
            "tier": "Strong Fit",
            "rating": 8,
            "calibration_notes": "",
        }
        for candidate_id, final_rank in ranks.items()
    ]
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_eval_markdown_report_single_run_omits_rank_stability(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    markdown = build_eval_markdown_report([report_path])

    assert "## Pipeline Stats" in markdown
    assert "## GEval Metrics" in markdown
    assert "## Rank Stability" not in markdown


def test_build_eval_markdown_report_multi_run_includes_rank_stability(tmp_path, monkeypatch):
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    markdown = build_eval_markdown_report([report_a, report_b])

    assert "## Rank Stability" in markdown
    assert "1.000" in markdown  # identical rankings -> spearman/kendall == 1.0


def test_write_eval_markdown_report_writes_file(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    out_path = tmp_path / "eval_report.md"

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    write_eval_markdown_report([report_path], out_path)

    assert out_path.exists()
    assert "## Pipeline Stats" in out_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest evaluation/test_report.py -v -k "markdown_report"`
Expected: FAIL with `ImportError: cannot import name 'build_eval_markdown_report'`

- [ ] **Step 3: Write minimal implementation**

Add to `evaluation/report.py` (add import at top, functions at bottom):

```python
from evaluation.rank_stability import rank_stability


def build_eval_markdown_report(report_paths: list[str | Path]) -> str:
    primary = report_paths[0]
    data = json.loads(Path(primary).read_text(encoding="utf-8"))
    stats = compute_pipeline_stats(primary)
    geval = compute_geval_scores(primary)

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
        f"| Hallucination-flagged candidates | {stats['hallucination_flagged']} |",
        "",
        "## GEval Metrics",
        "",
        "| Metric | n | Mean | Std Dev | Pass Rate |",
        "|---|---|---|---|---|",
    ]
    for name in ("Groundedness", "RecruiterAlignment", "EvidenceRelevancy"):
        m = geval[name]
        mean_str = f"{m['mean']:.3f}" if m["mean"] is not None else "N/A"
        std_str = f"{m['std']:.3f}" if m["std"] is not None else "N/A"
        pass_str = f"{m['pass_rate']:.1%}" if m["pass_rate"] is not None else "N/A"
        lines.append(f"| {name} | {m['n']} | {mean_str} | {std_str} | {pass_str} |")

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: PASS (all tests from Tasks 1-3)

- [ ] **Step 5: Commit**

```bash
git add evaluation/report.py evaluation/test_report.py
git commit -m "feat: render evaluation metric report as markdown"
```

---

### Task 4: CLI entry point

**Files:**
- Create: `evaluation/cli.py`
- Modify: `pyproject.toml`
- Test: `evaluation/test_cli.py`

**Interfaces:**
- Consumes: `write_eval_markdown_report` from `evaluation/report.py` (Task 3).
- Produces: `click` command `eval_report` in `evaluation/cli.py`, registered as the `evidencerank-eval-report` console script.

- [ ] **Step 1: Write the failing test**

Create `evaluation/test_cli.py`:

```python
import json
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner

from evaluation.cli import eval_report
from evaluation.metrics import (
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)


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


def test_eval_report_cli_writes_output_file(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    _write_minimal_report(report_path)
    out_path = tmp_path / "eval_report.md"

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    runner = CliRunner()
    result = runner.invoke(
        eval_report, ["--reports", str(report_path), "--out", str(out_path)]
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert str(out_path) in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest evaluation/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.cli'`

- [ ] **Step 3: Write minimal implementation**

Create `evaluation/cli.py`:

```python
import click

from evaluation.report import write_eval_markdown_report


@click.command()
@click.option(
    "--reports",
    "report_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
)
@click.option("--out", default="eval_report.md", type=click.Path())
def eval_report(report_paths, out):
    """Build an evaluation metric report from one or more report.json files.

    Pass one path for GEval metrics + pipeline stats only, or two or more to
    also include rank stability across runs.
    """
    write_eval_markdown_report(list(report_paths), out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    eval_report()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest evaluation/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Register the console script**

Modify `pyproject.toml` — add the new script alongside the existing `evidencerank` entry:

```diff
 [project.scripts]
 evidencerank = "evidencerank.cli:rank"
+evidencerank-eval-report = "evaluation.cli:eval_report"
```

- [ ] **Step 6: Re-sync and verify the console script installs**

Run: `uv sync`
Run: `uv run evidencerank-eval-report --help`
Expected: prints the command's `--reports`/`--out` usage text without error

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS (all existing tests plus the new `evaluation/test_report.py` and `evaluation/test_cli.py`)

- [ ] **Step 8: Commit**

```bash
git add evaluation/cli.py evaluation/test_cli.py pyproject.toml uv.lock
git commit -m "feat: add evidencerank-eval-report CLI entry point"
```

---

## Manual Verification (after Task 4)

Run the report generator against the real `report.json` already in this repo, against a live Ollama server, to confirm the full path works end-to-end (not just mocked):

```bash
uv run evidencerank-eval-report --reports report.json --out /tmp/eval_report_manual_check.md
cat /tmp/eval_report_manual_check.md
```

Expected: a Markdown document with `## Pipeline Stats` and `## GEval Metrics` tables populated with real numbers (no `## Rank Stability` section, since only one report path was given), and no errors from the GEval Ollama calls.
