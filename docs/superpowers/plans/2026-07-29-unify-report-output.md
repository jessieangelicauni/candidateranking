# Unify report.md and evaluation-metric.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `src/evidencerank/report.py` into `evaluation/report.py` so the pipeline produces two output files instead of three — `report.json` (unchanged) and a single `report.md` containing the ranking table plus the pipeline stats, stage timings, and rank-stability sections that used to live in a separate `evaluation-metric.md`.

**Architecture:** `evaluation/report.py` becomes the one module owning both outputs. The ranking-table renderer, which used to read live in-memory pipeline `state` (Pydantic objects), is rewritten to read a parsed `report.json` dict off disk instead — the same source the stats/rank-stability sections already use. This lets one function (`build_markdown_report`) serve both the `rank` command (which writes `report.json` then immediately reads it back) and the standalone `evidencerank-report`/`evidencerank-rank-stability` commands (which only ever have saved `report.json` files, never live state).

**Tech Stack:** Python 3.11, click, pytest, scipy (via `evaluation/rank_stability.py`, unchanged).

## Global Constraints

- No change to `report.json`'s structure or to ranking/calibration logic.
- No change to `evaluation/rank_stability.py` — stays a separate module, imported by `evaluation/report.py` exactly as it is today.
- `report.md`'s combined content, in order: title, JD/report-path metadata, `## Rankings` (the table), `## Pipeline Stats`, `## Stage Timings` (only if present), `## Rank Stability` (only if 2+ report paths).
- The standalone CLI command is renamed `evidencerank-eval-report` → `evidencerank-report`; its `--out` default changes from `evaluation-metric.md` to `report.md`.

---

### Task 1: Consolidate report generation into `evaluation/report.py`

**Files:**
- Modify: `evaluation/report.py`
- Test: `evaluation/test_report.py`

**Interfaces:**
- Produces: `build_json_report(state: dict) -> dict`, `write_json_report(state: dict, path: str | Path) -> None` (moved verbatim from `src/evidencerank/report.py` — Task 2 will delete that file and repoint its caller here). `compute_pipeline_stats(report_path: str | Path) -> dict` (unchanged). `build_markdown_report(report_paths: list[str | Path]) -> str` and `write_markdown_report(report_paths: list[str | Path], path: str | Path) -> None` (replace `build_eval_markdown_report`/`write_eval_markdown_report` — Tasks 2 and 3 will update every caller to these new names).
- Consumes: `evaluation.rank_stability.rank_stability` (unchanged, already imported).

- [ ] **Step 1: Replace `evaluation/report.py` with the consolidated version**

Replace the entire file content with:

```python
import json
from pathlib import Path

from evaluation.rank_stability import rank_stability


def build_json_report(state: dict) -> dict:
    return {
        "jd": state["jd"].model_dump(),
        "profiles": {
            candidate_id: profile.model_dump()
            for candidate_id, profile in state.get("profiles", {}).items()
        },
        "prefilter_results": {
            candidate_id: result.model_dump()
            for candidate_id, result in state.get("prefilter_results", {}).items()
        },
        "dropped": state.get("dropped", []),
        "judge_results": {
            candidate_id: result.model_dump()
            for candidate_id, result in state.get("judge_results", {}).items()
        },
        "calibrated_results": [result.model_dump() for result in state.get("calibrated_results", [])],
        "hallucination_reports": {
            candidate_id: report.model_dump()
            for candidate_id, report in state.get("hallucination_reports", {}).items()
        },
        "stage_timings": state.get("stage_timings", {}),
    }


def write_json_report(state: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_json_report(state), indent=2), encoding="utf-8")


def _escape_table_cell(text: str) -> str:
    """Make free-text safe to interpolate into a Markdown table cell.

    Escapes literal pipe characters (which would otherwise be parsed as
    column separators) and collapses newlines to spaces (which would
    otherwise break the row onto multiple lines / rows).
    """
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _format_evidence(judge_result: dict | None) -> str:
    """Render a candidate's Judge evidence claims as one Markdown table cell."""
    if judge_result is None:
        return ""
    joined = "; ".join(f"{claim['claim']}: {claim['quote']}" for claim in judge_result["evidence"])
    return _escape_table_cell(joined)


def _format_hallucination_flag(hallucination_report: dict | None) -> str:
    """Render a candidate's hallucination-check outcome as one Markdown table cell."""
    if hallucination_report is None or not hallucination_report["unverified_quotes"]:
        return "—"
    return f"{len(hallucination_report['unverified_quotes'])} removed"


def _build_ranking_table(data: dict) -> str:
    lines = [
        "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags | Calibration Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    judge_results = data.get("judge_results", {})
    hallucination_reports = data.get("hallucination_reports", {})
    for result in sorted(data.get("calibrated_results", []), key=lambda r: r["final_rank"]):
        candidate_id = result["candidate_id"]
        notes = _escape_table_cell(result["calibration_notes"])
        evidence = _format_evidence(judge_results.get(candidate_id))
        flags = _format_hallucination_flag(hallucination_reports.get(candidate_id))
        lines.append(
            f"| {result['final_rank']} | {candidate_id} | {result['tier']} "
            f"| {result['rating']} | {evidence} | {flags} | {notes} |"
        )
    return "\n".join(lines)


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


def build_markdown_report(report_paths: list[str | Path]) -> str:
    primary = report_paths[0]
    data = json.loads(Path(primary).read_text(encoding="utf-8"))
    stats = compute_pipeline_stats(primary)

    lines = [
        "# Candidate Ranking Report",
        "",
        f"**JD:** {data['jd']['title']}",
        f"**Primary report:** {primary}",
    ]
    if len(report_paths) > 1:
        extra = ", ".join(str(path) for path in report_paths[1:])
        lines.append(f"**Additional reports (rank stability):** {extra}")
    lines += [
        "",
        "## Rankings",
        "",
        _build_ranking_table(data),
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


def write_markdown_report(report_paths: list[str | Path], path: str | Path) -> None:
    Path(path).write_text(build_markdown_report(report_paths), encoding="utf-8")
```

- [ ] **Step 2: Run the (still old) test file to see it fail**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: The old test file's module-level import (`from evaluation.report import compute_pipeline_stats`) still succeeds, so collection doesn't fail outright, but every test that locally imports the now-removed pre-merge names (`from evaluation.report import build_eval_markdown_report` / `write_eval_markdown_report`, inside each test function body) fails at run time with `ImportError: cannot import name 'build_eval_markdown_report' from 'evaluation.report'` (or `write_eval_markdown_report`). Only the 2 `compute_pipeline_stats` tests pass; the other 7 fail this way.

- [ ] **Step 3: Replace `evaluation/test_report.py` with the consolidated version**

Replace the entire file content with:

```python
import json
from pathlib import Path

from evidencerank.models import (
    CalibratedResult,
    CandidateProfile,
    ContactInfo,
    EvidenceClaim,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    PrefilterResult,
    Tier,
)

from evaluation.report import (
    build_json_report,
    build_markdown_report,
    compute_pipeline_stats,
    write_json_report,
    write_markdown_report,
)


def _sample_state():
    return {
        "jd": JDRequirements(title="ML Engineer", required_skills=["Python"]),
        "profiles": {
            "strong": CandidateProfile(
                candidate_id="strong",
                raw_cv_text="Jane Doe, 5 years Python",
                contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
                skills=["Python"],
            )
        },
        "prefilter_results": {
            "strong": PrefilterResult(candidate_id="strong", similarity=0.9, passed=True),
            "weak": PrefilterResult(candidate_id="weak", similarity=0.1, passed=False),
        },
        "dropped": [{"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}],
        "judge_results": {
            "strong": JudgeResult(
                candidate_id="strong", tier=Tier.STRONG_FIT, rating=9,
                evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
            )
        },
        "calibrated_results": [
            CalibratedResult(
                candidate_id="strong", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only surviving candidate",
            )
        ],
        "hallucination_reports": {
            "strong": HallucinationReport(candidate_id="strong", unverified_quotes=[]),
        },
    }


def test_build_json_report_contains_all_sections():
    report = build_json_report(_sample_state())

    assert report["jd"]["title"] == "ML Engineer"
    assert report["profiles"]["strong"]["candidate_id"] == "strong"
    assert report["profiles"]["strong"]["raw_cv_text"] == "Jane Doe, 5 years Python"
    assert report["profiles"]["strong"]["contact"]["name"] == "Jane Doe"
    assert report["prefilter_results"]["strong"]["similarity"] == 0.9
    assert report["prefilter_results"]["strong"]["passed"] is True
    assert report["prefilter_results"]["weak"]["passed"] is False
    assert report["dropped"][0]["candidate_id"] == "weak"
    assert report["judge_results"]["strong"]["rating"] == 9
    assert report["calibrated_results"][0]["final_rank"] == 1
    assert report["hallucination_reports"]["strong"]["unverified_quotes"] == []


def test_build_json_report_defaults_missing_stages_to_empty():
    minimal_state = {"jd": JDRequirements(title="ML Engineer", required_skills=["Python"])}

    report = build_json_report(minimal_state)

    assert report["profiles"] == {}
    assert report["prefilter_results"] == {}


def test_build_json_report_includes_stage_timings():
    state = _sample_state()
    state["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}

    report = build_json_report(state)

    assert report["stage_timings"] == {"extract_profiles": 1.5, "judge": 3.25}


def test_build_json_report_defaults_missing_stage_timings_to_empty_dict():
    report = build_json_report(_sample_state())

    assert report["stage_timings"] == {}


def test_write_json_report_writes_valid_json(tmp_path):
    out_path = tmp_path / "report.json"
    write_json_report(_sample_state(), out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "strong"


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


def test_build_markdown_report_includes_rankings_and_pipeline_stats(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    markdown = build_markdown_report([report_path])

    assert "## Rankings" in markdown
    assert (
        "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags "
        "| Calibration Notes |" in markdown
    )
    assert "| 1 | alice | Strong Fit | 8 |  | — |  |" in markdown
    assert "## Pipeline Stats" in markdown


def test_build_markdown_report_single_run_omits_rank_stability(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1, "bob": 2})

    markdown = build_markdown_report([report_path])

    assert "## Rank Stability" not in markdown


def test_build_markdown_report_multi_run_includes_rank_stability(tmp_path):
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_calibrated_report(report_a, {"alice": 1, "bob": 2})
    _write_calibrated_report(report_b, {"alice": 1, "bob": 2})

    markdown = build_markdown_report([report_a, report_b])

    assert "## Rank Stability" in markdown
    assert "1.000" in markdown  # identical rankings -> spearman/kendall == 1.0


def test_write_markdown_report_writes_file(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    out_path = tmp_path / "report.md"

    write_markdown_report([report_path], out_path)

    assert out_path.exists()
    assert "## Pipeline Stats" in out_path.read_text(encoding="utf-8")


def test_build_markdown_report_includes_stage_timings_when_present(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}
    report_path.write_text(json.dumps(data), encoding="utf-8")

    markdown = build_markdown_report([report_path])

    assert "## Stage Timings" in markdown
    assert "| extract_profiles | 1.500 |" in markdown
    assert "| judge | 3.250 |" in markdown


def test_build_markdown_report_omits_stage_timings_when_absent(tmp_path):
    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    # NOTE: intentionally does NOT add "stage_timings" — simulates an older
    # report.json written before this key existed, to guard against a
    # KeyError regression.

    markdown = build_markdown_report([report_path])

    assert "## Stage Timings" not in markdown


def test_build_markdown_report_orders_rankings_by_rank_ascending(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={
            "first": {"raw_cv_text": "first cv"},
            "second": {"raw_cv_text": "second cv"},
            "third": {"raw_cv_text": "third cv"},
        },
        calibrated_results=[
            {
                "candidate_id": "third", "final_rank": 3, "tier": "Weak Fit",
                "rating": 4, "calibration_notes": "Ranked third",
            },
            {
                "candidate_id": "first", "final_rank": 1, "tier": "Strong Fit",
                "rating": 9, "calibration_notes": "Ranked first",
            },
            {
                "candidate_id": "second", "final_rank": 2, "tier": "Moderate Fit",
                "rating": 6, "calibration_notes": "Ranked second",
            },
        ],
    )

    markdown = build_markdown_report([report_path])
    lines = markdown.splitlines()

    assert lines.index("| 1 | first | Strong Fit | 9 |  | — | Ranked first |") < \
        lines.index("| 2 | second | Moderate Fit | 6 |  | — | Ranked second |") < \
        lines.index("| 3 | third | Weak Fit | 4 |  | — | Ranked third |")


def test_build_markdown_report_escapes_pipes_and_newlines_in_notes(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {
                "candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9,
                "calibration_notes": "Great fit | but watch out\nfor gaps in employment",
            }
        ],
    )

    markdown = build_markdown_report([report_path])
    data_row = next(line for line in markdown.splitlines() if line.startswith("| 1 |"))

    # No embedded newline leaked into the output.
    assert "\n" not in data_row

    # The literal pipe from the notes text was escaped (backslash-pipe), not left
    # as a bare separator that would split the notes into extra table columns.
    assert "Great fit \\| but watch out for gaps in employment" in data_row
    assert "Great fit | but" not in data_row

    # Exactly 7 well-formed columns: splitting on the escaped-pipe-protected row
    # (only unescaped pipes act as separators) yields the 7 data fields plus the
    # two empty strings from the leading/trailing pipe.
    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 9  # "", rank, candidate, tier, rating, evidence, flags, notes, ""


def test_build_markdown_report_escapes_pipes_and_newlines_in_evidence(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        judge_results={
            "strong": {
                "tier": "Strong Fit", "rating": 9,
                "evidence": [
                    {"claim": "Led team", "quote": "Managed 5 | 10 person teams\nacross two years"},
                    {"claim": "Shipped feature", "quote": "Delivered on time"},
                ],
            }
        },
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
    )

    markdown = build_markdown_report([report_path])
    data_row = next(line for line in markdown.splitlines() if line.startswith("| 1 |"))

    # No embedded newline leaked into the output.
    assert "\n" not in data_row

    # The literal pipe from the quote was escaped, and both evidence claims made it
    # into the same cell (joined, not split across rows/columns).
    assert "Led team: Managed 5 \\| 10 person teams across two years" in data_row
    assert "Shipped feature: Delivered on time" in data_row
    assert "Managed 5 | 10 person" not in data_row

    # Exactly 7 well-formed columns: only unescaped pipes act as separators.
    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 9  # "", rank, candidate, tier, rating, evidence, flags, notes, ""


def test_build_markdown_report_shows_removed_count_for_flagged_candidate(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        hallucination_reports={
            "strong": {
                "candidate_id": "strong",
                "unverified_quotes": ["fabricated quote one", "fabricated quote two"],
            },
        },
    )

    markdown = build_markdown_report([report_path])

    assert "| 2 removed |" in markdown


def test_build_markdown_report_shows_dash_when_no_hallucination_report_present(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        profiles={"strong": {"raw_cv_text": "strong cv"}},
        calibrated_results=[
            {"candidate_id": "strong", "final_rank": 1, "tier": "Strong Fit", "rating": 9, "calibration_notes": ""}
        ],
        hallucination_reports={},
    )

    markdown = build_markdown_report([report_path])

    assert "| — |" in markdown
```

- [ ] **Step 4: Run the updated test file to confirm it passes**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: PASS — all 18 tests green.

- [ ] **Step 5: Commit**

```bash
git add evaluation/report.py evaluation/test_report.py
git commit -m "refactor: consolidate report generation into evaluation/report.py"
```

---

### Task 2: Repoint `src/evidencerank/cli.py` at the consolidated module, delete the old one

**Files:**
- Modify: `src/evidencerank/cli.py`
- Modify: `tests/test_cli.py`
- Delete: `src/evidencerank/report.py`
- Delete: `tests/test_report.py`

**Interfaces:**
- Consumes: `write_json_report(state, path)`, `write_markdown_report(report_paths, path)` from `evaluation.report` (Task 1).

- [ ] **Step 1: Edit `src/evidencerank/cli.py`**

Change the imports at the top, from:

```python
from pathlib import Path

import click
from dotenv import load_dotenv

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file
from evidencerank.report import write_json_report, write_markdown_report

load_dotenv()

OUT_JSON = "report.json"
OUT_MD = "report.md"
OUT_EVAL_REPORT = "evaluation-metric.md"
PREFILTER_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 85.0
```

to:

```python
from pathlib import Path

import click
from dotenv import load_dotenv

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file

from evaluation.report import write_json_report, write_markdown_report

load_dotenv()

OUT_JSON = "report.json"
OUT_MD = "report.md"
PREFILTER_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 85.0
```

Change the `rank` command, from:

```python
def rank(jd_path, resumes_dir, llm_concurrency):
    """Rank every resume in RESUMES_DIR against the job description at JD."""
    final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)

    write_json_report(final_state, OUT_JSON)
    write_markdown_report(final_state, OUT_MD)
    click.echo(f"Wrote {OUT_JSON} and {OUT_MD}")

    from evaluation.report import write_eval_markdown_report

    write_eval_markdown_report([OUT_JSON], OUT_EVAL_REPORT)
    click.echo(f"Wrote {OUT_EVAL_REPORT}")
```

to:

```python
def rank(jd_path, resumes_dir, llm_concurrency):
    """Rank every resume in RESUMES_DIR against the job description at JD."""
    final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)

    write_json_report(final_state, OUT_JSON)
    write_markdown_report([OUT_JSON], OUT_MD)
    click.echo(f"Wrote {OUT_JSON} and {OUT_MD}")
```

- [ ] **Step 2: Delete the old production report module**

```bash
git rm src/evidencerank/report.py tests/test_report.py
```

- [ ] **Step 3: Run the (still old) test file to see it fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — every test that invokes `rank` fails, because the still-old test file monkeypatches `"evaluation.report.write_eval_markdown_report"` (a name that no longer exists — `evaluation.report` now only has `write_markdown_report`) and asserts on `evaluation-metric.md`, which the `rank` command no longer writes. Specifically: `test_rank_command_writes_json_and_markdown_reports`, `test_rank_command_passes_llm_concurrency_through_to_graph_state`, and `test_rank_command_defaults_llm_concurrency_to_four` all error with `AttributeError: <module 'evaluation.report' ...> does not have the attribute 'write_eval_markdown_report'` from the now-invalid monkeypatch target; `test_rank_command_always_writes_eval_report` fails with a plain `AssertionError` on `assert Path("evaluation-metric.md").exists()` (that file is no longer written at all); `test_rank_command_rejects_non_positive_llm_concurrency` still passes (it never touches report writing).

- [ ] **Step 4: Replace `tests/test_cli.py` with the updated version**

Replace the entire file content with:

```python
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
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Rankings" in content
        assert "candidate1" in content

    invoked_state = fake_graph.invoke.call_args[0][0]
    assert "candidate1" in invoked_state["raw_resumes"]
    assert invoked_state["prefilter_threshold"] == 0.7
    assert invoked_state["hallucination_threshold"] == 85.0


def test_rank_command_report_md_includes_pipeline_stats(monkeypatch):
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
        content = Path("report.md").read_text(encoding="utf-8")
        assert "## Pipeline Stats" in content


def test_rank_command_passes_llm_concurrency_through_to_graph_state(monkeypatch):
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
```

Note: this version drops the `monkeypatch.setattr("evaluation.report.write_eval_markdown_report", ...)` calls the old tests used — `write_markdown_report` is now pure, dependency-free computation (no LLM calls, no network), so there's nothing left to isolate the tests from; letting it run for real is simpler and just as fast. (It also sidesteps a subtlety: `src/evidencerank/cli.py` now imports `write_markdown_report` at module level via `from evaluation.report import ...`, which binds the name into `evidencerank.cli`'s own namespace — monkeypatching `evaluation.report.write_markdown_report` afterward would silently not affect that already-bound reference. Not needing the mock at all avoids this trap entirely.)

- [ ] **Step 5: Run the updated test file to confirm it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/cli.py tests/test_cli.py
git commit -m "refactor: repoint rank command at evaluation.report, delete old report module"
```

---

### Task 3: Rename `evidencerank-eval-report` to `evidencerank-report`

**Files:**
- Modify: `evaluation/cli.py`
- Modify: `evaluation/test_cli.py`
- Modify: `pyproject.toml:22`

**Interfaces:**
- Consumes: `write_json_report(state, path)`, `write_markdown_report(report_paths, path)` from `evaluation.report` (Task 1).

- [ ] **Step 1: Edit `evaluation/cli.py`**

Replace the whole file with:

```python
import click

from evidencerank.cli import run_pipeline

from evaluation.report import write_json_report, write_markdown_report


@click.command()
@click.option(
    "--reports",
    "report_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
)
@click.option("--out", default="report.md", type=click.Path())
def report(report_paths, out):
    """Build the combined ranking + evaluation report from one or more report.json files.

    Pass --reports once per report.json path. One path gives rankings and
    pipeline stats only; repeat --reports for each additional run to also
    include rank stability across runs, e.g.:

        evidencerank-report --reports a.json --reports b.json --out report.md
    """
    write_markdown_report(list(report_paths), out)
    click.echo(f"Wrote {out}")


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--runs", default=3, type=click.IntRange(min=2))
@click.option("--llm-concurrency", default=4, type=click.IntRange(min=1))
@click.option("--out", default="report.md", type=click.Path())
def rank_stability(jd_path, resumes_dir, runs, llm_concurrency, out):
    """Run the pipeline RUNS times on the same JD/resumes and build a
    combined report that includes rank stability across the runs.

    Writes run1.json, run2.json, ... (one full report.json per run, never
    overwritten) alongside OUT, so each run stays available for inspection -
    not just the aggregated report.md.
    """
    report_paths = []
    for i in range(1, runs + 1):
        click.echo(f"Run {i}/{runs}...")
        final_state = run_pipeline(jd_path, resumes_dir, llm_concurrency)
        path = f"run{i}.json"
        write_json_report(final_state, path)
        report_paths.append(path)
        click.echo(f"Wrote {path}")

    write_markdown_report(report_paths, out)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    report()
```

- [ ] **Step 2: Run the (still old) test file to see it fail**

Run: `uv run pytest evaluation/test_cli.py -v`
Expected: FAIL at collection — `ImportError: cannot import name 'eval_report' from 'evaluation.cli'` (the old test file still imports the pre-rename command name).

- [ ] **Step 3: Replace `evaluation/test_cli.py` with the updated version**

Replace the entire file content with:

```python
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
```

- [ ] **Step 4: Update the script entry in `pyproject.toml`**

Change:

```toml
evidencerank-eval-report = "evaluation.cli:eval_report"
```

to:

```toml
evidencerank-report = "evaluation.cli:report"
```

- [ ] **Step 5: Run the updated test file to confirm it passes**

Run: `uv run pytest evaluation/test_cli.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 6: Re-sync so the renamed script entry is installed**

Run: `uv sync`
Expected: exits 0. Confirm the new command is registered:
Run: `uv run evidencerank-report --help`
Expected: prints the `report` command's help text (starting with "Build the combined ranking + evaluation report...").

- [ ] **Step 7: Commit**

```bash
git add evaluation/cli.py evaluation/test_cli.py pyproject.toml uv.lock
git commit -m "refactor: rename evidencerank-eval-report to evidencerank-report"
```

---

### Task 4: Update `README.md`

**Files:**
- Modify: `README.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Update the header's spec cross-references**

Change:

```markdown
description, using local Ollama models orchestrated with LangGraph. See
`docs/superpowers/specs/2026-07-24-evidencerank-design.md` for the full design, and
`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md` /
`docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md` for the
evaluation-report tooling described below. See
`docs/superpowers/specs/2026-07-27-judge-grounding-and-hallucination-design.md` for a
later change that reorders the pipeline (Hallucination Checker now runs before the Pool
Calibrator, not after) and reworks Judge grounding — supersedes the pipeline diagram in
the original design doc.
```

to:

```markdown
description, using local Ollama models orchestrated with LangGraph. See
`docs/superpowers/specs/2026-07-24-evidencerank-design.md` for the full design, and
`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md` /
`docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md` for the
original evaluation-report tooling this section describes — see
`docs/superpowers/specs/2026-07-29-unify-report-output-design.md` for a later change
that merges that tooling's output (`evaluation-metric.md`) into `report.md`, superseding
those two docs' description of a separate evaluation-metric.md file. See
`docs/superpowers/specs/2026-07-27-judge-grounding-and-hallucination-design.md` for a
later change that reorders the pipeline (Hallucination Checker now runs before the Pool
Calibrator, not after) and reworks Judge grounding — supersedes the pipeline diagram in
the original design doc.
```

- [ ] **Step 2: Update the outputs paragraph in "Running the pipeline"**

Change:

```markdown
This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results), `report.md` (a ranked Markdown table), and
`evaluation-metric.md` (the evaluation metric report — see
[Evaluation metric report](#evaluation-metric-report) below), all written to the
directory you run `evidencerank` from. Each run overwrites the previous one's
`report.json`/`report.md`/`evaluation-metric.md` — rename them (e.g. `mv report.json
run1.json`) between runs if you need to keep more than one.
```

to:

```markdown
This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results) and `report.md` (a ranked Markdown table plus pipeline
stats — see [Evaluation metric report](#evaluation-metric-report) below for what the
stats section contains), both written to the directory you run `evidencerank` from.
Each run overwrites the previous one's `report.json`/`report.md` — rename them (e.g.
`mv report.json run1.json`) between runs if you need to keep more than one.
```

- [ ] **Step 3: Remove the now-redundant "evaluation metric report" paragraph**

Change:

```markdown
Every run also generates the evaluation metric report (`evaluation-metric.md`) — see
[Evaluation metric report](#evaluation-metric-report) below for what it contains. This is
pure computation over the run's `report.json` (no LLM calls, no extra setup beyond what
`rank` already requires).

## Model configuration
```

to:

```markdown
## Model configuration
```

- [ ] **Step 4: Update the "Research evaluation harness" intro**

Change:

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

to:

```markdown
## Research evaluation harness

`evaluation/report.py` builds every report the pipeline produces — including
`report.json` and `report.md` themselves, called directly by `src/evidencerank/cli.py`'s
`rank` command — plus the research-only aggregates appended to `report.md`'s Pipeline
Stats/Rank Stability sections (candidates submitted, pre-filter pass/drop, hallucination
rate, rank correlation across repeated runs). `evaluation/rank_stability.py` computes the
Spearman/Kendall-tau rank correlation piece.

Both the hallucination-rate and rank-stability signals are deterministic — fuzzy
string-matching for the former, rank-correlation statistics for the latter — with no LLM
judging another LLM's output involved, and no extra model or `ollama serve` requirement
beyond what the production pipeline itself already needs.
```

- [ ] **Step 5: Update the rank-stability command description**

Change:

```markdown
This runs the pipeline `--runs` times (default `3`, minimum `2`), writes each run's full
report as `run1.json`, `run2.json`, ... (never overwritten, so every run stays available
for inspection), and builds `evaluation-metric.md` from all of them — pipeline stats
from `run1.json`, rank stability (Spearman/Kendall-tau) across all of them.
`--llm-concurrency` and `--out` work the same as the other commands.

If you'd rather drive this manually (e.g. against runs you already have, or with
resumes/JD changing between runs), run the pipeline yourself N times, renaming
`report.json` after each run since every run overwrites it, then call
`evidencerank-eval-report` (below) with all the paths.
```

to:

```markdown
This runs the pipeline `--runs` times (default `3`, minimum `2`), writes each run's full
report as `run1.json`, `run2.json`, ... (never overwritten, so every run stays available
for inspection), and builds `report.md` from all of them — rankings and pipeline stats
from `run1.json`, rank stability (Spearman/Kendall-tau) across all of them.
`--llm-concurrency` and `--out` work the same as the other commands.

If you'd rather drive this manually (e.g. against runs you already have, or with
resumes/JD changing between runs), run the pipeline yourself N times, renaming
`report.json` after each run since every run overwrites it, then call
`evidencerank-report` (below) with all the paths.
```

- [ ] **Step 6: Rewrite the "Evaluation metric report" section**

Change:

```markdown
### Evaluation metric report

`uv run evidencerank-eval-report` builds a Markdown report combining pipeline stats and
(when 2+ runs are given) rank stability, from one or more existing `report.json` files:

```bash
uv run evidencerank-eval-report --reports report.json --out evaluation-metric.md
```

If you're evaluating a single run right after producing it, `evidencerank rank` (see
[Running the pipeline](#running-the-pipeline) above) already does this in the same run —
no separate command needed.

Pass `--reports` once per report path — repeat it for each additional run to also
include rank stability across runs:

```bash
uv run evidencerank-eval-report \
  --reports run1.json --reports run2.json --reports run3.json \
  --out evaluation-metric.md
```

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

to:

```markdown
### Evaluation metric report

`uv run evidencerank-report` builds a Markdown report combining the candidate rankings,
pipeline stats, and (when 2+ runs are given) rank stability, from one or more existing
`report.json` files:

```bash
uv run evidencerank-report --reports report.json --out report.md
```

If you're evaluating a single run right after producing it, `evidencerank rank` (see
[Running the pipeline](#running-the-pipeline) above) already does this in the same run —
no separate command needed.

Pass `--reports` once per report path — repeat it for each additional run to also
include rank stability across runs:

```bash
uv run evidencerank-report \
  --reports run1.json --reports run2.json --reports run3.json \
  --out report.md
```

The rankings and pipeline stats are always computed from the first `--reports` path
given; every path is used for rank stability. All of this is pure computation over
`report.json` — no `ollama serve` or model required to build this report.

When the underlying `report.json` includes per-stage timing (`stage_timings`,
added by the production pipeline), the report also includes a "Stage Timings"
table showing wall-clock seconds per stage — absent for older `report.json`
files that predate this field.

Quote authenticity is measured deterministically via Hallucination Rate in Pipeline
Stats — the hallucination checker strips unverified evidence before calibration, so
this is a direct count from that check, not a judgment call by any model.
```

- [ ] **Step 7: Verify no stray references remain**

Run: `grep -n "evaluation-metric.md\|evidencerank-eval-report" README.md`
Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: update README for the unified report.md output"
```

---

### Task 5: Final verification

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest --ignore=tests/agents/test_calibrator.py`
Expected: PASS — every test green. (`tests/agents/test_calibrator.py` is excluded because it has a pre-existing, unrelated collection failure from uncommitted changes elsewhere in the repo — not something this plan touches or fixes.)

- [ ] **Step 2: Grep the whole repo (excluding historical specs/plans and `.venv`) for leftover references**

Run: `grep -rln "evaluation-metric.md\|eval_report\|eval-report" --include="*.py" --include="*.md" --include="*.toml" . | grep -v ".venv" | grep -v "docs/superpowers/specs/" | grep -v "docs/superpowers/plans/2026-07-27"`
Expected: no output. (The excluded paths are historical spec/plan docs from before this change — `docs/superpowers/specs/*` and the older `docs/superpowers/plans/2026-07-27-eval-metric-report.md` / `2026-07-27-eval-report-cli-integration.md` — which intentionally stay as a record of past decisions, same convention as the rest of the repo's superseded docs.)

- [ ] **Step 3: Smoke-test the CLI end-to-end (manual, requires `ollama serve` running)**

Run against a small resumes subset (a full run against every resume in `resumes/` would take a long time — point `--resumes-dir` at a folder with 2-3 PDFs copied in for this check):

```bash
uv run evidencerank --jd ai_data_engineer.txt --resumes-dir <small-subset-dir> --llm-concurrency 3
```

Expected: exits 0, writes `report.json` and `report.md` only (no `evaluation-metric.md`). Open `report.md` and confirm it has `## Rankings` (with a populated table), `## Pipeline Stats`, and `## Stage Timings` sections, in that order.

Then confirm the renamed standalone command works against that same `report.json`:

```bash
uv run evidencerank-report --reports report.json --out standalone-report.md
```

Expected: exits 0, `standalone-report.md`'s content matches `report.md`'s.
