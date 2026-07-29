# Unify report.md and evaluation-metric.md into One Report Module

**Date:** 2026-07-29
**Status:** Approved for planning

## Purpose

Today a single `evidencerank rank` run produces three output files from two separate
modules: `report.json` and `report.md` from `src/evidencerank/report.py` (built from the
live in-memory pipeline `state`), and `evaluation-metric.md` from `evaluation/report.py`
(built by re-reading `report.json` from disk). Now that
`docs/superpowers/specs/2026-07-29-remove-geval-design.md` has removed the GEval LLM
calls from the latter, both modules do the same kind of work — render pipeline results
as Markdown — split across two files and two locations for no reason beyond history.
This merges them into one module (`evaluation/report.py`) producing two output files
instead of three: `report.json` (unchanged) and a single `report.md` containing the
ranking table plus the pipeline stats, stage timings, and rank-stability sections that
used to live in `evaluation-metric.md`.

## Scope

- Delete `src/evidencerank/report.py`; move `build_json_report`/`write_json_report` into
  `evaluation/report.py` unchanged.
- Rewrite the ranking-table builder to read a parsed `report.json` dict instead of live
  pipeline `state` (Pydantic objects), so it can run from the same on-disk source the
  stats/rank-stability sections already use — this is what makes a single combined
  `build_markdown_report` possible, including for the standalone CLI commands that never
  have live `state` to begin with.
- `evaluation-metric.md` stops being produced anywhere. `report.md` becomes the one
  Markdown output, containing everything both files used to contain.
- Rename the standalone `evidencerank-eval-report` CLI command to `evidencerank-report`
  (its job is no longer specifically "evaluation" once it's just "the report").
- Update `README.md`, `pyproject.toml`'s script entry, and all affected tests.
- No change to `report.json`'s structure, to ranking/calibration logic, or to
  `evaluation/rank_stability.py` (stays a separate module, imported by the merged
  `evaluation/report.py` exactly as it is today).

## Architecture

The pre-existing split was: `src/evidencerank/report.py` renders from live `state`
(only available at the moment the pipeline finishes running), while
`evaluation/report.py` renders from `report.json` re-read off disk (so the standalone
`evidencerank-eval-report`/`evidencerank-rank-stability` commands can regenerate a report
from a past run without re-executing the pipeline). This design keeps that disk-based
pattern and extends it to cover the ranking table too, rather than keeping two separate
rendering code paths. The `rank` command now writes `report.json` first, then builds
`report.md` by reading that file back — the same "read what was just written" pattern
`evaluation/report.py` already used for the stats section, just widened to the whole
file. This means every code path that produces `report.md` — live pipeline run or
standalone regeneration from saved `report.json` files — goes through the exact same
function.

```
Before:
  live state ──→ src/evidencerank/report.py ──→ report.json, report.md
  report.json ──→ evaluation/report.py ──→ evaluation-metric.md

After:
  live state ──→ evaluation/report.py:write_json_report ──→ report.json
  report.json ──→ evaluation/report.py:write_markdown_report ──→ report.md
```

## Components

### `evaluation/report.py` (rewritten — single module for both outputs)

- `build_json_report(state: dict) -> dict` / `write_json_report(state: dict, path) -> None`
  — moved verbatim from `src/evidencerank/report.py`, no behavior change.
- `_escape_table_cell(text: str) -> str` — moved verbatim (pure string logic, unaffected
  by the state-vs-dict source change).
- `_format_evidence(judge_result: dict | None) -> str` — ported from the old
  state-based version; reads `judge_result["evidence"]`, and each claim's
  `claim["claim"]` / `claim["quote"]` (dict keys — matches the shape
  `EvidenceClaim.model_dump()` already produces inside `report.json`), instead of the
  old `.evidence`, `.claim`, `.quote` attribute access on Pydantic objects.
- `_format_hallucination_flag(hallucination_report: dict | None) -> str` — ported
  identically, using `report["unverified_quotes"]` instead of `report.unverified_quotes`.
- `_build_ranking_table(data: dict) -> str` — the old `build_markdown_report(state)`
  logic, now reading `data["calibrated_results"]` (sorted by `result["final_rank"]`),
  `data["judge_results"]`, and `data["hallucination_reports"]` from a parsed `report.json`
  dict. Tier displays as `result["tier"]` directly (already a plain string in the JSON —
  `Tier` is a `str` `Enum`, so `model_dump()` + `json.dumps()` already serializes it as
  its value, e.g. `"Strong Fit"`, with no `.value` access needed on the dict).
- `compute_pipeline_stats(report_path: str | Path) -> dict` — unchanged.
- `build_markdown_report(report_paths: list[str | Path]) -> str` — replaces
  `build_eval_markdown_report`. Renders, in order:
  1. `# Candidate Ranking Report` title.
  2. `**JD:** <title>` / `**Primary report:** <path>` (and `**Additional reports (rank
     stability):**` when 2+ paths, same as today).
  3. `## Rankings` — the table from `_build_ranking_table(data)`, where `data` is the
     primary report's parsed JSON (`report_paths[0]`).
  4. `## Pipeline Stats` — unchanged content.
  5. `## Stage Timings` — unchanged content, only when `stage_timings` is present.
  6. `## Rank Stability` — unchanged content, only when `len(report_paths) >= 2`.
- `write_markdown_report(report_paths: list[str | Path], path: str | Path) -> None` —
  replaces `write_eval_markdown_report`.

### `src/evidencerank/cli.py`

- Delete `OUT_EVAL_REPORT`.
- `rank` command:
  ```python
  from evaluation.report import write_json_report, write_markdown_report
  ...
  write_json_report(final_state, OUT_JSON)
  write_markdown_report([OUT_JSON], OUT_MD)
  click.echo(f"Wrote {OUT_JSON} and {OUT_MD}")
  ```
  One `click.echo` instead of two — there is only one Markdown file now.

### `evaluation/cli.py`

- `eval_report` command and function renamed to `report`. `--out` default changes from
  `evaluation-metric.md` to `report.md`. Docstring updated to describe rebuilding
  `report.md` rather than an "evaluation metric report".
- `rank_stability` command: `--out` default changes to `report.md`; docstring's mention
  of "the aggregated evaluation-metric.md" becomes "report.md". Both commands' calls to
  `write_eval_markdown_report` become `write_markdown_report`.
- `write_json_report` import shifts from `evidencerank.report` to the local
  `evaluation.report` (same module now).

### `pyproject.toml`

- Script entry `evidencerank-eval-report = "evaluation.cli:eval_report"` becomes
  `evidencerank-report = "evaluation.cli:report"`.

### `README.md`

- Every mention of `evaluation-metric.md`, `evidencerank-eval-report`, and the
  three-output-file description is updated to describe the two outputs (`report.json`,
  `report.md`) and the renamed `evidencerank-report` command.

## Data Flow

1. `evidencerank rank` runs the pipeline, gets back live `state`.
2. `write_json_report(state, "report.json")` — unchanged from today.
3. `write_markdown_report(["report.json"], "report.md")` — reads `report.json` back off
   disk (the file just written in step 2) and renders the full combined report.
4. `evidencerank-report --reports a.json --reports b.json --out report.md` — same
   `write_markdown_report`, invoked standalone against existing report files, with
   rank-stability included once 2+ paths are given. This is the same call as step 3,
   just invoked directly by a user against saved files rather than by the `rank` command
   against a file it just wrote — there is no separate implementation for either path.
5. `evidencerank-rank-stability` — unchanged flow otherwise: runs the pipeline N times,
   writes `run1.json`, `run2.json`, ..., then calls `write_markdown_report` across all of
   them to produce one `report.md` with rank stability included.

## Testing

- `tests/test_report.py` is deleted.
- Its `build_json_report`/`write_json_report` tests move into `evaluation/test_report.py`
  verbatim except for the import path (`evidencerank.report` → `evaluation.report`).
- Its ranking-table tests (row ordering, pipe/newline escaping in evidence and
  calibration notes, hallucination-flag rendering) move into `evaluation/test_report.py`,
  rewritten to write a `report.json` fixture to `tmp_path` — following the existing
  `_write_report`/`_write_calibrated_report` helper pattern already in that file — and
  call `build_markdown_report([report_path])` instead of constructing live `state` with
  Pydantic model instances and calling the old `build_markdown_report(state)` directly.
- `evaluation/test_report.py`'s existing `build_eval_markdown_report`/
  `write_eval_markdown_report` tests are renamed to the new function names, with added
  assertions that `## Rankings` and a ranking-table row appear in the combined output.
- `evaluation/test_cli.py` and `tests/test_cli.py`: rename references from `eval_report`
  to `report`, and update `--out`/default-filename assertions from
  `evaluation-metric.md` to `report.md`.

## Out of Scope

- Any change to `report.json`'s structure or to ranking/calibration logic.
- Any change to `evaluation/rank_stability.py` — stays a separate, single-purpose module
  imported by `evaluation/report.py`, same as today.
- Renaming `run1.json`/`run2.json` or any other filename not explicitly covered above.
