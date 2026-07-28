# Eval Report CLI Integration Design

**Date:** 2026-07-27
**Status:** Approved for planning

## Purpose

`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md` added a standalone
`evidencerank-eval-report` CLI that builds a Markdown evaluation report (GEval metrics +
pipeline stats + optional rank stability) from one or more existing `report.json` files.
Today that requires two separate commands: `evidencerank rank` (production pipeline) and
then `evidencerank-eval-report` (evaluation harness) pointed at the output. This adds an
opt-in flag to `evidencerank rank` so a single command can produce the ranking output and
the evaluation report together, for the common case of evaluating a single run right after
producing it.

## Scope

- Add `--with-eval-report` (flag, default off) and `--out-eval-report` (path, default
  `evaluation-metric.md`) options to the `rank` command in `src/evidencerank/cli.py`.
- When `--with-eval-report` is set, after writing `report.json`/`report.md` as today, call
  the existing `evaluation.report.write_eval_markdown_report` on the just-written
  `report.json` path (single-run: GEval metrics + pipeline stats, no rank stability section,
  matching existing single-path behavior).
- No changes to `evaluation/report.py`, `evaluation/cli.py`, `evaluation/metrics.py`,
  `evaluation/rank_stability.py`, or the pipeline graph/agents (`src/evidencerank/graph.py`,
  `src/evidencerank/agents/*`).
- No changes to the standalone `evidencerank-eval-report` CLI — it continues to exist
  unchanged for multi-run (rank-stability) use and for re-running eval-report against past
  `report.json` files without re-running the pipeline.

## Architectural Note

The `evaluation/` package is deliberately decoupled from `src/evidencerank/` (per the prior
design spec): it reads `report.json` from disk only, never live pipeline `state`, and
`evaluation/` has never imported from `src/evidencerank/`. This feature adds the first
import in the other direction — `src/evidencerank/cli.py` importing
`evaluation.report.write_eval_markdown_report`. This is scoped narrowly and deliberately:
only the CLI orchestration layer (`cli.py`) gains this import; `graph.py` and the agent
modules are untouched and remain unaware `evaluation/` exists. The call still only reads the
`report.json` file `rank` just wrote to disk — it does not pass in-memory pipeline `state` —
so the "evaluation reads only from disk" boundary holds; only the import direction at the
CLI edge changes.

## Components

### `src/evidencerank/cli.py` (modify)

- New `@click.option("--with-eval-report", is_flag=True, default=False)`.
- New `@click.option("--out-eval-report", default="evaluation-metric.md", type=click.Path())`.
- `rank()` signature gains `with_eval_report` and `out_eval_report` parameters.
- After the existing `write_json_report` / `write_markdown_report` calls and their
  `click.echo`, add:
  ```python
  if with_eval_report:
      from evaluation.report import write_eval_markdown_report

      write_eval_markdown_report([out_json], out_eval_report)
      click.echo(f"Wrote {out_eval_report}")
  ```
  The import is local to the `if` branch (not a top-of-file import) so that running plain
  `evidencerank rank` (the common case) never imports `deepeval`/GEval machinery or requires
  it to be installed correctly — only paying that cost when the flag is actually used.

## Data Flow

1. `evidencerank rank --jd ... --resumes-dir ... --with-eval-report` runs the production
   pipeline exactly as today, writing `report.json` and `report.md`.
2. If `--with-eval-report` was passed, `cli.py` calls
   `write_eval_markdown_report([out_json], out_eval_report)` — the same function the
   standalone `evidencerank-eval-report` CLI calls — passing only the `report.json` path
   just written (a single-element list, so no rank-stability section is produced, matching
   existing behavior for a single report path).
3. That function internally re-reads `report.json` from disk (it does not receive the
   in-memory pipeline `state`) and calls the local Ollama eval judge model
   (`EVIDENCERANK_EVAL_MODEL`) for GEval scoring, same as the standalone CLI.

## Error Handling

If `write_eval_markdown_report` raises (e.g. Ollama unreachable, eval model not pulled), the
exception propagates and the `rank` command exits non-zero — consistent with the existing
"fail loudly, no swallowing" convention already established for the evaluation harness (see
the prior design spec's Error Handling section). `report.json` and `report.md` have already
been written to disk by this point in the command, so a failure in this step does not lose
the ranking output — only the eval report is missing, and the non-zero exit code makes that
failure visible rather than silently continuing.

## Testing

Extend `tests/test_cli.py` (existing tests already cover the `rank` command with mocked
pipeline/report-writing):

- New test: invoke `rank` with `--with-eval-report` (mocking the three GEval metrics'
  `.measure()`, same monkeypatch pattern as `evaluation/test_cli.py`), assert
  `evaluation-metric.md` is written and contains the expected `## Pipeline Stats` / `## GEval
  Metrics` sections.
- New test: invoke `rank` without `--with-eval-report` (default), assert no eval-report file
  is written — confirms the flag is genuinely opt-in and that plain `rank` runs are
  unaffected (no eval-report file appears, no GEval metrics invoked).

## Out of Scope (this build)

- No change to the standalone `evidencerank-eval-report` CLI or its `--reports`/`--out`
  interface.
- No multi-run / rank-stability support from `evidencerank rank` — that remains the
  standalone CLI's job (pointing it at multiple `report.json` files from separate `rank`
  invocations).
- No new environment variables or config — reuses `EVIDENCERANK_EVAL_MODEL` as already
  documented.
