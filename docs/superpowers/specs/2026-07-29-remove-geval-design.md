# Remove GEval from the Evaluation Harness

**Date:** 2026-07-29
**Status:** Approved for planning

## Purpose

`evaluation/metrics.py` defines two DeepEval `GEval` metrics (`RecruiterAlignment`,
`EvidenceRelevancy`) that use a local Ollama model as an LLM-as-judge to grade the
production pipeline's Judge output. This is a self-referential signal — one LLM grading
another LLM's output — with no ground truth anchor (no human-labeled data to validate
against). It adds LLM-judge noise to the evaluation report and requires `ollama serve` +
a second model just to produce `evaluation-metric.md`, even for the deterministic parts of
the report.

This removes GEval entirely. The evaluation report is left with only its deterministic
signals — Pipeline Stats, Hallucination Rate, Stage Timings, and (when 2+ runs are given)
Rank Stability — none of which involve an LLM judging another LLM. A future,
ground-truth-anchored replacement (e.g. a human-labeled gold set) is explicitly out of
scope here and would be its own separate design.

This supersedes the GEval-related portions of
`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md` and
`docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md` — their
Pipeline Stats / Rank Stability content still stands.

## Scope

- Delete the GEval metric definitions and their dedicated test file.
- Remove GEval scoring from the evaluation report builder and its Markdown output.
- Remove the `max_concurrency` plumbing that existed solely to bound concurrent GEval
  calls, everywhere it was threaded through (report builder, both CLI commands, the
  production `rank` command).
- Drop the now-unused `deepeval` dependency and `EVIDENCERANK_EVAL_MODEL` env var.
- Update `README.md` to describe the leaner evaluation harness and drop the Ollama
  requirement for `evaluation-metric.md` generation.
- No change to the production pipeline's ranking behavior, `report.json`, or `report.md` —
  this only touches the evaluation harness (`evaluation/`) and the eval-report wiring in
  `src/evidencerank/cli.py`.

## Components

### Deleted

- `evaluation/metrics.py`
- `evaluation/test_metrics.py`

### `evaluation/report.py`

- Remove `compute_geval_scores` and its GEval/DeepEval imports.
- `build_eval_markdown_report`:
  - Drops the "## GEval Metrics" section.
  - Drops the note contrasting hallucination-rate with the GEval signals (that note only
    made sense next to a GEval section that no longer exists).
- `build_eval_markdown_report` / `write_eval_markdown_report` drop the `max_concurrency`
  parameter — nothing left in this module makes LLM calls, so there's nothing left to
  bound concurrency on.

### `evaluation/cli.py`

- `eval_report` command: remove `--llm-concurrency` (it only ever fed GEval concurrency).
- `rank_stability` command: **keeps** `--llm-concurrency` — it still bounds each pipeline
  run's `extract_profiles`/`judge` LLM concurrency via `run_pipeline`. Stops passing it to
  `write_eval_markdown_report`.

### `src/evidencerank/cli.py`

- `rank` command's call to `write_eval_markdown_report(...)` drops the
  `max_concurrency=llm_concurrency` kwarg.

### `pyproject.toml`

- Remove the `deepeval>=1.1` dependency (nothing else in the repo imports it). `scipy`
  stays — `evaluation/rank_stability.py` still uses it.

### `.env.example` / `README.md`

- Remove `EVIDENCERANK_EVAL_MODEL` from `.env.example`.
- `README.md`: remove the `ollama pull qwen3:14b` setup step, the
  `EVIDENCERANK_EVAL_MODEL` environment variable row, and rewrite the "Research
  evaluation harness" section to describe only Hallucination Rate and Rank Stability.
  Note the behavior change explicitly: producing `evaluation-metric.md` (whether via
  `evidencerank rank` or `evidencerank-eval-report`) no longer requires `ollama serve` or
  any model — it's pure computation over `report.json`.

## Behavior Change

Today, every `evidencerank rank` run requires `ollama serve` plus the eval judge model
just to produce `evaluation-metric.md`, because of the GEval calls inside it. After this
change, `evaluation-metric.md` generation is pure computation over an existing
`report.json` (string fuzzy-matching for hallucination rate, already done earlier in the
pipeline; scipy stats for rank stability) — no Ollama dependency for the eval report.

## Testing

- `evaluation/test_report.py`: remove the `compute_geval_scores` tests and
  `test_build_eval_markdown_report_includes_geval_signal_caveat`. Update the remaining
  `build_eval_markdown_report` tests to assert `"## GEval Metrics" not in markdown`
  (instead of asserting it's present), and stop mocking/passing the GEval metrics.
- `evaluation/test_cli.py`: remove the `--llm-concurrency`/GEval-concurrency tests for
  `eval_report`. Update `test_rank_stability_passes_llm_concurrency_through_to_each_run`
  to drop its assertion on `eval_report_calls[0]["max_concurrency"]`.
- `tests/test_cli.py`: drop the `"## GEval Metrics" in content` assertion and the
  related max-concurrency-passthrough assertion for the eval report call.

## Out of Scope

- Any replacement or ground-truth-anchored metric (e.g. a human-labeled gold set) — a
  separate future project.
- Changes to the production pipeline's ranking logic, `report.json`, or `report.md`.
- The currently uncommitted, unrelated changes in `calibrator.py`, `cv_extractor.py`,
  `jd_parser.py`, `judge.py`.
- Editing the historical spec docs this supersedes — they stay as a record of past
  decisions, per existing repo convention (see how `README.md` cross-references
  superseding specs rather than editing old ones).
