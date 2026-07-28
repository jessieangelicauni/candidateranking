# Evaluation Metric Report Design

**Date:** 2026-07-27
**Status:** Approved for planning

## Purpose

EvidenceRank's research validation (see `docs/superpowers/specs/2026-07-24-evidencerank-design.md`,
"Evaluation Harness" section) defines three GEval metrics (Groundedness, RecruiterAlignment,
EvidenceRelevancy) and a rank-stability metric, but nothing currently runs the GEval metrics
against a pipeline run and aggregates the results, and rank-stability output isn't formatted
for inclusion in a paper. This adds a report generator that consumes one or more existing
`report.json` pipeline outputs and produces a single Markdown "evaluation metric report" —
aggregate GEval scores, pipeline stats, and (when multiple runs are given) rank stability —
suitable for a Q1-Scopus-track paper appendix.

This is a research-tooling addition to `evaluation/`, separate from the production pipeline
in `src/evidencerank/`, consistent with the existing separation of concerns.

## Scope

- Runs the three existing GEval metrics (`evaluation/metrics.py`) against a report's
  `judge_results` and reports aggregate statistics per metric.
- Reports pipeline-level stats (candidates submitted, prefilter pass/drop, hallucination
  flags) from the same report.
- When 2+ report paths are given, also reports rank stability (`evaluation/rank_stability.py`)
  across all of them.
- Aggregates only — no per-candidate score breakdown table (keeps the report compact and
  paper-appendix-ready; per-candidate detail is not part of this build).
- Markdown output only.
- New CLI entry point, following the existing `evidencerank` CLI pattern.

## Architecture

```
report.json (run 1) ──┐
report.json (run 2) ──┼─→ evaluation/report.py ──→ Markdown string ──→ evaluation-metric.md
report.json (run N) ──┘         │
                                 ├─ compute_geval_scores(report[0])     (GEval, via Ollama)
                                 ├─ compute_pipeline_stats(report[0])   (pure JSON aggregation)
                                 └─ rank_stability(report[0..N])        (existing, if N >= 2)
```

`evaluation/report.py` operates directly on `report.json` files on disk — unlike
`src/evidencerank/report.py`, which builds reports from live in-memory pipeline `state`. This
keeps the evaluation harness fully decoupled from a pipeline run: it can be pointed at any
past `report.json`.

## Components

### `evaluation/report.py` (new)

- `compute_geval_scores(report_path: str | Path) -> dict`
  - Loads `report.json`.
  - For each candidate in `judge_results`, builds an `LLMTestCase` via the existing
    `build_test_case()`:
    - `jd_requirements_text`: JD fields (`title`, `required_skills`, `nice_to_have_skills`,
      `min_experience_years`, `education`, `responsibilities`) serialized to plain text.
    - `judge_result_text`: candidate's `tier`, `rating`, and evidence claims (`claim: "quote"`
      per line) serialized to plain text.
    - `cv_text`: that candidate's `raw_cv_text` from `profiles`.
  - Runs `groundedness_metric`, `recruiter_alignment_metric`, `evidence_relevancy_metric`
    (each calls the local Ollama judge model, per existing `metrics.py` config) against every
    test case.
  - Returns, per metric name: `{"n": int, "mean": float, "std": float, "pass_rate": float}`,
    where `pass_rate` is the fraction of candidates scoring >= the metric's threshold (0.7),
    and `std` is sample standard deviation (`statistics.stdev`, i.e. `ddof=1`).
  - When `judge_results` is empty (`n=0`), returns `{"n": 0, "mean": None, "std": None,
    "pass_rate": None}` rather than raising — an empty pool (e.g., everyone dropped by the
    prefilter) is a valid, reportable outcome, not an error.
  - When `judge_results` has exactly one candidate (`n=1`), sample stdev is mathematically
    undefined (`statistics.stdev` requires n>=2); `std` is `None` in this case while `mean`
    and `pass_rate` are still computed normally.

- `compute_pipeline_stats(report_path: str | Path) -> dict`
  - From the same report: `total_candidates` (`len(profiles)`), `passed_prefilter`,
    `dropped_prefilter` (`len(dropped)`), `evaluated_by_judge` (`len(judge_results)`),
    `hallucination_flagged` (count of candidates in `hallucination_reports` where
    `not all_verified`).

- `build_eval_markdown_report(report_paths: list[str | Path]) -> str`
  - Computes GEval scores and pipeline stats from `report_paths[0]`.
  - Renders:
    1. Header — JD title (from `report_paths[0]`'s `jd.title`) and the report path(s) used.
    2. Pipeline Stats table.
    3. GEval Metrics table (metric, n, mean, std, pass rate).
    4. Rank Stability table — **only present** when `len(report_paths) >= 2`; calls the
       existing `rank_stability(report_paths)` and renders `mean_spearman`,
       `mean_kendall_tau`, `n_runs`.

- `write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None`
  - Writes `build_eval_markdown_report(report_paths)` to `path`.

### `evaluation/cli.py` (new)

- `click` command `eval_report`:
  - `--reports` (required, variadic `click.Path(exists=True)`) — one or more `report.json`
    paths; first path is primary (GEval + pipeline stats), all paths feed rank stability.
  - `--out` (default `evaluation-metric.md`).
  - Calls `write_eval_markdown_report`, echoes the output path on success.
- Registered in `pyproject.toml`'s `[project.scripts]` as:
  ```
  evidencerank-eval-report = "evaluation.cli:eval_report"
  ```

### `pyproject.toml` changes

- Add the `evidencerank-eval-report` script entry above.
- No new dependencies — `deepeval`, `scipy`, `numpy` are already present.

## Data Flow

1. User runs the production pipeline one or more times (`uv run evidencerank ...`), producing
   `report.json` per run.
2. User runs `uv run evidencerank-eval-report --reports report1.json --reports report2.json
   --out evaluation-metric.md` (repeat `--reports` once per report path — one path gives GEval
   metrics + pipeline stats only; 2+ paths also add rank stability).
3. `report.py` loads the primary report, builds GEval test cases from `judge_results` +
   `profiles` + `jd`, and calls the three GEval metrics — each metric call round-trips to the
   local Ollama judge model (`EVIDENCERANK_EVAL_MODEL`, same as production judge stage).
4. If multiple reports were given, `rank_stability()` reads `calibrated_results.final_rank`
   from each and computes pairwise Spearman/Kendall-tau.
5. All aggregates are rendered into one Markdown document and written to `--out`.

## Error Handling

- Malformed or missing expected fields in a `report.json` (e.g., not produced by this
  pipeline) raise naturally (`KeyError`/`json.JSONDecodeError`/Pydantic-adjacent errors from
  `deepeval`) — this is a research tool operating on trusted, self-produced pipeline output,
  not a user-input boundary, so no defensive fallback/validation is added.
- Empty `judge_results` is handled explicitly (see `compute_geval_scores` above) since it's a
  realistic, non-error outcome (aggressive prefilter threshold), not a malformed-input case.
- Single report path: rank stability section is omitted, not an error.

## Testing

Following the existing pattern in `evaluation/test_metrics.py` and
`evaluation/test_rank_stability.py`:

- Synthetic `report.json` fixtures written to `tmp_path` with a handful of candidates
  (mirroring `_write_report` helper style already in `test_rank_stability.py`, extended with
  `jd`, `profiles`, `judge_results`, `dropped`, `hallucination_reports` keys as needed per
  test).
- GEval metric `.measure()` calls are mocked/stubbed (e.g., monkeypatching the metric objects'
  `measure` method to return fixed scores) so tests don't require a live Ollama server —
  consistent with how the rest of the suite avoids live LLM calls.
- Coverage:
  - `compute_geval_scores`: aggregate math (mean/std/pass_rate) against known stubbed scores;
    zero-candidates case returns `n=0` / `None` fields without raising.
  - `compute_pipeline_stats`: counts match a constructed fixture with known
    prefilter/dropped/hallucination data.
  - `build_eval_markdown_report`: rank-stability section present for 2+ report paths, absent
    for 1.
  - CLI: `click.testing.CliRunner` invocation writes the expected output file (with GEval
    mocked).

## Out of Scope (this build)

- Per-candidate GEval score breakdown table (aggregates only, per decision above).
- Non-Markdown output formats (LaTeX/CSV/JSON) — not requested for this build.
- Any change to the production pipeline (`src/evidencerank/`) or its existing report format.
