# Judge Shortlist Before Calibration Design

**Date:** 2026-07-27
**Status:** Approved for planning

## Purpose

`calibrate_pool()` (`src/evidencerank/agents/calibrator.py`) currently receives every
candidate that survived the prefilter and was judged — not a bounded subset. Its own prompt
already describes the input as *"a shortlisted candidate pool"* (`CALIBRATOR_PROMPT`), but no
shortlisting actually happens before `calibrate_node` calls it: the code doesn't match what the
prompt assumes.

This mismatch is also a real scalability risk, not just a wording inconsistency. The
calibrator's single prompt embeds every judge result at once, so its size scales linearly with
pool size. A recent production run with 26 candidates already exceeded Ollama's default
4096-token context window, silently dropping candidates from the calibrated ranking (fixed
separately by raising `num_ctx` to 32768 and validating the output — see
`docs/superpowers/specs/` git history for that fix). A fixed `num_ctx` ceiling just moves the
same failure mode to a larger pool size; it doesn't remove it.

This design adds an explicit shortlist step between `judge` and `calibrate` so the calibrator's
input is reduced to the judge's top 10 by rating in the common case, and so the code matches its
own stated intent. This is not a hard bound: `JudgeResult.rating` only has 10 possible values
(1-10), so ties at the boundary are common, not rare, and every tied candidate is kept (see
Scope below) — a pool where many candidates share the top rating can still send the whole pool
to the calibrator. The actual backstop against silent data loss remains the existing
`CALIBRATOR_NUM_CTX` (32768) plus the output-count validation in `calibrate_pool()`
(`src/evidencerank/agents/calibrator.py`); this shortlist reduces how often that backstop is
needed, it does not replace it.

## Scope

- New pure selection function choosing the top 10 judged candidates by `rating`, with ties at
  the 10th-place boundary all included (not an arbitrary cut).
- New pipeline stage (`shortlist`) wired between `judge` and `calibrate`.
- `calibrate_pool()` is called with only the shortlisted subset.
- `report.json` gains a `not_shortlisted` list (same shape as the existing `dropped` list) for
  audit-trail transparency; `judge_results` in `report.json` is unchanged — it still contains
  every judged candidate, shortlisted or not.
- `report.md`'s ranked table is unaffected in structure — it continues to render
  `calibrated_results`, which now reflects only the shortlist.
- Shortlist size (10) is a hardcoded constant, not a CLI flag. No new CLI surface in this build.

## Architecture

```
extract_profiles → prefilter → judge → shortlist → calibrate → hallucination_check
```

`shortlist` is a new node, following the same pattern as `prefilter`: a pure, non-LLM function
in its own module under `agents/`, called from a thin `*_node` wrapper in `graph.py`.

## Components

### `src/evidencerank/agents/shortlist.py` (new)

```python
def select_shortlist(
    judge_results: list[JudgeResult], size: int = 10
) -> tuple[list[JudgeResult], list[dict]]:
```

- Sorts `judge_results` by `rating` descending.
- If `len(judge_results) <= size`, returns `(judge_results, [])` — everyone is shortlisted, no
  behavior change for small pools (a realistic case: many runs will have fewer than 10
  candidates total).
- Otherwise, finds the `rating` value of the candidate at index `size - 1` in the sorted list
  (the cutoff), and includes every candidate whose `rating >= cutoff` — so a multi-way tie at
  the boundary is never arbitrarily split. The shortlist can therefore be larger than `size`
  when ties extend past the boundary, by design.
- Returns `(shortlisted, not_shortlisted)`, where `not_shortlisted` is
  `[{"candidate_id": ..., "reason": "ranked outside judge's top 10 by rating"} for ...]`,
  mirroring the existing `dropped` shape produced by `prefilter_node`.

### `src/evidencerank/graph.py` (modified)

- New `PipelineState` keys: `shortlisted_ids: set[str]`, `not_shortlisted: list[dict[str, str]]`.
- New `shortlist_node`:
  ```python
  def shortlist_node(state: PipelineState) -> dict:
      click.echo("Running stage: shortlist")
      shortlisted, not_shortlisted = select_shortlist(list(state["judge_results"].values()))
      return {
          "shortlisted_ids": {result.candidate_id for result in shortlisted},
          "not_shortlisted": not_shortlisted,
      }
  ```
- `calibrate_node` changes to filter before calling `calibrate_pool`:
  ```python
  def calibrate_node(state: PipelineState) -> dict:
      click.echo("Running stage: calibrate")
      pool = [
          result
          for result in state["judge_results"].values()
          if result.candidate_id in state["shortlisted_ids"]
      ]
      calibrated = calibrate_pool(state["jd"], pool)
      return {"calibrated_results": calibrated}
  ```
- `build_graph()` inserts the new node and rewires edges:
  `judge → shortlist → calibrate` (replacing `judge → calibrate`).

### `src/evidencerank/report.py` (modified)

- `build_json_report()` adds `"not_shortlisted": state.get("not_shortlisted", [])` to the
  returned dict, alongside the existing `"dropped"` key.
- No change to `build_markdown_report()` — it already renders from `calibrated_results` only,
  which now naturally reflects the shortlist.

### `README.md` (modified)

- Update the stage list (`extract_profiles, prefilter, judge, calibrate,
  hallucination_check`) to include `shortlist`, and add a short paragraph explaining that only
  the judge's top 10 (by rating, ties included) proceed to calibration, with everyone else
  still fully recorded in `report.json`'s `judge_results` and newly added `not_shortlisted`.

## Data Flow

1. `judge_node` produces `judge_results` for every prefilter-passed candidate (unchanged).
2. `shortlist_node` reads `judge_results`, calls `select_shortlist`, and stores
   `shortlisted_ids` + `not_shortlisted` in state.
3. `calibrate_node` filters `judge_results` down to `shortlisted_ids` before building the
   calibrator prompt — reducing its size in the common case, though a pool with many
   candidates tied at the same top rating can still produce a shortlist as large as the
   original pool (see Purpose above).
4. `hallucination_check_node` is unaffected — it already iterates `judge_results` (the full
   judged pool), independent of calibration/shortlisting.
5. `report.py` writes `not_shortlisted` into `report.json` and renders `report.md` from
   `calibrated_results` as before.

## Error Handling

- No new failure modes are introduced. `select_shortlist` is a pure sort/filter over data
  already validated by `JudgeResult`'s pydantic schema — nothing here can raise on well-formed
  input.
- The existing calibrator output-count validation (comparing `calibrate_pool`'s returned
  candidate IDs against its input) continues to apply, now checked against the shortlist
  instead of the full pool.
- Empty `judge_results` (e.g., everyone dropped by the prefilter) naturally produces an empty
  shortlist and an empty `not_shortlisted` list — not an error, consistent with how empty
  pools are already handled elsewhere in the pipeline.

## Testing

- `tests/agents/test_shortlist.py` (new):
  - Exactly-10-or-fewer candidates: all shortlisted, `not_shortlisted` empty.
  - More than 10, no ties at the boundary: top 10 by rating shortlisted, exact
    `not_shortlisted` entries for the rest with the expected reason string.
  - Ties at the boundary (e.g., 12 candidates, 3-way tie for what would be rank 10): all tied
    candidates included, shortlist size > 10.
  - Empty input: returns `([], [])`.
- `tests/test_graph.py` (extended): `calibrate_pool` (mocked) is called with only the
  shortlisted subset of `judge_results`, not the full dict's values; `shortlisted_ids` and
  `not_shortlisted` land correctly in state.
- `tests/test_report.py` (extended): `build_json_report` includes `not_shortlisted` from state.

## Out of Scope (this build)

- Configurable shortlist size (CLI flag) — hardcoded at 10 per explicit decision; can be added
  later if needed.
- Any change to how `judge_results` or `hallucination_reports` are computed or stored — the
  shortlist only affects what reaches `calibrate_pool`.
- Surfacing `not_shortlisted` in `report.md` — kept out of the human-facing ranked table per
  explicit decision; `report.json` remains the full audit trail.
