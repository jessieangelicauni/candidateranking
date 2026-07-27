# Judge Grounding, Evidence Relevancy, and Corrective Hallucination Checking

**Date:** 2026-07-27
**Status:** Approved for planning

## Purpose

A pipeline run against the 26 resumes in `resumes/` (see `report.json`, `report.md`,
`eval_report.md`) scored badly on all three GEval metrics (Groundedness 7.7% pass,
RecruiterAlignment 11.5% pass, EvidenceRelevancy 0% pass) with 14/26 candidates
hallucination-flagged. Investigation of the actual evidence chains (not just the
aggregate numbers) found specific, fixable root causes rather than a generic
"small local model" problem:

1. The Judge prompt (`src/evidencerank/agents/judge.py`) never scopes "verbatim
   quote" to the raw CV text block — it shows the structured `skills` /
   `work_history` / `education` / `projects` fields in the same prompt with no
   instruction that they're context-only. The Judge quotes from them anyway (e.g.
   `skills: ['TensorFlow']` — literal Python list syntax — appears as "evidence" in
   `report.md`). That string doesn't exist in the resume, so it fails both the
   hallucination checker and GEval Groundedness.
2. Separately, some quotes *are* genuinely verbatim but don't support the specific
   claim they're attached to (e.g. a generic "results-driven engineer, 4+ years"
   bio line used as evidence for "over two years of ML experience"). This is a
   claim-to-evidence linking problem, not a fabrication problem, and explains why
   EvidenceRelevancy is worse than Groundedness.
3. `hallucination_check` runs *after* `calibrate` in the graph and its output is
   never used — flagged candidates aren't re-judged, demoted, or even marked in
   `report.md`. A hallucination-flagged candidate can rank #1 with no visible
   signal to a human reviewer.
4. The fuzzy-match hallucination check (`rapidfuzz.fuzz.partial_ratio`) compares
   raw, unnormalized text, so legitimate verbatim quotes can score below threshold
   purely from whitespace/line-wrap differences — adding false-positive flags on
   top of the true positives from (1).
5. No stage timing exists today, so latency has no visibility in the eval report.

This design fixes 1–5. It does not change the production ranking algorithm's
philosophy (holistic recruiter judgment, tier + rating) — only how evidence is
sourced, verified, and surfaced.

## Scope

**In scope:**
- Judge prompt changes to eliminate off-limits quoting and improve claim/evidence
  relevance (production pipeline, `src/evidencerank/agents/judge.py`).
- Reordering the graph so hallucination checking happens before calibration, and
  making it corrective: unverified evidence is stripped from a candidate's Judge
  result before calibration sees it (`src/evidencerank/graph.py`).
- Surfacing the hallucination signal in `report.md` (`src/evidencerank/report.py`).
- Normalizing text before fuzzy matching to cut false-positive flags
  (`src/evidencerank/agents/hallucination_checker.py`).
- Per-stage timing captured in `report.json` and rendered in `eval_report.md`
  (`src/evidencerank/graph.py`, `src/evidencerank/report.py`,
  `evaluation/report.py`).

**Out of scope (explicitly not doing now):**
- Swapping the Judge/Calibrator model (e.g. qwen2.5:7b vs 14b). This is a
  post-implementation experiment, not a code change — `EVIDENCERANK_MODEL_JUDGE`
  already supports it (see README). Once the prompt fixes land, re-run the
  pipeline with the env var overridden and compare GEval scores and latency. No
  design work needed for this; it's a validation step at the end of the
  implementation plan.
- Any new LLM-based self-critique/re-judging call. The hallucination check stays
  deterministic (fuzzy string matching), consistent with the existing design
  rationale in `docs/superpowers/specs/2026-07-24-evidencerank-design.md` (avoid
  an LLM checking an LLM for hallucination).
- Changing the `EvidenceClaim` schema (e.g. adding a machine-checkable
  skill-reference field). The relevance fix is prompt-only for now; if it proves
  insufficient after re-measurement, that's a follow-up design.

## Changes

### 1. Judge prompt: scope quoting to the CV text only

`JUDGE_PROMPT` in `judge.py` currently shows `redacted_cv_text` and the structured
profile fields back-to-back with one shared instruction. Split the instruction so
it explicitly says quotes must come only from the "Candidate resume" text block,
and that the structured fields below it are background context and must never be
quoted verbatim (they're paraphrased/normalized by the CV extractor and won't
literally appear in the source text). Add one negative example inline (a
sentence showing a disallowed quote drawn from a structured field) so the model
has a concrete pattern to avoid, not just an abstract rule.

### 2. Judge prompt: require claim-relevant quotes

Add an instruction that each quote must itself mention or directly demonstrate
the specific skill/technology/responsibility named in its claim — not merely be
true and present somewhere in the resume. Include one positive/negative example
pair: a quote that only establishes years-of-experience-in-general being
rejected as evidence for a specific technical claim, versus a quote that
actually names the relevant tech. This is a prompt-only change; no schema or
validation code changes.

### 3. Reorder the graph; make hallucination checking corrective

Change the edge order from
`judge → calibrate → hallucination_check → END` to
`judge → hallucination_check → calibrate → END`.

`hallucination_check_node` gains a second responsibility: after computing each
candidate's `HallucinationReport`, it produces a filtered copy of that
candidate's `JudgeResult` with any evidence item whose quote appears in
`unverified_quotes` removed, and that filtered version (tier/rating unchanged,
evidence pruned) is what flows into `calibrate_node` and into
`judge_results` in the final report. The full `HallucinationReport` (including
the removed quotes) is still recorded in `hallucination_reports` for the audit
trail — nothing is silently discarded, it's just excluded from calibration input
and from the evidence shown in `report.md`.

Expected side effect (not a bug): once evidence is pre-filtered, GEval
Groundedness on the final report should trend toward ~100%, since by
construction the remaining quotes already passed fuzzy verification. Note this
in the eval report's interpretation so a future reader doesn't mistake it for
the metric becoming meaningless — RecruiterAlignment and EvidenceRelevancy
remain the informative signals post-fix.

### 4. Surface hallucination flags in `report.md`

Add a "Hallucination Flags" column to the Markdown table in
`build_markdown_report()` (`src/evidencerank/report.py`), sourced from
`state["hallucination_reports"][candidate_id]`. Show e.g. `2 removed` or `—` (or
similar concise marker) so a human reviewer can see, next to each ranked
candidate, whether evidence was pruned for them.

### 5. Normalize text before fuzzy matching

In `hallucination_checker.py`, collapse whitespace (newlines, repeated spaces)
in both `claim.quote` and `raw_cv_text` before calling
`fuzz.partial_ratio`, so formatting differences alone don't cause a false
"unverified" flag. Keep the existing `DEFAULT_THRESHOLD = 85.0` and the
`threshold` parameter/CLI flag unchanged — only the inputs to the comparison
change, not the scoring function or threshold.

### 6. Per-stage timing

Wrap each node function in `graph.py` (or the graph-building loop) to record
wall-clock start/end around its body, accumulating into a new
`stage_timings: dict[str, float]` (seconds) key on `PipelineState`. Include it
in `build_json_report()`'s output. In `evaluation/report.py`, add a "Stage
Timings" section to the Markdown report when `stage_timings` is present in the
loaded `report.json` (guard for absence so older `report.json` files without
this key don't break report generation).

## Testing

- `tests/agents/test_judge.py`: extend to assert the prompt template contains
  the new scoping/relevance instructions (string-presence checks, consistent
  with existing prompt tests in this file).
- `tests/agents/test_hallucination_checker.py`: add cases for whitespace/newline
  normalization (a quote that differs only in line-wrapping should now verify).
- `tests/test_graph.py`: update for the new edge order
  (`judge → hallucination_check → calibrate`) and assert that a candidate whose
  evidence is fully/partially unverified reaches `calibrate_node` with that
  evidence pruned, while `hallucination_reports` still shows the original
  unverified quotes.
- `tests/test_report.py`: assert the new Hallucination Flags column renders
  correctly for both flagged and clean candidates.
- `evaluation/report.py` tests (if present) or new coverage: assert stage
  timings render when present and the report still builds when absent.
- Existing test suite must continue to pass (`uv run pytest`).

## Validation plan (post-implementation)

1. Re-run `uv run evidencerank ... --with-eval-report` against the same 26
   resumes and JD.
2. Compare `eval_report.md` before/after: hallucination-flagged count,
   Groundedness/RecruiterAlignment/EvidenceRelevancy pass rates, new stage
   timings.
3. If RecruiterAlignment/EvidenceRelevancy still lag after the prompt fixes,
   re-run once more with `EVIDENCERANK_MODEL_JUDGE=qwen2.5:7b-instruct` (no code
   change needed) to see whether the smaller model is now "good enough" given a
   cleaner prompt — informs a future latency-vs-quality decision, not part of
   this implementation.
