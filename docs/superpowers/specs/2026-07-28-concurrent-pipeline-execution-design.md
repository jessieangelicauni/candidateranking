# Concurrent Pipeline Execution

**Date:** 2026-07-28
**Status:** Approved for planning

## Purpose

Every LLM-calling stage in the pipeline processes candidates strictly one at
a time: `extract_profiles_node`, `judge_node`, and `hallucination_check_node`
(`src/evidencerank/graph.py`) all loop over candidates and block on a
synchronous `.invoke()` call before starting the next. This means the
pipeline never issues more than one request to Ollama at a time, so it never
exercises Ollama's own concurrent-request handling (`OLLAMA_NUM_PARALLEL`,
continuous batching) or benefits from GPU parallelism across requests —
wall-clock time for `extract_profiles` and `judge` scales linearly with
candidate count regardless of available hardware. Separately,
`prefilter_candidate` (`src/evidencerank/agents/prefilter.py`) re-embeds the
JD's required-skills text from scratch on every single candidate instead of
embedding it once, which is wasted redundant work independent of concurrency.

This adds bounded concurrency to the `extract_profiles` and `judge` stages,
and fixes the redundant JD re-embedding in `prefilter`, to reduce wall-clock
time for a single pipeline run on a single-GPU setup.

Scope is deliberately narrow: this is a single-run speed improvement, not a
caching strategy. Repeat-run cost (e.g. re-judging the same pool across
threshold-tuning runs) is a separate concern and out of scope here.

## Design

**Mechanism:** every LLM call in `extract_profiles`/`judge` already goes
through a LangChain `Runnable` (`get_chat_model(...).with_structured_output(...)`),
and every `Runnable` has a `.batch(inputs, config={"max_concurrency": N})`
method that dispatches inputs concurrently (thread-pool based) and returns
results in input order. Both stages switch from a per-candidate `.invoke()`
loop to building the full list of prompts up front and issuing one
`.batch()` call. No new dependency; `calibrate_node` is untouched (it's
already a single call over the whole pool) and so is
`hallucination_check_node` (pure CPU `rapidfuzz` matching, no LLM call —
already fast enough at this scale that adding concurrency isn't worth the
complexity).

**Concurrency configuration:** a new CLI option, `--llm-concurrency` (default
`4`, `type=int`), threaded into `PipelineState["max_concurrency"]` the same
way `--prefilter-threshold`/`--hallucination-threshold` are today, and read
by `extract_profiles_node`/`judge_node` via `state.get("max_concurrency", 4)`.
One shared knob for both stages, not a separate one per stage — the simplest
thing that works; can be split later if the 7b (`cv_extractor`) and 14b
(`judge`) models ever need different tuning. Default of `4` matches Ollama's
own default `OLLAMA_NUM_PARALLEL` in recent versions, so the client doesn't
queue more concurrent requests than the server will actually run in
parallel by default. README documents that this should be tuned alongside
`OLLAMA_NUM_PARALLEL` and available VRAM — raising it past what the GPU/Ollama
can actually run concurrently adds contention overhead without speeding
anything up.

**Error handling:** no new semantics. `.batch()`'s default
(`return_exceptions=False`) fails the whole call if any single input raises —
identical to today's behavior, where one candidate raising inside the loop
stops the stage immediately. This is a pure performance change.

**Components:**

1. `src/evidencerank/agents/judge.py`:
   - Extract `_build_judge_prompt(jd: JDRequirements, profile: CandidateProfile) -> str`
     from the body of `judge_candidate` (identity redaction, work-history/project
     redaction, prompt formatting — logic unchanged, just relocated).
   - `judge_candidate` becomes a thin wrapper: build the prompt via the helper,
     `.invoke()` it, construct the `JudgeResult`. Behavior and tests unchanged.
   - Add `judge_candidates(jd: JDRequirements, profiles: list[CandidateProfile], max_concurrency: int) -> dict[str, JudgeResult]`:
     builds one prompt per profile via `_build_judge_prompt`, calls
     `model.batch(prompts, config={"max_concurrency": max_concurrency})` once,
     and zips the results back to `profile.candidate_id` to build the result dict.

2. `src/evidencerank/agents/cv_extractor.py`:
   - Extract `_build_cv_extractor_prompt(cv_text: str) -> str` from `extract_cv`.
   - `extract_cv` becomes a thin wrapper over the helper + `.invoke()`. Unchanged
     behavior and tests.
   - Add `cached_extract_cvs(candidates: dict[str, str], max_concurrency: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, CandidateProfile]`:
     for each `(candidate_id, cv_text)`, compute the cache key and check the
     cache (cheap, CPU-only) — collect hits directly into the result dict,
     collect misses into a list. If there are any misses, build one prompt per
     miss via the helper, issue one `model.batch(prompts, config={"max_concurrency": max_concurrency})`
     call, then save each new result to cache and add it to the result dict
     (same per-entry cache-write behavior as `cached_extract_cv` today). If
     there are no misses, skips `.batch()` entirely (no empty-batch call).

3. `src/evidencerank/agents/prefilter.py`:
   - Add `prefilter_candidates(jd_required_skills: list[str], candidate_skills: dict[str, list[str]], threshold: float) -> dict[str, PrefilterResult]`:
     builds `jd_text` once and one candidate text per candidate, calls
     `embedder.encode([jd_text, *candidate_texts])` exactly once, then computes
     `cosine_similarity` between the JD vector and each candidate vector.
     `prefilter_candidate` (singular) is untouched.

4. `src/evidencerank/graph.py`:
   - `extract_profiles_node` calls `cached_extract_cvs(state["raw_resumes"], max_concurrency)` instead of the per-candidate dict comprehension.
   - `prefilter_node` calls `prefilter_candidates(state["jd"].required_skills, {cid: profile.skills for cid, profile in state["profiles"].items()}, threshold)` instead of looping, then builds `dropped` from the returned results the same way as today.
   - `judge_node` calls `judge_candidates(state["jd"], [profile for cid, profile in state["profiles"].items() if state["prefilter_results"][cid].passed], max_concurrency)` instead of looping.
   - All three read `max_concurrency = state.get("max_concurrency", 4)`.

5. `src/evidencerank/cli.py`:
   - Add `@click.option("--llm-concurrency", default=4, type=int)`, pass through as `"max_concurrency": llm_concurrency` in the `graph.invoke(...)` state dict.

6. `README.md`:
   - Document `--llm-concurrency` alongside the existing threshold options,
     noting the default matches Ollama's default `OLLAMA_NUM_PARALLEL` and
     that raising it further only helps if both Ollama and available VRAM can
     actually sustain that many concurrent requests.

## Out of scope

- Caching `judge`/`calibrate` results across runs — separate concern (repeat-run
  cost), not requested for this change.
- Concurrency for `calibrate_node` — architecturally a single whole-pool call,
  nothing to parallelize.
- Concurrency for `hallucination_check_node` — pure CPU fuzzy-matching, already
  fast at this scale; not worth the added complexity.
- Per-stage concurrency limits (separate knobs for `cv_extractor` vs `judge`) —
  not requested; one shared `--llm-concurrency` value is simplest.
- Per-item error isolation in batched calls (e.g. skip a failing candidate and
  continue) — not requested, and would change existing fail-fast semantics.

## Testing

- `tests/agents/test_judge.py`: existing tests unchanged (still exercise
  `judge_candidate` via `.invoke()`). Add new tests for `judge_candidates`
  mocking `fake_structured_model.batch` to return a list of verdicts in
  order, asserting: the returned dict is keyed by candidate_id, the prompts
  list passed to `.batch()` has one entry per profile, and `config={"max_concurrency": N}`
  is passed through as given.
- `tests/agents/test_cv_extractor.py`: existing tests unchanged. Add tests
  for `cached_extract_cvs` covering: all-hits (no `.batch()` call made),
  all-misses (one `.batch()` call covering every candidate, each result
  cached), and a mixed case (only misses go through `.batch()`, hits bypass
  it entirely).
- `tests/agents/test_prefilter.py`: existing tests unchanged. Add a test for
  `prefilter_candidates` asserting `embedder.encode()` is called exactly once
  regardless of candidate count (regression test for the redundant
  JD-re-embedding fix), plus a correctness test that per-candidate `passed`/
  `similarity` values match what `prefilter_candidate` would produce for the
  same inputs.
- `tests/test_graph.py`: update the existing monkeypatch-based node tests to
  patch `judge_candidates`/`cached_extract_cvs`/`prefilter_candidates`
  instead of the singular functions, since the nodes now call the plural
  functions.
- `tests/test_cli.py`: add/extend a test asserting `--llm-concurrency` is
  parsed and passed through into the state dict given to `graph.invoke(...)`.
- Existing test suite must continue to pass (`uv run pytest`).
