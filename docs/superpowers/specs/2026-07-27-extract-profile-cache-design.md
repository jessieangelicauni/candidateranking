# Extract-Profile Cache

**Date:** 2026-07-27
**Status:** Approved for planning

## Purpose

Every pipeline run recomputes every stage from scratch — there is no caching
anywhere in `src/evidencerank/`. During iterative development (tuning the
Judge or Calibrator prompts/thresholds against the same resume set), this
means `extract_profiles` — an LLM call per candidate that produces output
completely independent of the JD, the Judge, or the Calibrator — gets
recomputed on every single re-run, even though its result would be identical.
Observed timing on a 6-candidate run: `extract_profiles` took ~570s, second
only to `judge` (~620s). This adds a persistent, content-addressed cache for
that one stage so repeated dev-loop runs against the same resumes skip
re-extracting a profile whose inputs haven't changed.

Scope is deliberately narrow: only `extract_profiles` is cached. `judge` and
`calibrate` are exactly what gets tuned during this kind of iteration, so
caching them would risk serving stale results across the very prompt changes
the dev loop exists to test. `prefilter` (embedding-based, ~1s/candidate) and
`hallucination_check` (deterministic, ~0.002s) are already cheap enough that
caching them isn't worth the complexity.

## Design

**Cache key:** SHA-256 hash of three parts, joined: the candidate's
`cv_text`, the `CV_EXTRACTOR_PROMPT` template string, and the resolved model
name for the `cv_extractor` stage (`resolve_model_name("cv_extractor")`).
Keying on content rather than candidate_id or file path means:
- The same resume text always hits cache regardless of what the PDF is
  named or which JD/run it's used in.
- Editing `CV_EXTRACTOR_PROMPT` or switching the `cv_extractor` model (via
  `EVIDENCERANK_MODEL_CV_EXTRACTOR`) automatically invalidates every cache
  entry — no manual cache-clearing step is ever needed.

**Storage:** one JSON file per cache key, at
`.cache/evidencerank/extract_profiles/<hash>.json`, containing the
`ExtractedProfileFields` the LLM produced (contact, skills, work_history,
education, projects) — not the full `CandidateProfile`, since `candidate_id`
and `cv_text` are supplied by the caller at lookup time, not part of what the
LLM computed. `.cache/` is added to `.gitignore`.

**Components:**

1. `src/evidencerank/cache.py` (new) — three small, generic functions with no
   framework or abstraction beyond what's needed:
   - `compute_cache_key(*parts: str) -> str` — SHA-256 hex digest of the
     parts joined with a separator.
   - `load_cached_json(cache_dir: Path, key: str) -> dict | None` — returns
     the parsed JSON at `cache_dir/<key>.json`, or `None` if it doesn't exist.
   - `save_cached_json(cache_dir: Path, key: str, data: dict) -> None` —
     creates `cache_dir` if needed and writes `data` as JSON to
     `cache_dir/<key>.json`.

2. `src/evidencerank/agents/cv_extractor.py` — add `cached_extract_cv`
   alongside the existing `extract_cv` (left unchanged, still directly
   unit-testable with no cache side effects):

   ```python
   DEFAULT_CACHE_DIR = Path(".cache/evidencerank/extract_profiles")

   def cached_extract_cv(
       candidate_id: str,
       cv_text: str,
       cache_dir: Path = DEFAULT_CACHE_DIR,
   ) -> CandidateProfile:
       key = compute_cache_key(cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"))
       cached = load_cached_json(cache_dir, key)
       if cached is not None:
           return CandidateProfile(
               candidate_id=candidate_id,
               raw_cv_text=cv_text,
               **cached,
           )
       profile = extract_cv(candidate_id, cv_text)
       save_cached_json(
           cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
       )
       return profile
   ```

   `model_dump(exclude={"candidate_id", "raw_cv_text"})` leaves exactly the
   `ExtractedProfileFields` shape (`contact`, `skills`, `work_history`,
   `education`, `projects`) since `CandidateProfile` adds only those two
   fields on top of `ExtractedProfileFields`.

3. `src/evidencerank/graph.py` — `extract_profiles_node` calls
   `cached_extract_cv` instead of `extract_cv`. No signature or state-shape
   change; this is a drop-in swap of which function gets called.

4. `.gitignore` — add `.cache/`.

5. `README.md` — one paragraph noting that extracted profiles are cached at
   `.cache/evidencerank/extract_profiles/`, keyed by resume content + the
   extractor prompt/model, and that deleting the directory (or changing
   either input) forces re-extraction.

## Amendment (post-implementation review)

The final whole-branch review empirically verified the cache end-to-end and
found three gaps not visible from any single task's diff. Fixed as follows:

1. **Schema drift wasn't covered by the key.** The three-part key (above)
   covers `cv_text`, the prompt, and the model — but not the shape of
   `ExtractedProfileFields` itself, which is just as much part of the LLM's
   contract (it's passed to `with_structured_output`). A schema change
   (e.g. a new field) let stale cache entries either silently produce
   profiles missing the new data (new optional field) or crash on every hit
   (new required field), with no documented recovery. **Fixed:** the cache
   key gains a fourth part — a hash of `ExtractedProfileFields.model_json_schema()`
   — so any schema change self-invalidates the same way a prompt or model
   change does. This supersedes the "three parts" wording above; the key is
   now four parts: `cv_text`, `CV_EXTRACTOR_PROMPT`, resolved model name,
   and the schema hash.

2. **A corrupt/truncated cache file permanently broke the pipeline.**
   `load_cached_json` called `json.loads` with no handling for a
   partially-written file (e.g. from a process killed mid-write) — every
   subsequent run failed identically on the same file until a human found
   and deleted it. **Fixed:** `load_cached_json` treats `JSONDecodeError`/
   `UnicodeDecodeError` the same as a missing file (returns `None`), so a
   corrupt entry degrades to a cache miss and self-heals via the normal
   miss path (re-extract, overwrite).

3. **The cache is an undocumented second store of unredacted candidate
   PII.** The cached payload includes `contact` (name, email, phone,
   location) — exactly the field set the Judge is deliberately blinded to
   — persisted indefinitely in `.cache/`, unlike `report.json` which the
   README already flags as sensitive. **Fixed:** the README's existing
   privacy note (about `report.json` holding unredacted identity data) is
   extended to name `.cache/evidencerank/extract_profiles/` as a second
   location holding unredacted contact data, not covered by the Judge's
   redaction. The README also now notes the cache path is relative to the
   directory `evidencerank` is invoked from.

## Out of scope

- Caching `judge`, `calibrate`, `prefilter`, or `hallucination_check` — not
  requested; `judge`/`calibrate` are actively tuned during the workflow this
  cache serves, so caching them would undermine the dev loop rather than
  help it.
- A CLI flag to bypass or clear the cache — not requested; deleting
  `.cache/evidencerank/extract_profiles/` (or editing the prompt/model,
  which self-invalidates) covers the only scenarios that need it.
- A configurable cache directory (flag or env var) — not requested; the
  location is fixed.

## Testing

- `tests/agents/test_cv_extractor.py`: add a cache-hit test (pre-seed a
  cache file at the expected key for given `cv_text`/prompt/model, call
  `cached_extract_cv`, assert the mocked LLM is never invoked and the
  returned profile's fields match the seeded cache) and a cache-miss test
  (empty `tmp_path` cache dir, call `cached_extract_cv`, assert the mocked
  LLM is invoked once and a correctly-shaped cache file is written). Both
  tests pass an explicit `cache_dir=tmp_path` — no monkeypatching of a
  module-level constant needed.
- A new `tests/test_cache.py` for `compute_cache_key` (same inputs → same
  key; different inputs → different key) and `load_cached_json` /
  `save_cached_json` round-tripping through `tmp_path`.
- `tests/test_graph.py`: no behavior change expected (the swap from
  `extract_cv` to `cached_extract_cv` is transparent to the graph's existing
  monkeypatch-based tests, which patch `evidencerank.graph.extract_cv` today
  and will need to patch `evidencerank.graph.cached_extract_cv` instead).
- Existing test suite must continue to pass (`uv run pytest`).
