# Concurrent Pipeline Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce single-run wall-clock time on a single-GPU Ollama setup by batching the `extract_profiles` and `judge` stages' per-candidate LLM calls through LangChain's built-in bounded concurrency, and by fixing `prefilter`'s redundant re-embedding of the JD text on every candidate.

**Architecture:** Each of the three affected agent modules (`judge.py`, `cv_extractor.py`, `prefilter.py`) gets a new plural function alongside its existing singular one: `judge_candidates`, `cached_extract_cvs`, `prefilter_candidates`. Each plural function builds all its prompts/inputs up front and issues exactly one call — `model.batch(prompts, config={"max_concurrency": N})` for the LLM stages, one `embedder.encode([...])` call for the embedding stage — instead of looping with per-candidate `.invoke()`/`.encode()` calls. `graph.py`'s three corresponding nodes are rewritten to call the new plural functions. A new `--llm-concurrency` CLI option (default `4`) flows into `PipelineState["max_concurrency"]`, read by the two LLM-batching nodes.

**Tech Stack:** Python, LangChain (`Runnable.batch()` — already a transitive capability of `ChatOllama.with_structured_output()`, no new dependency), `sentence-transformers`, `click`, `pytest`, `unittest.mock.MagicMock` for test doubles.

## Global Constraints

- No new dependencies — `.batch()` is already available on every LangChain `Runnable` in use.
- Existing singular functions (`judge_candidate`, `extract_cv`, `cached_extract_cv`, `prefilter_candidate`) keep their exact current signatures and behavior; existing tests for them must pass unmodified.
- `.batch()` calls use the default `return_exceptions=False` — one failing candidate must fail the whole stage, identical to today's loop behavior. Do not add per-item error isolation.
- `calibrate_node` and `hallucination_check_node` are out of scope — do not modify `calibrator.py` or `hallucination_checker.py`.
- Do not change any existing default threshold value (`--prefilter-threshold`, `--hallucination-threshold`, or the `prefilter_threshold` fallback literal currently in `graph.py`) — only the function each node calls changes, not the values passed through.
- New `--llm-concurrency` CLI option defaults to `4`.

---

## Task 1: `judge_candidates` — batched Judge calls

**Files:**
- Modify: `src/evidencerank/agents/judge.py:41-74`
- Test: `tests/agents/test_judge.py`

**Interfaces:**
- Consumes: `evidencerank.llm.get_chat_model`, `evidencerank.models.{CandidateProfile, JDRequirements, JudgeResult, JudgeVerdict}`, `evidencerank.privacy.{detect_probable_name, redact_identity}` — all pre-existing and unchanged.
- Produces: `judge_candidates(jd: JDRequirements, profiles: list[CandidateProfile], max_concurrency: int) -> dict[str, JudgeResult]` — consumed by Task 4's `judge_node`.

- [ ] **Step 1: Write the failing test**

Add to `tests/agents/test_judge.py` (it already imports `MagicMock`, `judge_candidate`, and the model classes used below — add `judge_candidates` to the existing `from evidencerank.agents.judge import judge_candidate` line, and reuse the module's existing `_make_profile()` helper):

```python
def test_judge_candidates_batches_prompts_and_returns_results_by_candidate_id(monkeypatch):
    verdict_a = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    verdict_b = JudgeVerdict(
        tier=Tier.WEAK_FIT,
        rating=3,
        evidence=[EvidenceClaim(claim="Has DevOps experience", quote="9 years of DevOps experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.batch.return_value = [verdict_a, verdict_b]
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    profile_a = _make_profile()
    profile_b = CandidateProfile(
        candidate_id="c2",
        raw_cv_text="Allison Doyle\nDevOps Engineer\n9 years of DevOps experience",
        contact=ContactInfo(name="Allison Doyle"),
        skills=["DevOps"],
    )

    results = judge_candidates(jd, [profile_a, profile_b], max_concurrency=4)

    assert set(results.keys()) == {"c1", "c2"}
    assert results["c1"].candidate_id == "c1"
    assert results["c1"].tier == Tier.STRONG_FIT
    assert results["c1"].rating == 8
    assert results["c2"].candidate_id == "c2"
    assert results["c2"].tier == Tier.WEAK_FIT
    assert results["c2"].rating == 3
    call_args, call_kwargs = fake_structured_model.batch.call_args
    prompts_sent = call_args[0]
    assert len(prompts_sent) == 2
    assert "Allison Doyle" not in prompts_sent[1]
    assert call_kwargs["config"] == {"max_concurrency": 4}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/agents/test_judge.py::test_judge_candidates_batches_prompts_and_returns_results_by_candidate_id -v`
Expected: FAIL with `ImportError: cannot import name 'judge_candidates'`

- [ ] **Step 3: Write the implementation**

Replace lines 41-74 of `src/evidencerank/agents/judge.py` (the current `judge_candidate` function) with:

```python
def _build_judge_prompt(jd: JDRequirements, profile: CandidateProfile) -> str:
    contact = profile.contact
    if not contact.name:
        probable_name = detect_probable_name(profile.raw_cv_text)
        if probable_name:
            contact = contact.model_copy(update={"name": probable_name})

    redacted_text = redact_identity(profile.raw_cv_text, contact)

    redacted_work_history = []
    for entry in profile.work_history:
        entry_dump = entry.model_dump()
        entry_dump["achievements"] = [
            redact_identity(achievement, contact) for achievement in entry.achievements
        ]
        redacted_work_history.append(entry_dump)

    redacted_projects = []
    for entry in profile.projects:
        entry_dump = entry.model_dump()
        entry_dump["description"] = redact_identity(entry.description, contact)
        redacted_projects.append(entry_dump)

    return JUDGE_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        redacted_cv_text=redacted_text,
        skills=profile.skills,
        work_history=redacted_work_history,
        education=[entry.model_dump() for entry in profile.education],
        projects=redacted_projects,
    )


def judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult:
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompt = _build_judge_prompt(jd, profile)
    verdict = model.invoke(prompt)
    return JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())


def judge_candidates(
    jd: JDRequirements, profiles: list[CandidateProfile], max_concurrency: int
) -> dict[str, JudgeResult]:
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompts = [_build_judge_prompt(jd, profile) for profile in profiles]
    verdicts = model.batch(prompts, config={"max_concurrency": max_concurrency})
    return {
        profile.candidate_id: JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())
        for profile, verdict in zip(profiles, verdicts)
    }
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: all tests pass (existing `judge_candidate` tests plus the new `judge_candidates` test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/judge.py tests/agents/test_judge.py
git commit -m "feat: add judge_candidates for concurrent batch Judge calls"
```

---

## Task 2: `cached_extract_cvs` — batched + cached CV extraction

**Files:**
- Modify: `src/evidencerank/agents/cv_extractor.py:19-48`
- Test: `tests/agents/test_cv_extractor.py`

**Interfaces:**
- Consumes: `evidencerank.cache.{compute_cache_key, load_cached_json, save_cached_json}`, `evidencerank.llm.{get_chat_model, resolve_model_name}`, `evidencerank.models.{CandidateProfile, ExtractedProfileFields}` — all pre-existing and unchanged.
- Produces: `cached_extract_cvs(candidates: dict[str, str], max_concurrency: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, CandidateProfile]` — consumed by Task 4's `extract_profiles_node`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_cv_extractor.py` (add `cached_extract_cvs` to the existing import line from `evidencerank.agents.cv_extractor`):

```python
def test_cached_extract_cvs_all_hits_skips_llm_entirely(tmp_path, monkeypatch):
    candidates = {
        "c1": "Jane Doe resume text...",
        "c2": "John Smith resume text...",
    }
    cached_fields = {
        "c1": {
            "contact": {"name": "Jane Doe", "email": "", "phone": "", "location": ""},
            "skills": ["Python"], "work_history": [], "education": [], "projects": [],
        },
        "c2": {
            "contact": {"name": "John Smith", "email": "", "phone": "", "location": ""},
            "skills": ["Go"], "work_history": [], "education": [], "projects": [],
        },
    }
    for candidate_id, cv_text in candidates.items():
        key = compute_cache_key(
            cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"),
            json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
        )
        save_cached_json(tmp_path, key, cached_fields[candidate_id])

    fake_chat_model = MagicMock()
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model", lambda stage: fake_chat_model
    )

    profiles = cached_extract_cvs(candidates, max_concurrency=4, cache_dir=tmp_path)

    assert set(profiles.keys()) == {"c1", "c2"}
    assert profiles["c1"].skills == ["Python"]
    assert profiles["c2"].skills == ["Go"]
    fake_chat_model.with_structured_output.assert_not_called()


def test_cached_extract_cvs_all_misses_batches_and_caches_each(tmp_path, monkeypatch):
    candidates = {
        "c1": "Jane Doe resume text...",
        "c2": "John Smith resume text...",
    }
    extracted_c1 = ExtractedProfileFields(contact=ContactInfo(name="Jane Doe"), skills=["Python"])
    extracted_c2 = ExtractedProfileFields(contact=ContactInfo(name="John Smith"), skills=["Go"])
    fake_structured_model = MagicMock()
    fake_structured_model.batch.return_value = [extracted_c1, extracted_c2]
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model", lambda stage: fake_chat_model
    )

    profiles = cached_extract_cvs(candidates, max_concurrency=4, cache_dir=tmp_path)

    assert profiles["c1"].skills == ["Python"]
    assert profiles["c2"].skills == ["Go"]
    call_args, call_kwargs = fake_structured_model.batch.call_args
    assert len(call_args[0]) == 2
    assert call_kwargs["config"] == {"max_concurrency": 4}

    for candidate_id, cv_text in candidates.items():
        key = compute_cache_key(
            cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"),
            json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
        )
        assert (tmp_path / f"{key}.json").exists()


def test_cached_extract_cvs_mixed_hits_and_misses_only_batches_misses(tmp_path, monkeypatch):
    candidates = {
        "c1": "Jane Doe resume text...",
        "c2": "John Smith resume text...",
    }
    cached_fields_c1 = {
        "contact": {"name": "Jane Doe", "email": "", "phone": "", "location": ""},
        "skills": ["Python"], "work_history": [], "education": [], "projects": [],
    }
    key_c1 = compute_cache_key(
        candidates["c1"], CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )
    save_cached_json(tmp_path, key_c1, cached_fields_c1)

    extracted_c2 = ExtractedProfileFields(contact=ContactInfo(name="John Smith"), skills=["Go"])
    fake_structured_model = MagicMock()
    fake_structured_model.batch.return_value = [extracted_c2]
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model", lambda stage: fake_chat_model
    )

    profiles = cached_extract_cvs(candidates, max_concurrency=4, cache_dir=tmp_path)

    assert profiles["c1"].skills == ["Python"]
    assert profiles["c2"].skills == ["Go"]
    call_args, _ = fake_structured_model.batch.call_args
    prompts_sent = call_args[0]
    assert len(prompts_sent) == 1
    assert candidates["c2"] in prompts_sent[0]
```

This test file's imports need `ContactInfo` (already imported) and `MagicMock` (already imported) — no new imports required beyond adding `cached_extract_cvs` to the existing `from evidencerank.agents.cv_extractor import ...` line.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_cv_extractor.py -v -k cached_extract_cvs`
Expected: FAIL with `ImportError: cannot import name 'cached_extract_cvs'`

- [ ] **Step 3: Write the implementation**

Replace lines 19-48 of `src/evidencerank/agents/cv_extractor.py` (the current `extract_cv` and `cached_extract_cv` functions) with:

```python
def _build_cv_extractor_prompt(cv_text: str) -> str:
    return CV_EXTRACTOR_PROMPT.format(cv_text=cv_text)


def _cache_key_for(cv_text: str) -> str:
    return compute_cache_key(
        cv_text,
        CV_EXTRACTOR_PROMPT,
        resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )


def extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile:
    model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
    fields = model.invoke(_build_cv_extractor_prompt(cv_text))
    return CandidateProfile(
        candidate_id=candidate_id,
        raw_cv_text=cv_text,
        **fields.model_dump(),
    )


def cached_extract_cv(
    candidate_id: str,
    cv_text: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> CandidateProfile:
    key = _cache_key_for(cv_text)
    cached = load_cached_json(cache_dir, key)
    if cached is not None:
        return CandidateProfile(candidate_id=candidate_id, raw_cv_text=cv_text, **cached)

    profile = extract_cv(candidate_id, cv_text)
    save_cached_json(
        cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
    )
    return profile


def cached_extract_cvs(
    candidates: dict[str, str],
    max_concurrency: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, CandidateProfile]:
    results: dict[str, CandidateProfile] = {}
    misses: list[tuple[str, str, str]] = []
    for candidate_id, cv_text in candidates.items():
        key = _cache_key_for(cv_text)
        cached = load_cached_json(cache_dir, key)
        if cached is not None:
            results[candidate_id] = CandidateProfile(
                candidate_id=candidate_id, raw_cv_text=cv_text, **cached
            )
        else:
            misses.append((candidate_id, cv_text, key))

    if misses:
        model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
        prompts = [_build_cv_extractor_prompt(cv_text) for _, cv_text, _ in misses]
        fields_list = model.batch(prompts, config={"max_concurrency": max_concurrency})
        for (candidate_id, cv_text, key), fields in zip(misses, fields_list):
            profile = CandidateProfile(
                candidate_id=candidate_id, raw_cv_text=cv_text, **fields.model_dump()
            )
            save_cached_json(
                cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
            )
            results[candidate_id] = profile

    return results
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run pytest tests/agents/test_cv_extractor.py -v`
Expected: all tests pass (existing `extract_cv`/`cached_extract_cv` tests plus the three new `cached_extract_cvs` tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/cv_extractor.py tests/agents/test_cv_extractor.py
git commit -m "feat: add cached_extract_cvs for concurrent batch CV extraction"
```

---

## Task 3: `prefilter_candidates` — single-encode batched pre-filter

**Files:**
- Modify: `src/evidencerank/agents/prefilter.py`
- Test: `tests/agents/test_prefilter.py`

**Interfaces:**
- Consumes: `evidencerank.models.PrefilterResult`, `evidencerank.agents.prefilter.{_get_embedder, cosine_similarity}` — pre-existing and unchanged.
- Produces: `prefilter_candidates(jd_required_skills: list[str], candidate_skills: dict[str, list[str]], threshold: float = 0.6) -> dict[str, PrefilterResult]` — consumed by Task 4's `prefilter_node`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_prefilter.py`:

```python
def test_prefilter_candidates_calls_encode_exactly_once(monkeypatch):
    encode_calls = []

    class FakeEmbedder:
        def encode(self, texts):
            texts = list(texts)
            encode_calls.append(texts)
            return np.array([[1.0, 0.0, 0.0]] * len(texts))

    monkeypatch.setattr("evidencerank.agents.prefilter._get_embedder", lambda: FakeEmbedder())

    results = prefilter_candidates(
        jd_required_skills=["Python", "PyTorch"],
        candidate_skills={"c1": ["Python"], "c2": ["PyTorch"], "c3": ["Baking"]},
        threshold=0.5,
    )

    assert len(encode_calls) == 1
    assert len(encode_calls[0]) == 4  # 1 JD text + 3 candidate texts
    assert set(results.keys()) == {"c1", "c2", "c3"}
    assert all(result.passed for result in results.values())


def test_prefilter_candidates_matches_single_candidate_results():
    jd_required_skills = ["Python", "Machine Learning", "PyTorch"]
    candidate_skills = {
        "matching": ["Python", "PyTorch", "Deep Learning", "Model training"],
        "unrelated": ["Baking", "Pastry decoration", "Kitchen sanitation", "Menu planning"],
    }

    batched = prefilter_candidates(jd_required_skills, candidate_skills, threshold=0.5)

    for candidate_id, skills in candidate_skills.items():
        single = prefilter_candidate(
            candidate_id=candidate_id,
            jd_required_skills=jd_required_skills,
            candidate_skills=skills,
            threshold=0.5,
        )
        assert batched[candidate_id].passed == single.passed
        assert abs(batched[candidate_id].similarity - single.similarity) < 1e-6
```

Update the file's import line to:

```python
from evidencerank.agents.prefilter import (
    cosine_similarity,
    prefilter_candidate,
    prefilter_candidates,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_prefilter.py -v -k prefilter_candidates`
Expected: FAIL with `ImportError: cannot import name 'prefilter_candidates'`

- [ ] **Step 3: Write the implementation**

Append to `src/evidencerank/agents/prefilter.py`:

```python
def prefilter_candidates(
    jd_required_skills: list[str],
    candidate_skills: dict[str, list[str]],
    threshold: float = 0.6,
) -> dict[str, PrefilterResult]:
    embedder = _get_embedder()
    jd_text = ", ".join(jd_required_skills)
    candidate_ids = list(candidate_skills.keys())
    candidate_texts = [", ".join(candidate_skills[candidate_id]) for candidate_id in candidate_ids]
    vectors = embedder.encode([jd_text, *candidate_texts])
    jd_vec, candidate_vecs = vectors[0], vectors[1:]

    results: dict[str, PrefilterResult] = {}
    for candidate_id, candidate_vec in zip(candidate_ids, candidate_vecs):
        similarity = cosine_similarity(jd_vec, candidate_vec)
        results[candidate_id] = PrefilterResult(
            candidate_id=candidate_id,
            similarity=similarity,
            passed=similarity >= threshold,
        )
    return results
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run pytest tests/agents/test_prefilter.py -v`
Expected: all tests pass (existing `prefilter_candidate`/`cosine_similarity` tests plus the two new `prefilter_candidates` tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/prefilter.py tests/agents/test_prefilter.py
git commit -m "feat: add prefilter_candidates to embed the JD once per run instead of per candidate"
```

---

## Task 4: Wire the batched functions into the pipeline graph

**Files:**
- Modify: `src/evidencerank/graph.py:1-24` (imports), `:27-40` (`PipelineState`), `:43-80` (the three node functions)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `judge_candidates` (Task 1), `cached_extract_cvs` (Task 2), `prefilter_candidates` (Task 3).
- Produces: `PipelineState["max_concurrency"]: int` (read via `state.get("max_concurrency", 4)`) — consumed by Task 5's `cli.py`.

- [ ] **Step 1: Update the failing tests**

`tests/test_graph.py`'s `_patch_pipeline_fakes` helper and both test functions currently patch and exercise the singular per-candidate functions (`cached_extract_cv`, `prefilter_candidate`, `judge_candidate`). Replace the whole file's content with:

```python
from evidencerank.graph import build_graph
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


def _patch_pipeline_fakes(
    monkeypatch,
    *,
    extract_cvs,
    prefilter_candidates,
    judge_candidates,
    calibrate_pool,
    check_evidence,
):
    monkeypatch.setattr("evidencerank.graph.cached_extract_cvs", extract_cvs)
    monkeypatch.setattr("evidencerank.graph.prefilter_candidates", prefilter_candidates)
    monkeypatch.setattr("evidencerank.graph.judge_candidates", judge_candidates)
    monkeypatch.setattr("evidencerank.graph.calibrate_pool", calibrate_pool)
    monkeypatch.setattr("evidencerank.graph.check_evidence", check_evidence)


def test_graph_runs_extract_prefilter_judge_hallucination_calibrate(monkeypatch):
    jd = JDRequirements(title="ML Engineer", required_skills=["Python", "PyTorch"])

    # NOTE: "weak" is intentionally NOT last here. If hallucination_check_node
    # were to pair judge_results with profiles positionally (e.g. via zip)
    # instead of doing a proper dict lookup by candidate_id, the dropped
    # "weak" profile sitting in the middle would shift the positional
    # alignment and cause raw_cv_text to be mismatched between strong_a and
    # strong_b. Putting "weak" last would let such a bug pass by coincidence.
    raw_resumes = {
        "strong_a": "Python resume text - candidate A unique marker AAA",
        "weak": "Photoshop resume text",
        "strong_b": "Python resume text - candidate B unique marker BBB",
    }

    def fake_extract_cvs(candidates, max_concurrency):
        return {
            candidate_id: CandidateProfile(
                candidate_id=candidate_id,
                raw_cv_text=raw_text,
                contact=ContactInfo(name=candidate_id),
                skills=["Python"] if candidate_id != "weak" else ["Photoshop"],
            )
            for candidate_id, raw_text in candidates.items()
        }

    def fake_prefilter_candidates(jd_required_skills, candidate_skills, threshold):
        return {
            candidate_id: PrefilterResult(
                candidate_id=candidate_id,
                similarity=0.9 if candidate_id != "weak" else 0.1,
                passed=candidate_id != "weak",
            )
            for candidate_id in candidate_skills
        }

    def fake_judge_candidates(jd_requirements, profiles, max_concurrency):
        return {
            profile.candidate_id: JudgeResult(
                candidate_id=profile.candidate_id,
                tier=Tier.STRONG_FIT,
                rating=9,
                evidence=[
                    EvidenceClaim(claim="Strong fit", quote="Python"),
                    EvidenceClaim(claim="Fabricated claim", quote="FABRICATED unverifiable quote text"),
                ],
            )
            for profile in profiles
        }

    # Records every call to calibrate_pool so we can assert it is invoked
    # exactly once over the full surviving pool, not once per candidate.
    calibrate_calls: list[tuple[JDRequirements, list[JudgeResult]]] = []

    def fake_calibrate_pool(jd_requirements, judge_results):
        calibrate_calls.append((jd_requirements, list(judge_results)))
        return [
            CalibratedResult(
                candidate_id=r.candidate_id, final_rank=i + 1, tier=r.tier,
                rating=r.rating, calibration_notes="Ranked within pool",
            )
            for i, r in enumerate(judge_results)
        ]

    # Records the (candidate_id, raw_cv_text) pairs check_evidence was called
    # with, so we can assert each candidate was checked against its OWN raw
    # CV text rather than a mismatched/swapped one. Also flags any evidence
    # quote starting with "FABRICATED" as unverified, simulating a real
    # hallucination-checker finding.
    hallucination_calls: list[tuple[str, str]] = []

    def fake_check_evidence(judge_result, raw_cv_text, threshold):
        hallucination_calls.append((judge_result.candidate_id, raw_cv_text))
        unverified = [
            claim.quote for claim in judge_result.evidence if claim.quote.startswith("FABRICATED")
        ]
        return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)

    _patch_pipeline_fakes(
        monkeypatch,
        extract_cvs=fake_extract_cvs,
        prefilter_candidates=fake_prefilter_candidates,
        judge_candidates=fake_judge_candidates,
        calibrate_pool=fake_calibrate_pool,
        check_evidence=fake_check_evidence,
    )

    graph = build_graph()
    final_state = graph.invoke(
        {
            "jd": jd,
            "raw_resumes": raw_resumes,
        }
    )

    assert set(final_state["profiles"].keys()) == {"strong_a", "strong_b", "weak"}
    assert final_state["dropped"] == [
        {"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}
    ]
    assert set(final_state["judge_results"].keys()) == {"strong_a", "strong_b"}
    assert len(final_state["calibrated_results"]) == 2
    assert {r.candidate_id for r in final_state["calibrated_results"]} == {"strong_a", "strong_b"}

    # Regression guard 1: calibrate_pool must be invoked exactly once with
    # the full surviving pool, not once per candidate.
    assert len(calibrate_calls) == 1
    _, pooled_judge_results = calibrate_calls[0]
    assert {r.candidate_id for r in pooled_judge_results} == {"strong_a", "strong_b"}

    # Regression guard 2: each candidate's hallucination check must be run
    # against its OWN raw CV text, not a swapped/mismatched one.
    recorded_raw_text_by_candidate = dict(hallucination_calls)
    assert len(hallucination_calls) == 2
    for candidate_id in ("strong_a", "strong_b"):
        assert recorded_raw_text_by_candidate[candidate_id] == raw_resumes[candidate_id]

    # Regression guard 3: hallucination_reports keeps the ORIGINAL unverified
    # quote for audit, even though it gets stripped from what calibrate/final
    # judge_results see.
    for candidate_id in ("strong_a", "strong_b"):
        report = final_state["hallucination_reports"][candidate_id]
        assert report.all_verified is False
        assert "FABRICATED unverifiable quote text" in report.unverified_quotes

    # Regression guard 4: the fabricated evidence item never reaches
    # calibrate_pool or the final judge_results — only the verified "Python"
    # quote survives, proving filtering happens BEFORE calibration.
    for r in pooled_judge_results:
        assert [c.quote for c in r.evidence] == ["Python"]
    for candidate_id in ("strong_a", "strong_b"):
        assert [c.quote for c in final_state["judge_results"][candidate_id].evidence] == ["Python"]

    # Regression guard 5: every stage records a non-negative timing, keyed by
    # node name, so latency is visible in the eventual report.json.
    assert set(final_state["stage_timings"].keys()) == {
        "extract_profiles", "prefilter", "judge", "hallucination_check", "shortlist", "calibrate",
    }
    for seconds in final_state["stage_timings"].values():
        assert isinstance(seconds, float)
        assert seconds >= 0.0


def test_graph_shortlists_top_10_by_rating_before_calibrating(monkeypatch):
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    raw_resumes = {f"c{i}": f"Python resume {i}" for i in range(12)}

    def fake_extract_cvs(candidates, max_concurrency):
        return {
            candidate_id: CandidateProfile(
                candidate_id=candidate_id,
                raw_cv_text=raw_text,
                contact=ContactInfo(name=candidate_id),
                skills=["Python"],
            )
            for candidate_id, raw_text in candidates.items()
        }

    def fake_prefilter_candidates(jd_required_skills, candidate_skills, threshold):
        return {
            candidate_id: PrefilterResult(candidate_id=candidate_id, similarity=0.9, passed=True)
            for candidate_id in candidate_skills
        }

    # 12 candidates, ratings capped to the valid 1-10 range (JudgeResult.rating
    # is Field(ge=1, le=10)): c0-c7 at 10, c8-c9 at 9, c10-c11 at 3. The
    # cutoff for the top 10 lands cleanly between the rating-9 and rating-3
    # groups, so the top 10 by rating is exactly c0..c9 with no boundary tie
    # to resolve here (tie-at-the-boundary behavior is already covered in
    # isolation by tests/agents/test_shortlist.py).
    def fake_judge_candidates(jd_requirements, profiles, max_concurrency):
        ratings = [10] * 8 + [9] * 2 + [3] * 2
        results = {}
        for profile in profiles:
            index = int(profile.candidate_id[1:])
            results[profile.candidate_id] = JudgeResult(
                candidate_id=profile.candidate_id,
                tier=Tier.STRONG_FIT,
                rating=ratings[index],
                evidence=[EvidenceClaim(claim="Strong fit", quote="Python")],
            )
        return results

    calibrate_calls: list[list[str]] = []

    def fake_calibrate_pool(jd_requirements, judge_results):
        calibrate_calls.append([r.candidate_id for r in judge_results])
        return [
            CalibratedResult(
                candidate_id=r.candidate_id, final_rank=i + 1, tier=r.tier,
                rating=r.rating, calibration_notes="Ranked within shortlist",
            )
            for i, r in enumerate(judge_results)
        ]

    def fake_check_evidence(judge_result, raw_cv_text, threshold):
        return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=[])

    _patch_pipeline_fakes(
        monkeypatch,
        extract_cvs=fake_extract_cvs,
        prefilter_candidates=fake_prefilter_candidates,
        judge_candidates=fake_judge_candidates,
        calibrate_pool=fake_calibrate_pool,
        check_evidence=fake_check_evidence,
    )

    graph = build_graph()
    final_state = graph.invoke({"jd": jd, "raw_resumes": raw_resumes})

    expected_shortlist = {f"c{i}" for i in range(10)}
    expected_cut = {f"c{i}" for i in range(10, 12)}

    assert len(calibrate_calls) == 1
    assert set(calibrate_calls[0]) == expected_shortlist
    assert set(final_state["shortlisted_results"].keys()) == expected_shortlist
    assert {entry["candidate_id"] for entry in final_state["not_shortlisted"]} == expected_cut
    assert all(
        entry["reason"] == "ranked outside judge's top 10 by rating"
        for entry in final_state["not_shortlisted"]
    )
    # judge_results in state still contains everyone judged, shortlisted or not.
    assert set(final_state["judge_results"].keys()) == expected_shortlist | expected_cut
    # hallucination check still runs against the full judged pool, not just the shortlist.
    assert set(final_state["hallucination_reports"].keys()) == expected_shortlist | expected_cut
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL — `AttributeError: <module 'evidencerank.graph'> does not have the attribute 'cached_extract_cvs'` (the monkeypatch targets don't exist in `graph.py` yet)

- [ ] **Step 3: Write the implementation**

In `src/evidencerank/graph.py`, replace the import block (lines 7-16) with:

```python
from evidencerank.agents.calibrator import calibrate_pool
from evidencerank.agents.cv_extractor import cached_extract_cvs
from evidencerank.agents.hallucination_checker import (
    DEFAULT_THRESHOLD,
    check_evidence,
    filter_verified_evidence,
)
from evidencerank.agents.judge import judge_candidates
from evidencerank.agents.prefilter import prefilter_candidates
from evidencerank.agents.shortlist import select_shortlist
```

Add `max_concurrency: int` to the `PipelineState` TypedDict (after the existing `hallucination_threshold: float` line):

```python
    hallucination_threshold: float
    max_concurrency: int
```

Replace the three node functions (`extract_profiles_node`, `prefilter_node`, `judge_node`) with:

```python
def extract_profiles_node(state: PipelineState) -> dict:
    click.echo("Running stage: extract_profiles")
    max_concurrency = state.get("max_concurrency", 4)
    profiles = cached_extract_cvs(state["raw_resumes"], max_concurrency=max_concurrency)
    return {"profiles": profiles}


def prefilter_node(state: PipelineState) -> dict:
    click.echo("Running stage: prefilter")
    threshold = state.get("prefilter_threshold", 0.7)
    candidate_skills = {
        candidate_id: profile.skills for candidate_id, profile in state["profiles"].items()
    }
    results = prefilter_candidates(state["jd"].required_skills, candidate_skills, threshold=threshold)
    dropped: list[dict[str, str]] = [
        {"candidate_id": candidate_id, "reason": "pre-filter: no relevant skill overlap"}
        for candidate_id, result in results.items()
        if not result.passed
    ]
    return {"prefilter_results": results, "dropped": dropped}


def judge_node(state: PipelineState) -> dict:
    click.echo("Running stage: judge")
    max_concurrency = state.get("max_concurrency", 4)
    passing_profiles = [
        state["profiles"][candidate_id]
        for candidate_id, result in state["prefilter_results"].items()
        if result.passed
    ]
    judge_results = judge_candidates(state["jd"], passing_profiles, max_concurrency=max_concurrency)
    return {"judge_results": judge_results}
```

Leave `shortlist_node`, `calibrate_node`, `hallucination_check_node`, `_timed_node`, and `build_graph` exactly as they are — none of them reference the functions being swapped.

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: both tests pass

- [ ] **Step 5: Run the full test suite to confirm no other regression**

Run: `uv run pytest`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/graph.py tests/test_graph.py
git commit -m "feat: wire concurrent batch functions into extract_profiles/prefilter/judge nodes"
```

---

## Task 5: `--llm-concurrency` CLI option

**Files:**
- Modify: `src/evidencerank/cli.py:14-54`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `PipelineState["max_concurrency"]` (Task 4).
- Produces: `--llm-concurrency` CLI flag (default `4`) — documented in Task 6's README update.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
def test_rank_command_passes_llm_concurrency_through_to_graph_state(tmp_path, monkeypatch):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    _make_pdf(resumes_dir / "candidate1.pdf", "Candidate One\nPython, PyTorch")

    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_final_state = {
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
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final_state
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(tmp_path / "out.json"),
            "--out-md", str(tmp_path / "out.md"),
            "--llm-concurrency", "8",
        ],
    )

    assert result.exit_code == 0, result.output
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 8


def test_rank_command_defaults_llm_concurrency_to_four(tmp_path, monkeypatch):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    _make_pdf(resumes_dir / "candidate1.pdf", "Candidate One\nPython, PyTorch")

    fake_jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    monkeypatch.setattr("evidencerank.cli.parse_jd", lambda jd_text: fake_jd)

    fake_final_state = {
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
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_final_state
    monkeypatch.setattr("evidencerank.cli.build_graph", lambda: fake_graph)

    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(tmp_path / "out.json"),
            "--out-md", str(tmp_path / "out.md"),
        ],
    )

    assert result.exit_code == 0, result.output
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert invoked_state["max_concurrency"] == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k llm_concurrency`
Expected: FAIL — `Error: No such option: --llm-concurrency` for the first test; `KeyError: 'max_concurrency'` for the second

- [ ] **Step 3: Write the implementation**

In `src/evidencerank/cli.py`, add the option after `--hallucination-threshold` (line 20):

```python
@click.option("--hallucination-threshold", default=85.0, type=float)
@click.option("--llm-concurrency", default=4, type=int)
```

Add `llm_concurrency` to the `rank` function signature (after `hallucination_threshold`):

```python
def rank(
    jd_path,
    resumes_dir,
    out_json,
    out_md,
    prefilter_threshold,
    hallucination_threshold,
    llm_concurrency,
    with_eval_report,
    out_eval_report,
):
```

Add `"max_concurrency": llm_concurrency` to the state dict passed to `graph.invoke(...)`:

```python
    final_state = graph.invoke(
        {
            "jd": jd_requirements,
            "raw_resumes": raw_resumes,
            "prefilter_threshold": prefilter_threshold,
            "hallucination_threshold": hallucination_threshold,
            "max_concurrency": llm_concurrency,
        }
    )
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all tests pass

- [ ] **Step 5: Run the full test suite to confirm no other regression**

Run: `uv run pytest`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/cli.py tests/test_cli.py
git commit -m "feat: add --llm-concurrency CLI option for batched extract/judge stages"
```

---

## Task 6: Document `--llm-concurrency` in the README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (documentation only).

- [ ] **Step 1: Add the documentation**

In `README.md`, immediately after the existing paragraph that documents `--prefilter-threshold`/`--hallucination-threshold` (the paragraph beginning "Two thresholds are tunable:"), add:

```markdown
`--llm-concurrency` (default `4`) bounds how many candidates' `extract_profiles`
and `judge` LLM calls run concurrently, using LangChain's `Runnable.batch()`
instead of one sequential call per candidate. The default matches Ollama's own
default concurrent-request limit (`OLLAMA_NUM_PARALLEL`) on recent versions —
raising `--llm-concurrency` past what Ollama and your GPU's VRAM can actually
run at once adds contention overhead without speeding anything up, so tune it
alongside `OLLAMA_NUM_PARALLEL` rather than in isolation.
```

- [ ] **Step 2: Verify the README renders sensibly**

Run: `grep -n "llm-concurrency" README.md`
Expected: the new paragraph is present, immediately following the existing thresholds paragraph

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document --llm-concurrency option"
```
