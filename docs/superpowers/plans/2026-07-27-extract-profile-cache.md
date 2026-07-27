# Extract-Profile Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent, content-addressed cache for the `extract_profiles` pipeline stage so repeated dev-loop runs against the same resumes skip re-extracting a candidate profile whose inputs (resume text, extractor prompt, extractor model) haven't changed.

**Architecture:** A tiny generic JSON file-cache module (`src/evidencerank/cache.py`) provides key computation and load/save primitives with no framework beyond what's needed. `src/evidencerank/agents/cv_extractor.py` gains a `cached_extract_cv` wrapper around the existing (unchanged) `extract_cv`, keyed on a SHA-256 hash of `(cv_text, CV_EXTRACTOR_PROMPT, resolved cv_extractor model name)`. `graph.py`'s `extract_profiles_node` is switched to call the cached wrapper instead of the raw function — a one-line swap, no state-shape change.

**Tech Stack:** Python 3.11, `hashlib` (stdlib), `json` (stdlib), `pathlib.Path`, `pytest`.

## Global Constraints

- Only `extract_profiles` is cached — `judge`, `calibrate`, `prefilter`, and `hallucination_check` are explicitly out of scope for this plan.
- Cache location is fixed at `.cache/evidencerank/extract_profiles/` — no CLI flag or env var to configure it, no `--no-cache` bypass flag.
- Cache key must be derived from `cv_text`, `CV_EXTRACTOR_PROMPT`, and `resolve_model_name("cv_extractor")` — not from `candidate_id` or file path — so identical resume content always hits cache regardless of filename, and editing the prompt or switching models self-invalidates with no manual cache-clearing step.
- The cached payload is the `ExtractedProfileFields`-shaped subset of a `CandidateProfile` (`contact`, `skills`, `work_history`, `education`, `projects`) — not `candidate_id` or `raw_cv_text`, which are supplied by the caller at lookup time.
- `extract_cv()` itself stays unchanged and directly unit-testable with no cache side effects.
- `uv run pytest` must pass after every task.

---

### Task 1: Cache primitives

**Files:**
- Create: `src/evidencerank/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `compute_cache_key(*parts: str) -> str`, `load_cached_json(cache_dir: Path, key: str) -> dict | None`, `save_cached_json(cache_dir: Path, key: str, data: dict) -> None` — these three functions are what Task 2 imports and calls.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
from evidencerank.cache import compute_cache_key, load_cached_json, save_cached_json


def test_compute_cache_key_same_inputs_same_key():
    key_a = compute_cache_key("resume text", "prompt text", "model-name")
    key_b = compute_cache_key("resume text", "prompt text", "model-name")

    assert key_a == key_b


def test_compute_cache_key_different_inputs_different_key():
    key_a = compute_cache_key("resume text", "prompt text", "model-name")
    key_b = compute_cache_key("different resume text", "prompt text", "model-name")

    assert key_a != key_b


def test_compute_cache_key_avoids_boundary_collision():
    # Without a delimiter between parts, ("ab", "c") and ("a", "bc") would
    # hash identically (both concatenate to "abc").
    key_a = compute_cache_key("ab", "c")
    key_b = compute_cache_key("a", "bc")

    assert key_a != key_b


def test_load_cached_json_returns_none_when_missing(tmp_path):
    result = load_cached_json(tmp_path, "nonexistent-key")

    assert result is None


def test_save_and_load_cached_json_round_trips(tmp_path):
    data = {"skills": ["Python", "SQL"], "contact": {"name": "Jane Doe"}}

    save_cached_json(tmp_path, "some-key", data)
    result = load_cached_json(tmp_path, "some-key")

    assert result == data


def test_save_cached_json_creates_cache_dir_if_missing(tmp_path):
    cache_dir = tmp_path / "nested" / "cache" / "dir"

    save_cached_json(cache_dir, "some-key", {"a": 1})

    assert (cache_dir / "some-key.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.cache'`

- [ ] **Step 3: Implement `src/evidencerank/cache.py`**

```python
import hashlib
import json
from pathlib import Path


def compute_cache_key(*parts: str) -> str:
    # \x1f (ASCII unit separator) can't appear in ordinary text, so it can't
    # cause a boundary collision between adjacent parts the way a plain
    # join or a printable delimiter like "|" could.
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_cached_json(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_cached_json(cache_dir: Path, key: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/cache.py tests/test_cache.py
git commit -m "feat: add content-addressed JSON file cache primitives"
```

---

### Task 2: Cached CV extraction

**Files:**
- Modify: `src/evidencerank/agents/cv_extractor.py`
- Test: `tests/agents/test_cv_extractor.py`

**Interfaces:**
- Consumes: `compute_cache_key`, `load_cached_json`, `save_cached_json` from `evidencerank.cache` (Task 1); `resolve_model_name` from `evidencerank.llm` (already exists — `resolve_model_name(stage: str) -> str`, see `src/evidencerank/llm.py`).
- Produces: `cached_extract_cv(candidate_id: str, cv_text: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> CandidateProfile` — this is what Task 3's `graph.py` calls instead of `extract_cv`. `DEFAULT_CACHE_DIR = Path(".cache/evidencerank/extract_profiles")` is the fixed default; tests override it with `cache_dir=tmp_path`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_cv_extractor.py` (add `import json` and the new imports at the top, alongside the existing `from unittest.mock import MagicMock` / `from evidencerank.agents.cv_extractor import extract_cv` / `from evidencerank.models import ContactInfo, ExtractedProfileFields` lines — the file becomes):

```python
import json
from unittest.mock import MagicMock

from evidencerank.agents.cv_extractor import CV_EXTRACTOR_PROMPT, cached_extract_cv, extract_cv
from evidencerank.cache import compute_cache_key, save_cached_json
from evidencerank.llm import resolve_model_name
from evidencerank.models import ContactInfo, ExtractedProfileFields


def test_extract_cv_assembles_candidate_profile(monkeypatch):
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
        skills=["Python", "SQL"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = extract_cv("c1", "Jane Doe resume text...")

    assert profile.candidate_id == "c1"
    assert profile.raw_cv_text == "Jane Doe resume text..."
    assert profile.contact.name == "Jane Doe"
    assert profile.skills == ["Python", "SQL"]
    fake_chat_model.with_structured_output.assert_called_once_with(ExtractedProfileFields)


def test_cached_extract_cv_skips_llm_on_cache_hit(tmp_path, monkeypatch):
    cv_text = "Jane Doe resume text..."
    cached_fields = {
        "contact": {"name": "Jane Doe", "email": "jane@example.com", "phone": "", "location": ""},
        "skills": ["Python", "SQL"],
        "work_history": [],
        "education": [],
        "projects": [],
    }
    key = compute_cache_key(cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"))
    save_cached_json(tmp_path, key, cached_fields)

    fake_chat_model = MagicMock()
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = cached_extract_cv("c1", cv_text, cache_dir=tmp_path)

    assert profile.candidate_id == "c1"
    assert profile.raw_cv_text == cv_text
    assert profile.contact.name == "Jane Doe"
    assert profile.skills == ["Python", "SQL"]
    fake_chat_model.with_structured_output.assert_not_called()


def test_cached_extract_cv_calls_llm_and_writes_cache_on_miss(tmp_path, monkeypatch):
    cv_text = "John Smith resume text..."
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="John Smith", email="john@example.com"),
        skills=["Go", "Rust"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = cached_extract_cv("c2", cv_text, cache_dir=tmp_path)

    assert profile.candidate_id == "c2"
    assert profile.skills == ["Go", "Rust"]
    fake_chat_model.with_structured_output.assert_called_once_with(ExtractedProfileFields)

    key = compute_cache_key(cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"))
    cache_file = tmp_path / f"{key}.json"
    assert cache_file.exists()
    cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached_data["skills"] == ["Go", "Rust"]
    assert cached_data["contact"]["name"] == "John Smith"


def test_cached_extract_cv_second_call_with_same_input_skips_llm(tmp_path, monkeypatch):
    cv_text = "Repeat candidate resume text..."
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="Repeat Candidate"),
        skills=["Java"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    first = cached_extract_cv("c3", cv_text, cache_dir=tmp_path)
    second = cached_extract_cv("c3", cv_text, cache_dir=tmp_path)

    assert first.skills == second.skills == ["Java"]
    fake_structured_model.invoke.assert_called_once()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/agents/test_cv_extractor.py -v`
Expected: `test_extract_cv_assembles_candidate_profile` PASSES (unchanged behavior); the three new `cached_extract_cv` tests FAIL with `ImportError: cannot import name 'cached_extract_cv'`.

- [ ] **Step 3: Implement `cached_extract_cv`**

Replace the full contents of `src/evidencerank/agents/cv_extractor.py` with:

```python
from pathlib import Path

from evidencerank.cache import compute_cache_key, load_cached_json, save_cached_json
from evidencerank.llm import get_chat_model, resolve_model_name
from evidencerank.models import CandidateProfile, ExtractedProfileFields

CV_EXTRACTOR_PROMPT = """You are an expert technical recruiter. Read the resume below and \
extract the candidate's contact info, skills, work history, education, and projects exactly \
as stated. Do not infer skills or experience that are not explicitly present in the text.

Resume:
{cv_text}
"""

DEFAULT_CACHE_DIR = Path(".cache/evidencerank/extract_profiles")


def extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile:
    model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
    fields = model.invoke(CV_EXTRACTOR_PROMPT.format(cv_text=cv_text))
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
    key = compute_cache_key(cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"))
    cached = load_cached_json(cache_dir, key)
    if cached is not None:
        return CandidateProfile(candidate_id=candidate_id, raw_cv_text=cv_text, **cached)

    profile = extract_cv(candidate_id, cv_text)
    save_cached_json(
        cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
    )
    return profile
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_cv_extractor.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/agents/cv_extractor.py tests/agents/test_cv_extractor.py
git commit -m "feat: cache extract_profiles output, keyed on resume content + prompt + model"
```

---

### Task 3: Wire the cache into the pipeline

**Files:**
- Modify: `src/evidencerank/graph.py`
- Modify: `tests/test_graph.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: `cached_extract_cv(candidate_id: str, cv_text: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> CandidateProfile` from Task 2.
- Produces: no new interface — `extract_profiles_node`'s return shape (`{"profiles": dict[str, CandidateProfile]}`) is unchanged; this task only swaps which function it calls internally.

- [ ] **Step 1: Write the failing test**

In `tests/test_graph.py`, change the monkeypatch target from `extract_cv` to `cached_extract_cv` (the fake function itself, `fake_extract_cv`, stays as-is — only the patched attribute name changes):

Change:
```python
    monkeypatch.setattr("evidencerank.graph.extract_cv", fake_extract_cv)
```
to:
```python
    monkeypatch.setattr("evidencerank.graph.cached_extract_cv", fake_extract_cv)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAILS with `AttributeError: <module 'evidencerank.graph'> does not have the attribute 'cached_extract_cv'` (graph.py doesn't import or use that name yet).

- [ ] **Step 3: Update `graph.py`**

Change the import line:
```python
from evidencerank.agents.cv_extractor import extract_cv
```
to:
```python
from evidencerank.agents.cv_extractor import cached_extract_cv
```

Change `extract_profiles_node`:
```python
def extract_profiles_node(state: PipelineState) -> dict:
    click.echo("Running stage: extract_profiles")
    profiles = {
        candidate_id: cached_extract_cv(candidate_id, raw_text)
        for candidate_id, raw_text in state["raw_resumes"].items()
    }
    return {"profiles": profiles}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Add `.cache/` to `.gitignore`**

Add a line to `.gitignore` (anywhere in the existing list is fine — append at the end):
```
.cache/
```

- [ ] **Step 6: Document the cache in README**

In `README.md`, insert this paragraph right after the existing paragraph that ends `...hallucination check results) and `report.md` (a ranked Markdown table).` and before the `Two thresholds are tunable...` paragraph:

```
Extracted candidate profiles are cached at `.cache/evidencerank/extract_profiles/`,
keyed by a hash of the resume text plus the CV-extractor prompt and model — so
re-running the pipeline against the same resumes skips re-extracting a profile
whose inputs haven't changed. Editing the extractor prompt or switching the
`EVIDENCERANK_MODEL_CV_EXTRACTOR` model automatically invalidates the relevant
cache entries; deleting the directory forces full re-extraction.
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/evidencerank/graph.py tests/test_graph.py .gitignore README.md
git commit -m "feat: wire extract-profile cache into the pipeline graph"
```
