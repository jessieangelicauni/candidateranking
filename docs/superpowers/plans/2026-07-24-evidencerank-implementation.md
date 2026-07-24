# EvidenceRank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build EvidenceRank, a LangGraph-orchestrated multi-agent pipeline that ranks candidate resumes against a job description using LLM semantic reasoning grounded in quoted CV evidence, running entirely on local Ollama models.

**Architecture:** A single LangGraph `StateGraph` carries a `PipelineState` through five nodes: extract profiles → embedding pre-filter (cut) → LLM judge each surviving candidate → LLM pool calibration (whole-pool reconciliation) → deterministic hallucination check (fuzzy-match every quote against source CV text). Each LLM stage is a small, independently testable function in `src/evidencerank/agents/`; the graph wires them together. Per-candidate work is implemented as a loop inside each node (not LangGraph's dynamic `Send` fan-out) because local Ollama inference on a single GPU serializes model calls anyway — looping keeps the code simpler and version-stable without losing the pipeline's logical structure.

**Tech Stack:** Python 3.12, `uv` for project/dependency management, LangGraph + LangChain (`langchain-ollama`) for orchestration and LLM calls, Ollama for local model serving, `sentence-transformers` (`bge-small-en-v1.5`) for the embedding pre-filter, `rapidfuzz` for hallucination/grounding verification, `pydantic` v2 for schemas, `click` for the CLI, `deepeval` + `scipy` for the research evaluation harness, `pytest` for tests.

## Global Constraints

- No deterministic, rule-based formula may decide a candidate's fit tier, rating, or rank — those are LLM judgments only. (spec: "Hard constraint")
- Deterministic logic is allowed only for the embedding pre-filter (a compute-saving cut, not a fit judgment) and the hallucination checker (grounding verification, not a fit judgment).
- All LLM calls use local Ollama models; no external API model calls. Model names per stage are configurable via `EVIDENCERANK_MODEL_<STAGE>` environment variables, not hardcoded.
- Default models (sized for a 16GB VRAM GPU): `qwen2.5:7b-instruct` for JD parsing and CV extraction; `qwen2.5:14b-instruct` for judging and calibration.
- Identity fields (name, email, phone, location) must be redacted from what the Judge stage sees (blind evaluation); the unredacted raw CV text is still used by the hallucination checker, which must verify against true source text.
- Hallucination checker default behavior is to flag unverified quotes as warnings, not to silently drop or auto-reject a candidate; auto re-judge is opt-in only.
- The DeepEval evaluation harness lives in `evaluation/`, separate from the production pipeline in `src/evidencerank/`.
- Output is JSON (full trail) + Markdown (ranking table) — no web UI in this build.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/evidencerank/__init__.py`
- Create: `src/evidencerank/agents/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: an installable `evidencerank` package importable from `src/evidencerank`, and an importable top-level `evaluation` package, both resolvable by `uv run pytest`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "evidencerank"
version = "0.1.0"
description = "Evidence-grounded multi-agent LLM candidate ranking pipeline"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "langchain-ollama>=0.2",
    "pydantic>=2.7",
    "pdfplumber>=0.11",
    "sentence-transformers>=3.0",
    "rapidfuzz>=3.9",
    "click>=8.1",
    "deepeval>=1.1",
    "scipy>=1.13",
    "numpy>=1.26",
]

[project.scripts]
evidencerank = "evidencerank.cli:rank"

[dependency-groups]
dev = [
    "pytest>=8.2",
    "fpdf2>=2.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/evidencerank"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests", "evaluation"]
```

- [ ] **Step 2: Create package init files**

`src/evidencerank/__init__.py`:
```python
```

`src/evidencerank/agents/__init__.py`:
```python
```

`evaluation/__init__.py`:
```python
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
report.json
report.md
*.egg-info/
```

- [ ] **Step 4: Sync dependencies**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, installs all dependencies without error.

- [ ] **Step 5: Verify the package imports**

Run: `uv run python -c "import evidencerank; import evaluation; print('ok')"`
Expected: prints `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/evidencerank/__init__.py src/evidencerank/agents/__init__.py evaluation/__init__.py
git commit -m "chore: scaffold evidencerank project with uv"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `src/evidencerank/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `JDRequirements`, `ContactInfo`, `WorkHistoryEntry`, `EducationEntry`, `ProjectEntry`, `ExtractedProfileFields`, `CandidateProfile`, `PrefilterResult`, `Tier`, `EvidenceClaim`, `JudgeVerdict`, `JudgeResult`, `CalibratedResult`, `CalibrationOutput`, `HallucinationReport` — used by every later task.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
import pytest
from pydantic import ValidationError

from evidencerank.models import (
    CalibratedResult,
    CalibrationOutput,
    CandidateProfile,
    ContactInfo,
    EducationEntry,
    EvidenceClaim,
    ExtractedProfileFields,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    JudgeVerdict,
    PrefilterResult,
    ProjectEntry,
    Tier,
    WorkHistoryEntry,
)


def test_jd_requirements_defaults():
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    assert jd.nice_to_have_skills == []
    assert jd.min_experience_years == 0
    assert jd.responsibilities == []


def test_candidate_profile_inherits_extracted_fields():
    profile = CandidateProfile(
        candidate_id="c1",
        raw_cv_text="raw text",
        contact=ContactInfo(name="Jane Doe"),
        skills=["Python", "SQL"],
        work_history=[
            WorkHistoryEntry(
                title="ML Engineer", company="Acme", start_date="2020",
                end_date="2023", achievements=["Shipped model"],
            )
        ],
        education=[EducationEntry(degree="BSc", institution="MIT", year="2019")],
        projects=[ProjectEntry(name="Recommender", description="...", tech=["Python"])],
    )
    assert isinstance(profile, ExtractedProfileFields)
    assert profile.candidate_id == "c1"
    assert profile.contact.name == "Jane Doe"


def test_judge_result_rating_bounds():
    with pytest.raises(ValidationError):
        JudgeVerdict(tier=Tier.STRONG_FIT, rating=11, evidence=[])
    with pytest.raises(ValidationError):
        JudgeVerdict(tier=Tier.STRONG_FIT, rating=0, evidence=[])
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python")],
    )
    result = JudgeResult(candidate_id="c1", **verdict.model_dump())
    assert result.candidate_id == "c1"
    assert result.rating == 8


def test_calibration_output_wraps_list():
    output = CalibrationOutput(
        results=[
            CalibratedResult(
                candidate_id="c1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Top of pool",
            )
        ]
    )
    assert len(output.results) == 1
    assert output.results[0].final_rank == 1


def test_prefilter_result_pass_flag():
    result = PrefilterResult(candidate_id="c1", similarity=0.72, passed=True)
    assert result.passed is True


def test_hallucination_report_all_verified_property():
    verified = HallucinationReport(candidate_id="c1", unverified_quotes=[])
    unverified = HallucinationReport(candidate_id="c1", unverified_quotes=["fabricated quote"])
    assert verified.all_verified is True
    assert unverified.all_verified is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.models'`

- [ ] **Step 3: Write `src/evidencerank/models.py`**

```python
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""


class WorkHistoryEntry(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: str
    achievements: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    degree: str
    institution: str
    year: str


class ProjectEntry(BaseModel):
    name: str
    description: str
    tech: list[str] = Field(default_factory=list)


class JDRequirements(BaseModel):
    title: str
    required_skills: list[str]
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = 0
    education: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class ExtractedProfileFields(BaseModel):
    contact: ContactInfo
    skills: list[str]
    work_history: list[WorkHistoryEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)


class CandidateProfile(ExtractedProfileFields):
    candidate_id: str
    raw_cv_text: str


class PrefilterResult(BaseModel):
    candidate_id: str
    similarity: float
    passed: bool


class Tier(str, Enum):
    STRONG_FIT = "Strong Fit"
    MODERATE_FIT = "Moderate Fit"
    WEAK_FIT = "Weak Fit"
    NOT_A_FIT = "Not a Fit"


class EvidenceClaim(BaseModel):
    claim: str
    quote: str


class JudgeVerdict(BaseModel):
    tier: Tier
    rating: int = Field(ge=1, le=10)
    evidence: list[EvidenceClaim]


class JudgeResult(JudgeVerdict):
    candidate_id: str


class CalibratedResult(BaseModel):
    candidate_id: str
    final_rank: int
    tier: Tier
    rating: int
    calibration_notes: str


class CalibrationOutput(BaseModel):
    results: list[CalibratedResult]


class HallucinationReport(BaseModel):
    candidate_id: str
    unverified_quotes: list[str] = Field(default_factory=list)

    @property
    def all_verified(self) -> bool:
        return len(self.unverified_quotes) == 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/models.py tests/test_models.py
git commit -m "feat: add evidencerank pydantic schemas"
```

---

## Task 3: File Loaders

**Files:**
- Create: `src/evidencerank/io.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Consumes: nothing internal.
- Produces: `load_text_file(path) -> str`, `load_resume_text(path) -> str` — used by `cli.py` (Task 14).

- [ ] **Step 1: Write the failing test**

`tests/test_io.py`:
```python
from pathlib import Path

from fpdf import FPDF

from evidencerank.io import load_resume_text, load_text_file


def _make_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_load_resume_text_extracts_pdf_content(tmp_path):
    pdf_path = tmp_path / "resume.pdf"
    _make_pdf(pdf_path, "Jane Example\nPython, SQL, Docker")

    text = load_resume_text(pdf_path)

    assert "Jane Example" in text
    assert "Python, SQL, Docker" in text


def test_load_text_file_reads_plain_text(tmp_path):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("Machine Learning Engineer\nPython required", encoding="utf-8")

    assert load_text_file(jd_path) == "Machine Learning Engineer\nPython required"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.io'`

- [ ] **Step 3: Write `src/evidencerank/io.py`**

```python
from pathlib import Path

import pdfplumber


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_resume_text(path: str | Path) -> str:
    with pdfplumber.open(Path(path)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_io.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/io.py tests/test_io.py
git commit -m "feat: add JD/resume file loaders"
```

---

## Task 4: Ollama Model Configuration

**Files:**
- Create: `src/evidencerank/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing internal.
- Produces: `resolve_model_name(stage: str) -> str`, `get_chat_model(stage: str, temperature: float = 0.0) -> ChatOllama`, `DEFAULT_MODELS: dict[str, str]` — used by every agent module (Tasks 6, 7, 9, 10).

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:
```python
import pytest

from evidencerank.llm import DEFAULT_MODELS, resolve_model_name


def test_resolve_model_name_returns_default():
    assert resolve_model_name("judge") == DEFAULT_MODELS["judge"]


def test_resolve_model_name_respects_env_override(monkeypatch):
    monkeypatch.setenv("EVIDENCERANK_MODEL_JUDGE", "custom-model:latest")
    assert resolve_model_name("judge") == "custom-model:latest"


def test_resolve_model_name_rejects_unknown_stage():
    with pytest.raises(ValueError):
        resolve_model_name("not_a_stage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.llm'`

- [ ] **Step 3: Write `src/evidencerank/llm.py`**

```python
import os

from langchain_ollama import ChatOllama

DEFAULT_MODELS: dict[str, str] = {
    "jd_parser": "qwen2.5:7b-instruct",
    "cv_extractor": "qwen2.5:7b-instruct",
    "judge": "qwen2.5:14b-instruct",
    "calibrator": "qwen2.5:14b-instruct",
}

_ENV_PREFIX = "EVIDENCERANK_MODEL_"


def resolve_model_name(stage: str) -> str:
    if stage not in DEFAULT_MODELS:
        raise ValueError(f"Unknown stage: {stage!r}. Known stages: {sorted(DEFAULT_MODELS)}")
    env_key = f"{_ENV_PREFIX}{stage.upper()}"
    return os.environ.get(env_key, DEFAULT_MODELS[stage])


def get_chat_model(stage: str, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=resolve_model_name(stage), temperature=temperature)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/llm.py tests/test_llm.py
git commit -m "feat: add per-stage Ollama model configuration"
```

---

## Task 5: Identity Redaction (Fairness)

**Files:**
- Create: `src/evidencerank/privacy.py`
- Test: `tests/test_privacy.py`

**Interfaces:**
- Consumes: `ContactInfo` (Task 2).
- Produces: `redact_identity(raw_cv_text: str, contact: ContactInfo) -> str` — used by the Judge agent (Task 9).

- [ ] **Step 1: Write the failing test**

`tests/test_privacy.py`:
```python
from evidencerank.models import ContactInfo
from evidencerank.privacy import redact_identity


def test_redact_identity_removes_name_email_phone_location():
    contact = ContactInfo(
        name="Daniel Taylor",
        email="daniel.taylor@protonmail.com",
        phone="745-310-7622x683",
        location="Hensleyton, UAE",
    )
    raw_text = (
        "Daniel Taylor\n"
        "daniel.taylor@protonmail.com | 745-310-7622x683 | Hensleyton, UAE\n"
        "SUMMARY\nSenior frontend engineer with 7 years of Python experience."
    )

    redacted = redact_identity(raw_text, contact)

    assert "Daniel Taylor" not in redacted
    assert "daniel.taylor@protonmail.com" not in redacted
    assert "745-310-7622x683" not in redacted
    assert "Hensleyton, UAE" not in redacted
    assert "Python experience" in redacted


def test_redact_identity_handles_empty_contact_fields():
    contact = ContactInfo()
    raw_text = "Skills: Python, SQL"

    redacted = redact_identity(raw_text, contact)

    assert redacted == "Skills: Python, SQL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_privacy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.privacy'`

- [ ] **Step 3: Write `src/evidencerank/privacy.py`**

```python
import re

from evidencerank.models import ContactInfo

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s()x]{7,}\d)")


def redact_identity(raw_cv_text: str, contact: ContactInfo) -> str:
    text = raw_cv_text
    if contact.name:
        text = text.replace(contact.name, "[REDACTED NAME]")
    if contact.location:
        text = text.replace(contact.location, "[REDACTED LOCATION]")
    text = _EMAIL_RE.sub("[REDACTED EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED PHONE]", text)
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_privacy.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/privacy.py tests/test_privacy.py
git commit -m "feat: add identity redaction for blind Judge evaluation"
```

---

## Task 6: JD Parser Agent

**Files:**
- Create: `src/evidencerank/agents/jd_parser.py`
- Test: `tests/agents/test_jd_parser.py`
- Create: `tests/agents/__init__.py`

**Interfaces:**
- Consumes: `get_chat_model` (Task 4), `JDRequirements` (Task 2).
- Produces: `parse_jd(jd_text: str) -> JDRequirements` — used by `cli.py` (Task 14).

- [ ] **Step 1: Write the failing test**

`tests/agents/__init__.py`:
```python
```

`tests/agents/test_jd_parser.py`:
```python
from unittest.mock import MagicMock

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.models import JDRequirements


def test_parse_jd_returns_structured_requirements(monkeypatch):
    expected = JDRequirements(
        title="Machine Learning Engineer",
        required_skills=["Python", "PyTorch"],
        nice_to_have_skills=["Docker"],
        min_experience_years=2,
        education="Bachelor's in Computer Science",
        responsibilities=["Train models"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = expected
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.jd_parser.get_chat_model",
        lambda stage: fake_chat_model,
    )

    result = parse_jd("Machine Learning Engineer JD text...")

    assert result == expected
    fake_chat_model.with_structured_output.assert_called_once_with(JDRequirements)
    fake_structured_model.invoke.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_jd_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.jd_parser'`

- [ ] **Step 3: Write `src/evidencerank/agents/jd_parser.py`**

```python
from evidencerank.llm import get_chat_model
from evidencerank.models import JDRequirements

JD_PARSER_PROMPT = """You are an expert technical recruiter. Read the job description below \
and extract its requirements precisely. Do not invent requirements that are not stated or \
clearly implied by the text.

Job description:
{jd_text}
"""


def parse_jd(jd_text: str) -> JDRequirements:
    model = get_chat_model("jd_parser").with_structured_output(JDRequirements)
    return model.invoke(JD_PARSER_PROMPT.format(jd_text=jd_text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_jd_parser.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/jd_parser.py tests/agents/__init__.py tests/agents/test_jd_parser.py
git commit -m "feat: add JD Parser agent"
```

---

## Task 7: CV Extractor Agent

**Files:**
- Create: `src/evidencerank/agents/cv_extractor.py`
- Test: `tests/agents/test_cv_extractor.py`

**Interfaces:**
- Consumes: `get_chat_model` (Task 4), `ExtractedProfileFields`, `CandidateProfile`, `ContactInfo` (Task 2).
- Produces: `extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile` — used by `graph.py` (Task 12).

- [ ] **Step 1: Write the failing test**

`tests/agents/test_cv_extractor.py`:
```python
from unittest.mock import MagicMock

from evidencerank.agents.cv_extractor import extract_cv
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_cv_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.cv_extractor'`

- [ ] **Step 3: Write `src/evidencerank/agents/cv_extractor.py`**

```python
from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, ExtractedProfileFields

CV_EXTRACTOR_PROMPT = """You are an expert technical recruiter. Read the resume below and \
extract the candidate's contact info, skills, work history, education, and projects exactly \
as stated. Do not infer skills or experience that are not explicitly present in the text.

Resume:
{cv_text}
"""


def extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile:
    model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
    fields = model.invoke(CV_EXTRACTOR_PROMPT.format(cv_text=cv_text))
    return CandidateProfile(
        candidate_id=candidate_id,
        raw_cv_text=cv_text,
        **fields.model_dump(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_cv_extractor.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/cv_extractor.py tests/agents/test_cv_extractor.py
git commit -m "feat: add CV Extractor agent"
```

---

## Task 8: Embedding Pre-filter

**Files:**
- Create: `src/evidencerank/agents/prefilter.py`
- Test: `tests/agents/test_prefilter.py`

**Interfaces:**
- Consumes: `PrefilterResult` (Task 2).
- Produces: `cosine_similarity(a, b) -> float`, `prefilter_candidate(candidate_id, jd_required_skills, candidate_skills, threshold=0.5) -> PrefilterResult` — used by `graph.py` (Task 12).

- [ ] **Step 1: Write the failing test**

`tests/agents/test_prefilter.py`:
```python
import numpy as np

from evidencerank.agents.prefilter import cosine_similarity, prefilter_candidate


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_prefilter_candidate_matching_skills_passes():
    result = prefilter_candidate(
        candidate_id="c1",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Python", "PyTorch", "Deep Learning", "Model training"],
        threshold=0.4,
    )
    assert result.candidate_id == "c1"
    assert result.passed is True
    assert 0.0 <= result.similarity <= 1.0


def test_prefilter_candidate_unrelated_skills_fails():
    result = prefilter_candidate(
        candidate_id="c2",
        jd_required_skills=["Python", "Machine Learning", "PyTorch"],
        candidate_skills=["Photoshop", "Illustrator", "Figma"],
        threshold=0.6,
    )
    assert result.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_prefilter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.prefilter'`

- [ ] **Step 3: Write `src/evidencerank/agents/prefilter.py`**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

from evidencerank.models import PrefilterResult

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def prefilter_candidate(
    candidate_id: str,
    jd_required_skills: list[str],
    candidate_skills: list[str],
    threshold: float = 0.5,
) -> PrefilterResult:
    embedder = _get_embedder()
    jd_text = ", ".join(jd_required_skills)
    candidate_text = ", ".join(candidate_skills)
    jd_vec, candidate_vec = embedder.encode([jd_text, candidate_text])
    similarity = cosine_similarity(jd_vec, candidate_vec)
    return PrefilterResult(
        candidate_id=candidate_id,
        similarity=similarity,
        passed=similarity >= threshold,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_prefilter.py -v`
Expected: PASS (4 tests). Note: the first run downloads the `bge-small-en-v1.5` model (~130MB) and will be slower; subsequent runs use the local cache.

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/prefilter.py tests/agents/test_prefilter.py
git commit -m "feat: add embedding-based skill pre-filter"
```

---

## Task 9: Candidate Judge Agent

**Files:**
- Create: `src/evidencerank/agents/judge.py`
- Test: `tests/agents/test_judge.py`

**Interfaces:**
- Consumes: `get_chat_model` (Task 4), `redact_identity` (Task 5), `JDRequirements`, `CandidateProfile`, `JudgeVerdict`, `JudgeResult` (Task 2).
- Produces: `judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult` — used by `graph.py` (Task 12).

- [ ] **Step 1: Write the failing test**

`tests/agents/test_judge.py`:
```python
from unittest.mock import MagicMock

from evidencerank.agents.judge import judge_candidate
from evidencerank.models import (
    ContactInfo,
    EvidenceClaim,
    ExtractedProfileFields,
    CandidateProfile,
    JDRequirements,
    JudgeVerdict,
    Tier,
)


def _make_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="c1",
        raw_cv_text="Daniel Taylor\nSkills: Python, Machine Learning\n5 years of Python experience",
        contact=ContactInfo(name="Daniel Taylor", email="daniel@example.com"),
        skills=["Python", "Machine Learning"],
    )


def test_judge_candidate_redacts_identity_before_prompting(monkeypatch):
    verdict = JudgeVerdict(
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = verdict
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.judge.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])

    result = judge_candidate(jd, _make_profile())

    assert result.candidate_id == "c1"
    assert result.tier == Tier.STRONG_FIT
    assert result.rating == 8
    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "Daniel Taylor" not in prompt_sent
    assert "[REDACTED NAME]" in prompt_sent
    fake_chat_model.with_structured_output.assert_called_once_with(JudgeVerdict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.judge'`

- [ ] **Step 3: Write `src/evidencerank/agents/judge.py`**

```python
from evidencerank.llm import get_chat_model
from evidencerank.models import CandidateProfile, JDRequirements, JudgeResult, JudgeVerdict
from evidencerank.privacy import redact_identity

JUDGE_PROMPT = """You are an experienced technical recruiter evaluating a candidate for a role. \
Reason holistically like a human recruiter: longer relevant experience increases confidence, \
measurable impact matters more than job titles, and technical skill alignment with the role's \
requirements matters most. Give your own holistic judgment — do not compute or describe a \
numeric formula.

Every claim you make MUST be backed by a verbatim quote copied exactly from the resume text \
below. Never quote text that does not appear in the resume text.

Job requirements:
{jd_requirements}

Candidate resume (identity redacted):
{redacted_cv_text}

Candidate structured profile:
skills: {skills}
work_history: {work_history}
education: {education}
projects: {projects}

Assign a tier (Strong Fit, Moderate Fit, Weak Fit, Not a Fit) and a rating from 1 to 10.
"""


def judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult:
    redacted_text = redact_identity(profile.raw_cv_text, profile.contact)
    model = get_chat_model("judge").with_structured_output(JudgeVerdict)
    prompt = JUDGE_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        redacted_cv_text=redacted_text,
        skills=profile.skills,
        work_history=[entry.model_dump() for entry in profile.work_history],
        education=[entry.model_dump() for entry in profile.education],
        projects=[entry.model_dump() for entry in profile.projects],
    )
    verdict = model.invoke(prompt)
    return JudgeResult(candidate_id=profile.candidate_id, **verdict.model_dump())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/judge.py tests/agents/test_judge.py
git commit -m "feat: add Candidate Judge agent with blind evaluation"
```

---

## Task 10: Pool Calibrator Agent

**Files:**
- Create: `src/evidencerank/agents/calibrator.py`
- Test: `tests/agents/test_calibrator.py`

**Interfaces:**
- Consumes: `get_chat_model` (Task 4), `JDRequirements`, `JudgeResult`, `CalibrationOutput`, `CalibratedResult` (Task 2).
- Produces: `calibrate_pool(jd: JDRequirements, judge_results: list[JudgeResult]) -> list[CalibratedResult]` — used by `graph.py` (Task 12).

- [ ] **Step 1: Write the failing test**

`tests/agents/test_calibrator.py`:
```python
from unittest.mock import MagicMock

from evidencerank.agents.calibrator import calibrate_pool
from evidencerank.models import (
    CalibratedResult,
    CalibrationOutput,
    EvidenceClaim,
    JDRequirements,
    JudgeResult,
    Tier,
)


def test_calibrate_pool_returns_ranked_list(monkeypatch):
    expected_output = CalibrationOutput(
        results=[
            CalibratedResult(
                candidate_id="c1", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Deepest relevant experience in pool",
            ),
            CalibratedResult(
                candidate_id="c2", final_rank=2, tier=Tier.MODERATE_FIT,
                rating=6, calibration_notes="Adjacent domain experience only",
            ),
        ]
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = expected_output
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.calibrator.get_chat_model",
        lambda stage: fake_chat_model,
    )
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    judge_results = [
        JudgeResult(
            candidate_id="c1", tier=Tier.STRONG_FIT, rating=9,
            evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
        ),
        JudgeResult(
            candidate_id="c2", tier=Tier.MODERATE_FIT, rating=6,
            evidence=[EvidenceClaim(claim="Some ML exposure", quote="1 year of ML projects")],
        ),
    ]

    results = calibrate_pool(jd, judge_results)

    assert results == expected_output.results
    fake_chat_model.with_structured_output.assert_called_once_with(CalibrationOutput)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_calibrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.calibrator'`

- [ ] **Step 3: Write `src/evidencerank/agents/calibrator.py`**

```python
from evidencerank.llm import get_chat_model
from evidencerank.models import CalibratedResult, CalibrationOutput, JDRequirements, JudgeResult

CALIBRATOR_PROMPT = """You are an experienced technical recruiter performing a final calibration \
pass across a shortlisted candidate pool for one role. Each candidate below was already judged \
independently; your job is to reconcile relative ordering across the whole pool — correct for \
any leniency or anchoring drift between the independent judgments — and produce a final rank \
order (1 = best fit). Briefly explain any adjustment you make in calibration_notes.

Job requirements:
{jd_requirements}

Independent judge results for every candidate in the pool:
{judge_results}
"""


def calibrate_pool(jd: JDRequirements, judge_results: list[JudgeResult]) -> list[CalibratedResult]:
    model = get_chat_model("calibrator").with_structured_output(CalibrationOutput)
    prompt = CALIBRATOR_PROMPT.format(
        jd_requirements=jd.model_dump_json(),
        judge_results=[result.model_dump() for result in judge_results],
    )
    output = model.invoke(prompt)
    return output.results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_calibrator.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/calibrator.py tests/agents/test_calibrator.py
git commit -m "feat: add Pool Calibrator agent"
```

---

## Task 11: Hallucination Checker

**Files:**
- Create: `src/evidencerank/agents/hallucination_checker.py`
- Test: `tests/agents/test_hallucination_checker.py`

**Interfaces:**
- Consumes: `JudgeResult`, `HallucinationReport` (Task 2).
- Produces: `check_evidence(judge_result: JudgeResult, raw_cv_text: str, threshold: float = 85.0) -> HallucinationReport` — used by `graph.py` (Task 12).

- [ ] **Step 1: Write the failing test**

`tests/agents/test_hallucination_checker.py`:
```python
from evidencerank.agents.hallucination_checker import check_evidence
from evidencerank.models import EvidenceClaim, JudgeResult, Tier

RAW_CV_TEXT = "Daniel Taylor\nSkills: Python, Machine Learning\n5 years of Python experience"


def test_check_evidence_verifies_real_quote():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )

    report = check_evidence(judge_result, RAW_CV_TEXT)

    assert report.candidate_id == "c1"
    assert report.all_verified is True


def test_check_evidence_flags_fabricated_quote():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience"),
            EvidenceClaim(claim="Led a team of 10 engineers", quote="managed a team of 10 engineers"),
        ],
    )

    report = check_evidence(judge_result, RAW_CV_TEXT)

    assert report.all_verified is False
    assert "managed a team of 10 engineers" in report.unverified_quotes
    assert "5 years of Python experience" not in report.unverified_quotes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_hallucination_checker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.hallucination_checker'`

- [ ] **Step 3: Write `src/evidencerank/agents/hallucination_checker.py`**

```python
from rapidfuzz import fuzz

from evidencerank.models import HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0


def check_evidence(
    judge_result: JudgeResult,
    raw_cv_text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    unverified = []
    for claim in judge_result.evidence:
        score = fuzz.partial_ratio(claim.quote, raw_cv_text)
        if score < threshold:
            unverified.append(claim.quote)
    return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_hallucination_checker.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/hallucination_checker.py tests/agents/test_hallucination_checker.py
git commit -m "feat: add deterministic hallucination/grounding checker"
```

---

## Task 12: LangGraph Pipeline

**Files:**
- Create: `src/evidencerank/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `extract_cv` (Task 7), `prefilter_candidate` (Task 8), `judge_candidate` (Task 9), `calibrate_pool` (Task 10), `check_evidence`, `DEFAULT_THRESHOLD` (Task 11), all schemas (Task 2).
- Produces: `PipelineState` (TypedDict), `build_graph() -> CompiledGraph` whose `.invoke(initial_state) -> PipelineState` — used by `cli.py` (Task 14).

- [ ] **Step 1: Write the failing test**

`tests/test_graph.py`:
```python
from unittest.mock import MagicMock

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


def test_graph_runs_extract_prefilter_judge_calibrate_hallucination(monkeypatch):
    jd = JDRequirements(title="ML Engineer", required_skills=["Python", "PyTorch"])

    def fake_extract_cv(candidate_id, raw_text):
        return CandidateProfile(
            candidate_id=candidate_id,
            raw_cv_text=raw_text,
            contact=ContactInfo(name=candidate_id),
            skills=["Python"] if candidate_id == "strong" else ["Photoshop"],
        )

    def fake_prefilter_candidate(candidate_id, jd_required_skills, candidate_skills, threshold):
        passed = candidate_id == "strong"
        return PrefilterResult(candidate_id=candidate_id, similarity=0.9 if passed else 0.1, passed=passed)

    def fake_judge_candidate(jd_requirements, profile):
        return JudgeResult(
            candidate_id=profile.candidate_id,
            tier=Tier.STRONG_FIT,
            rating=9,
            evidence=[EvidenceClaim(claim="Strong fit", quote="Python")],
        )

    def fake_calibrate_pool(jd_requirements, judge_results):
        return [
            CalibratedResult(
                candidate_id=r.candidate_id, final_rank=1, tier=r.tier,
                rating=r.rating, calibration_notes="Only candidate in pool",
            )
            for r in judge_results
        ]

    def fake_check_evidence(judge_result, raw_cv_text, threshold):
        return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=[])

    monkeypatch.setattr("evidencerank.graph.extract_cv", fake_extract_cv)
    monkeypatch.setattr("evidencerank.graph.prefilter_candidate", fake_prefilter_candidate)
    monkeypatch.setattr("evidencerank.graph.judge_candidate", fake_judge_candidate)
    monkeypatch.setattr("evidencerank.graph.calibrate_pool", fake_calibrate_pool)
    monkeypatch.setattr("evidencerank.graph.check_evidence", fake_check_evidence)

    graph = build_graph()
    final_state = graph.invoke(
        {
            "jd": jd,
            "raw_resumes": {"strong": "Python resume text", "weak": "Photoshop resume text"},
        }
    )

    assert set(final_state["profiles"].keys()) == {"strong", "weak"}
    assert final_state["dropped"] == [
        {"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}
    ]
    assert set(final_state["judge_results"].keys()) == {"strong"}
    assert len(final_state["calibrated_results"]) == 1
    assert final_state["calibrated_results"][0].candidate_id == "strong"
    assert final_state["hallucination_reports"]["strong"].all_verified is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.graph'`

- [ ] **Step 3: Write `src/evidencerank/graph.py`**

```python
from typing import TypedDict

from langgraph.graph import END, StateGraph

from evidencerank.agents.calibrator import calibrate_pool
from evidencerank.agents.cv_extractor import extract_cv
from evidencerank.agents.hallucination_checker import DEFAULT_THRESHOLD, check_evidence
from evidencerank.agents.judge import judge_candidate
from evidencerank.agents.prefilter import prefilter_candidate
from evidencerank.models import (
    CalibratedResult,
    CandidateProfile,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    PrefilterResult,
)


class PipelineState(TypedDict, total=False):
    jd: JDRequirements
    raw_resumes: dict[str, str]
    profiles: dict[str, CandidateProfile]
    prefilter_results: dict[str, PrefilterResult]
    dropped: list[dict[str, str]]
    judge_results: dict[str, JudgeResult]
    calibrated_results: list[CalibratedResult]
    hallucination_reports: dict[str, HallucinationReport]
    prefilter_threshold: float
    hallucination_threshold: float


def extract_profiles_node(state: PipelineState) -> dict:
    profiles = {
        candidate_id: extract_cv(candidate_id, raw_text)
        for candidate_id, raw_text in state["raw_resumes"].items()
    }
    return {"profiles": profiles}


def prefilter_node(state: PipelineState) -> dict:
    threshold = state.get("prefilter_threshold", 0.5)
    results: dict[str, PrefilterResult] = {}
    dropped: list[dict[str, str]] = []
    for candidate_id, profile in state["profiles"].items():
        result = prefilter_candidate(
            candidate_id,
            state["jd"].required_skills,
            profile.skills,
            threshold=threshold,
        )
        results[candidate_id] = result
        if not result.passed:
            dropped.append(
                {"candidate_id": candidate_id, "reason": "pre-filter: no relevant skill overlap"}
            )
    return {"prefilter_results": results, "dropped": dropped}


def judge_node(state: PipelineState) -> dict:
    judge_results: dict[str, JudgeResult] = {}
    for candidate_id, result in state["prefilter_results"].items():
        if not result.passed:
            continue
        profile = state["profiles"][candidate_id]
        judge_results[candidate_id] = judge_candidate(state["jd"], profile)
    return {"judge_results": judge_results}


def calibrate_node(state: PipelineState) -> dict:
    calibrated = calibrate_pool(state["jd"], list(state["judge_results"].values()))
    return {"calibrated_results": calibrated}


def hallucination_check_node(state: PipelineState) -> dict:
    threshold = state.get("hallucination_threshold", DEFAULT_THRESHOLD)
    reports = {}
    for candidate_id, judge_result in state["judge_results"].items():
        raw_text = state["profiles"][candidate_id].raw_cv_text
        reports[candidate_id] = check_evidence(judge_result, raw_text, threshold=threshold)
    return {"hallucination_reports": reports}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract_profiles", extract_profiles_node)
    graph.add_node("prefilter", prefilter_node)
    graph.add_node("judge", judge_node)
    graph.add_node("calibrate", calibrate_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    graph.set_entry_point("extract_profiles")
    graph.add_edge("extract_profiles", "prefilter")
    graph.add_edge("prefilter", "judge")
    graph.add_edge("judge", "calibrate")
    graph.add_edge("calibrate", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph.compile()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/graph.py tests/test_graph.py
git commit -m "feat: wire pipeline stages into a LangGraph StateGraph"
```

---

## Task 13: Report Generation

**Files:**
- Create: `src/evidencerank/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `PipelineState` shape (Task 12), `CalibratedResult`, `Tier` (Task 2).
- Produces: `build_json_report(state) -> dict`, `write_json_report(state, path)`, `build_markdown_report(state) -> str`, `write_markdown_report(state, path)` — used by `cli.py` (Task 14).

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import json

from evidencerank.models import (
    CalibratedResult,
    EvidenceClaim,
    HallucinationReport,
    JDRequirements,
    JudgeResult,
    Tier,
)
from evidencerank.report import build_json_report, build_markdown_report, write_json_report, write_markdown_report


def _sample_state():
    return {
        "jd": JDRequirements(title="ML Engineer", required_skills=["Python"]),
        "dropped": [{"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}],
        "judge_results": {
            "strong": JudgeResult(
                candidate_id="strong", tier=Tier.STRONG_FIT, rating=9,
                evidence=[EvidenceClaim(claim="Strong Python background", quote="5 years Python")],
            )
        },
        "calibrated_results": [
            CalibratedResult(
                candidate_id="strong", final_rank=1, tier=Tier.STRONG_FIT,
                rating=9, calibration_notes="Only surviving candidate",
            )
        ],
        "hallucination_reports": {
            "strong": HallucinationReport(candidate_id="strong", unverified_quotes=[]),
        },
    }


def test_build_json_report_contains_all_sections():
    report = build_json_report(_sample_state())

    assert report["jd"]["title"] == "ML Engineer"
    assert report["dropped"][0]["candidate_id"] == "weak"
    assert report["judge_results"]["strong"]["rating"] == 9
    assert report["calibrated_results"][0]["final_rank"] == 1
    assert report["hallucination_reports"]["strong"]["unverified_quotes"] == []


def test_write_json_report_writes_valid_json(tmp_path):
    out_path = tmp_path / "report.json"
    write_json_report(_sample_state(), out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "strong"


def test_build_markdown_report_has_ranked_table_row():
    markdown = build_markdown_report(_sample_state())

    assert "| Rank | Candidate | Tier | Rating | Calibration Notes |" in markdown
    assert "| 1 | strong | Strong Fit | 9 | Only surviving candidate |" in markdown


def test_write_markdown_report_writes_file(tmp_path):
    out_path = tmp_path / "report.md"
    write_markdown_report(_sample_state(), out_path)

    assert "strong" in out_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.report'`

- [ ] **Step 3: Write `src/evidencerank/report.py`**

```python
import json
from pathlib import Path


def build_json_report(state: dict) -> dict:
    return {
        "jd": state["jd"].model_dump(),
        "dropped": state.get("dropped", []),
        "judge_results": {
            candidate_id: result.model_dump()
            for candidate_id, result in state.get("judge_results", {}).items()
        },
        "calibrated_results": [result.model_dump() for result in state.get("calibrated_results", [])],
        "hallucination_reports": {
            candidate_id: report.model_dump()
            for candidate_id, report in state.get("hallucination_reports", {}).items()
        },
    }


def write_json_report(state: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_json_report(state), indent=2), encoding="utf-8")


def build_markdown_report(state: dict) -> str:
    lines = ["| Rank | Candidate | Tier | Rating | Calibration Notes |", "|---|---|---|---|---|"]
    for result in sorted(state.get("calibrated_results", []), key=lambda r: r.final_rank):
        lines.append(
            f"| {result.final_rank} | {result.candidate_id} | {result.tier.value} "
            f"| {result.rating} | {result.calibration_notes} |"
        )
    return "\n".join(lines)


def write_markdown_report(state: dict, path: str | Path) -> None:
    Path(path).write_text(build_markdown_report(state), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/report.py tests/test_report.py
git commit -m "feat: add JSON and Markdown report generation"
```

---

## Task 14: CLI Entry Point

**Files:**
- Create: `src/evidencerank/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_text_file`, `load_resume_text` (Task 3), `parse_jd` (Task 6), `build_graph` (Task 12), `write_json_report`, `write_markdown_report` (Task 13).
- Produces: `rank` click command, registered as the `evidencerank` console script (Task 1's `pyproject.toml`).

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner
from fpdf import FPDF

from evidencerank.cli import rank
from evidencerank.models import CalibratedResult, JDRequirements, Tier


def _make_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def test_rank_command_writes_json_and_markdown_reports(tmp_path, monkeypatch):
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

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    runner = CliRunner()
    result = runner.invoke(
        rank,
        [
            "--jd", str(jd_path),
            "--resumes-dir", str(resumes_dir),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_json.exists()
    assert out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["calibrated_results"][0]["candidate_id"] == "candidate1"
    invoked_state = fake_graph.invoke.call_args[0][0]
    assert "candidate1" in invoked_state["raw_resumes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.cli'`

- [ ] **Step 3: Write `src/evidencerank/cli.py`**

```python
from pathlib import Path

import click

from evidencerank.agents.jd_parser import parse_jd
from evidencerank.graph import build_graph
from evidencerank.io import load_resume_text, load_text_file
from evidencerank.report import write_json_report, write_markdown_report


@click.command()
@click.option("--jd", "jd_path", required=True, type=click.Path(exists=True))
@click.option("--resumes-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out-json", default="report.json", type=click.Path())
@click.option("--out-md", default="report.md", type=click.Path())
@click.option("--prefilter-threshold", default=0.5, type=float)
@click.option("--hallucination-threshold", default=85.0, type=float)
def rank(jd_path, resumes_dir, out_json, out_md, prefilter_threshold, hallucination_threshold):
    """Rank every resume in RESUMES_DIR against the job description at JD."""
    jd_text = load_text_file(jd_path)
    jd_requirements = parse_jd(jd_text)

    raw_resumes = {
        pdf_path.stem: load_resume_text(pdf_path)
        for pdf_path in sorted(Path(resumes_dir).glob("*.pdf"))
    }

    graph = build_graph()
    final_state = graph.invoke(
        {
            "jd": jd_requirements,
            "raw_resumes": raw_resumes,
            "prefilter_threshold": prefilter_threshold,
            "hallucination_threshold": hallucination_threshold,
        }
    )

    write_json_report(final_state, out_json)
    write_markdown_report(final_state, out_md)
    click.echo(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    rank()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/cli.py tests/test_cli.py
git commit -m "feat: add evidencerank CLI entry point"
```

---

## Task 15: DeepEval Custom Metrics

**Files:**
- Create: `evaluation/metrics.py`
- Test: `evaluation/test_metrics.py`

**Interfaces:**
- Consumes: nothing internal (standalone research evaluation harness, deliberately decoupled from `src/evidencerank`).
- Produces: `groundedness_metric`, `recruiter_alignment_metric`, `evidence_relevancy_metric` (each a `deepeval.metrics.GEval`), `build_test_case(jd_requirements_text, judge_result_text, cv_text) -> LLMTestCase` — used by researchers running DeepEval against pipeline output, and by Task 16's rank-stability script's sibling test suite.

- [ ] **Step 1: Write the failing test**

`evaluation/test_metrics.py`:
```python
from deepeval.test_case import LLMTestCaseParams

from evaluation.metrics import (
    build_test_case,
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)


def test_groundedness_metric_uses_context_param():
    assert groundedness_metric.threshold == 0.7
    assert LLMTestCaseParams.CONTEXT in groundedness_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in groundedness_metric.evaluation_params


def test_recruiter_alignment_metric_uses_input_and_output_params():
    assert LLMTestCaseParams.INPUT in recruiter_alignment_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in recruiter_alignment_metric.evaluation_params


def test_evidence_relevancy_metric_uses_input_and_output_params():
    assert LLMTestCaseParams.INPUT in evidence_relevancy_metric.evaluation_params
    assert LLMTestCaseParams.ACTUAL_OUTPUT in evidence_relevancy_metric.evaluation_params


def test_build_test_case_wraps_fields_correctly():
    case = build_test_case("JD requirements text", "Judge output text", "CV text")

    assert case.input == "JD requirements text"
    assert case.actual_output == "Judge output text"
    assert case.context == ["CV text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest evaluation/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.metrics'`

- [ ] **Step 3: Write `evaluation/metrics.py`**

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

groundedness_metric = GEval(
    name="Groundedness",
    criteria=(
        "Determine whether every claim in 'actual_output' is directly supported by a "
        "verbatim quote that appears in 'context'. Penalize any claim not backed by a "
        "quote found in the context."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.7,
)

recruiter_alignment_metric = GEval(
    name="RecruiterAlignment",
    criteria=(
        "Determine whether 'actual_output' reflects sound recruiter judgment given "
        "'input' (the job requirements): does it weigh relevant experience depth, "
        "measurable impact, and technical skill alignment appropriately, rather than "
        "superficial keyword matching?"
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

evidence_relevancy_metric = GEval(
    name="EvidenceRelevancy",
    criteria=(
        "Determine whether the quoted evidence in 'actual_output' is relevant to the "
        "job requirement claim it supports in 'input', not merely present somewhere in "
        "the resume."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)


def build_test_case(jd_requirements_text: str, judge_result_text: str, cv_text: str) -> LLMTestCase:
    return LLMTestCase(
        input=jd_requirements_text,
        actual_output=judge_result_text,
        context=[cv_text],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest evaluation/test_metrics.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py evaluation/test_metrics.py
git commit -m "feat: add DeepEval groundedness/alignment/relevancy metrics"
```

---

## Task 16: Rank-Stability Script

**Files:**
- Create: `evaluation/rank_stability.py`
- Test: `evaluation/test_rank_stability.py`

**Interfaces:**
- Consumes: JSON report files produced by `write_json_report` (Task 13), specifically the `calibrated_results` list shape (`candidate_id`, `final_rank`).
- Produces: `load_rank_map(report_path) -> dict[str, int]`, `rank_stability(report_paths: list[str]) -> dict` — standalone research script, run manually across repeated pipeline runs on the same input.

- [ ] **Step 1: Write the failing test**

`evaluation/test_rank_stability.py`:
```python
import json
from pathlib import Path

from evaluation.rank_stability import load_rank_map, rank_stability


def _write_report(path: Path, ranks: dict[str, int]) -> None:
    data = {
        "calibrated_results": [
            {"candidate_id": candidate_id, "final_rank": final_rank}
            for candidate_id, final_rank in ranks.items()
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_rank_map_reads_candidate_ranks(tmp_path):
    report_path = tmp_path / "run1.json"
    _write_report(report_path, {"a": 1, "b": 2})

    rank_map = load_rank_map(report_path)

    assert rank_map == {"a": 1, "b": 2}


def test_rank_stability_identical_rankings_scores_one(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_report(run1, {"a": 1, "b": 2, "c": 3})
    _write_report(run2, {"a": 1, "b": 2, "c": 3})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == 1.0
    assert result["mean_kendall_tau"] == 1.0
    assert result["n_runs"] == 2


def test_rank_stability_reversed_rankings_scores_negative_one(tmp_path):
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_report(run1, {"a": 1, "b": 2, "c": 3})
    _write_report(run2, {"a": 3, "b": 2, "c": 1})

    result = rank_stability([str(run1), str(run2)])

    assert result["mean_spearman"] == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest evaluation/test_rank_stability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.rank_stability'`

- [ ] **Step 3: Write `evaluation/rank_stability.py`**

```python
import json
from itertools import combinations
from pathlib import Path

from scipy.stats import kendalltau, spearmanr


def load_rank_map(report_path: str | Path) -> dict[str, int]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {entry["candidate_id"]: entry["final_rank"] for entry in data["calibrated_results"]}


def rank_stability(report_paths: list[str]) -> dict:
    rank_maps = [load_rank_map(path) for path in report_paths]
    candidate_ids = sorted(rank_maps[0].keys())

    spearman_scores = []
    kendall_scores = []
    for map_a, map_b in combinations(rank_maps, 2):
        ranks_a = [map_a[candidate_id] for candidate_id in candidate_ids]
        ranks_b = [map_b[candidate_id] for candidate_id in candidate_ids]
        spearman_scores.append(spearmanr(ranks_a, ranks_b).correlation)
        kendall_scores.append(kendalltau(ranks_a, ranks_b).correlation)

    return {
        "mean_spearman": sum(spearman_scores) / len(spearman_scores),
        "mean_kendall_tau": sum(kendall_scores) / len(kendall_scores),
        "n_runs": len(report_paths),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest evaluation/test_rank_stability.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/rank_stability.py evaluation/test_rank_stability.py
git commit -m "feat: add rank-stability evaluation script"
```

---

## Task 17: README and Manual End-to-End Verification

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks; this is the final task.

- [ ] **Step 1: Write `README.md`**

```markdown
# EvidenceRank

Evidence-grounded multi-agent LLM pipeline for ranking IT job candidates against a job
description, using local Ollama models orchestrated with LangGraph. See
`docs/superpowers/specs/2026-07-24-evidencerank-design.md` for the full design.

## Setup

1. Install [Ollama](https://ollama.com) and start the server: `ollama serve`
2. Pull the default models:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull qwen2.5:14b-instruct
   ```
3. Install project dependencies: `uv sync`
4. Run the test suite: `uv run pytest`

## Running the pipeline

```bash
uv run evidencerank \
  --jd machine_learning_engineer.txt \
  --resumes-dir . \
  --out-json report.json \
  --out-md report.md
```

This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results) and `report.md` (a ranked Markdown table).

## Model configuration

Override any stage's model with an environment variable:

```bash
export EVIDENCERANK_MODEL_JUDGE=qwen2.5:32b-instruct
```

Valid stages: `jd_parser`, `cv_extractor`, `judge`, `calibrator`.

## Research evaluation harness

The `evaluation/` package is separate from the production pipeline (`src/evidencerank/`):

- `evaluation/metrics.py` — DeepEval `GEval` metrics (Groundedness, RecruiterAlignment,
  EvidenceRelevancy) to run against pipeline output.
- `evaluation/rank_stability.py` — computes Spearman/Kendall-tau rank correlation across
  repeated runs on the same input, to report LLM judgment consistency.

To measure rank stability, run the pipeline N times on the same JD/resumes with
different `--out-json` paths, then:

```python
from evaluation.rank_stability import rank_stability
print(rank_stability(["run1.json", "run2.json", "run3.json"]))
```
```

- [ ] **Step 2: Manually verify the end-to-end pipeline against the sample data**

This step requires a running Ollama server with the models pulled (Setup steps 1-2
above); it is a manual verification, not an automated test, since it depends on live
local inference.

Run:
```bash
uv run evidencerank --jd machine_learning_engineer.txt --resumes-dir . --out-json report.json --out-md report.md
```

Expected: the command exits 0, prints `Wrote report.json and report.md`, and both files
are created. Open `report.md` and confirm it contains a ranked table of the 5 sample
candidates with tiers and evidence-backed calibration notes; open `report.json` and spot
check that quoted evidence in `judge_results` actually appears in the corresponding
candidate's original resume text.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```
