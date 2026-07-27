# Judge Grounding and Corrective Hallucination Checking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the root causes behind bad GEval scores (Groundedness 7.7% pass, RecruiterAlignment 11.5% pass, EvidenceRelevancy 0% pass, 14/26 hallucination-flagged) found in a real pipeline run: an unscoped Judge quoting instruction, weak claim-to-evidence relevance, an audit-only hallucination check with no effect on ranking, whitespace-fragile fuzzy matching, and no stage-latency visibility.

**Architecture:** Prompt-only fixes to the Judge (`src/evidencerank/agents/judge.py`) to scope quoting and require relevance; a pure-function evidence filter (`src/evidencerank/agents/hallucination_checker.py`) plus a reordered LangGraph pipeline (`judge → hallucination_check → calibrate`, was `judge → calibrate → hallucination_check`) so unverified evidence is stripped before calibration instead of only being reported after; a new Markdown column surfacing the hallucination outcome per candidate; and per-stage wall-clock timing threaded through `report.json` into `eval_report.md`.

**Tech Stack:** Python 3.11, LangGraph (`StateGraph`), Pydantic v2, `rapidfuzz`, `pytest`, `click`, existing `uv` toolchain.

## Global Constraints

- No new LLM calls anywhere in this plan — the hallucination check stays deterministic (`rapidfuzz`-based), per the existing design rationale in `docs/superpowers/specs/2026-07-24-evidencerank-design.md` (avoid an LLM checking an LLM for hallucination).
- CLI flags and their defaults are unchanged: `--prefilter-threshold` (0.5), `--hallucination-threshold` (85.0). Do not rename or remove any existing flag.
- `report.json`'s existing top-level keys (`jd`, `profiles`, `prefilter_results`, `dropped`, `judge_results`, `calibrated_results`, `hallucination_reports`) keep their existing shapes; the only structural addition is a new optional `stage_timings` key. Code reading `report.json` (`evaluation/report.py`) must tolerate its absence (old report files without it must not raise `KeyError`).
- `uv run pytest` must pass after every task.
- Model configuration (`src/evidencerank/llm.py`, `DEFAULT_MODELS`) is out of scope — no model swap in this plan (see spec's "Out of scope").

---

### Task 1: Scope the Judge prompt to CV-text-only, relevance-checked quoting

**Files:**
- Modify: `src/evidencerank/agents/judge.py:5-27` (the `JUDGE_PROMPT` constant)
- Test: `tests/agents/test_judge.py`

**Interfaces:**
- Consumes: nothing new — `judge_candidate(jd: JDRequirements, profile: CandidateProfile) -> JudgeResult` keeps its existing signature (`src/evidencerank/agents/judge.py:30`).
- Produces: nothing new — behavior change is entirely in the prompt text sent to the model.

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_judge.py` (below the existing two tests, same file, same imports already present):

```python
def test_judge_candidate_prompt_scopes_quoting_to_resume_text_only(monkeypatch):
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

    judge_candidate(jd, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "ONLY from that block" in prompt_sent
    assert 'Never quote the "Candidate structured profile" section' in prompt_sent
    assert "background context only — do not quote from this section" in prompt_sent


def test_judge_candidate_prompt_requires_claim_relevant_quotes(monkeypatch):
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

    judge_candidate(jd, _make_profile())

    prompt_sent = fake_structured_model.invoke.call_args[0][0]
    assert "directly demonstrate the specific skill, technology, or responsibility" in prompt_sent
    assert "results-driven engineer with 4+ years of experience" in prompt_sent
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: the two new tests FAIL (assertion errors — the phrases aren't in the current prompt yet); the two pre-existing tests still PASS.

- [ ] **Step 3: Rewrite `JUDGE_PROMPT`**

Replace `src/evidencerank/agents/judge.py:5-27` with:

```python
JUDGE_PROMPT = """You are an experienced technical recruiter evaluating a candidate for a role. \
Reason holistically like a human recruiter: longer relevant experience increases confidence, \
measurable impact matters more than job titles, and technical skill alignment with the role's \
requirements matters most. Give your own holistic judgment — do not compute or describe a \
numeric formula.

Every claim you make MUST be backed by a verbatim quote copied exactly, character-for-character, \
from the "Candidate resume" text block below — and ONLY from that block. Never quote the \
"Candidate structured profile" section (skills/work_history/education/projects): it is \
paraphrased summary data for your background context only, and none of its wording is \
guaranteed to appear in the resume text. For example, quoting "skills: ['TensorFlow']" is NOT \
allowed — that is Python list syntax from the structured profile, not resume text.

Each quote must also directly demonstrate the specific skill, technology, or responsibility \
named in its claim, not merely be true and present somewhere in the resume. For example, if the \
claim is "candidate has machine learning experience," a quote that only establishes years of \
experience in general (e.g. "results-driven engineer with 4+ years of experience") is NOT \
sufficient evidence — the quote must itself name machine learning, a related framework, or a \
related task.

Job requirements:
{jd_requirements}

Candidate resume (identity redacted):
{redacted_cv_text}

Candidate structured profile (background context only — do not quote from this section):
skills: {skills}
work_history: {work_history}
education: {education}
projects: {projects}

Assign a tier (Strong Fit, Moderate Fit, Weak Fit, Not a Fit) and a rating from 1 to 10.
"""
```

Nothing else in `judge.py` changes — `judge_candidate()`'s body still formats the same five placeholders.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/test_judge.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS (no other file references `JUDGE_PROMPT`'s exact text).

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/agents/judge.py tests/agents/test_judge.py
git commit -m "fix: scope Judge quoting to CV text and require claim-relevant quotes"
```

---

### Task 2: Normalize whitespace before fuzzy-matching evidence quotes

**Files:**
- Modify: `src/evidencerank/agents/hallucination_checker.py`
- Test: `tests/agents/test_hallucination_checker.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `check_evidence(judge_result: JudgeResult, raw_cv_text: str, threshold: float = 85.0) -> HallucinationReport` — signature unchanged, only the internal comparison changes. `HallucinationReport.unverified_quotes` still holds the claim's *original* (non-normalized) quote text, exactly as before, so downstream consumers (report rendering, Task 3's filter) see quotes matching what the Judge actually produced.

- [ ] **Step 1: Write the failing test**

Add to `tests/agents/test_hallucination_checker.py` (same file, after the existing two tests):

```python
def test_check_evidence_verifies_quote_despite_whitespace_differences():
    raw_cv_text = "Reduced latency by 40%"
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(
                claim="Reduced latency",
                quote="Reduced\n\n\nlatency\n\n\nby\n\n\n40%",
            )
        ],
    )

    report = check_evidence(judge_result, raw_cv_text)

    assert report.all_verified is True
```

This exact quote/text pair was verified empirically: `fuzz.partial_ratio` on the raw (unnormalized) strings scores ~72.7 (below the 85.0 default threshold, so today's implementation would flag it as a false-positive hallucination), while normalizing whitespace on both sides first scores 100.0.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/agents/test_hallucination_checker.py -v`
Expected: `test_check_evidence_verifies_quote_despite_whitespace_differences` FAILS (`report.all_verified` is `False`); the two pre-existing tests still PASS.

- [ ] **Step 3: Add whitespace normalization**

Replace the full contents of `src/evidencerank/agents/hallucination_checker.py` with:

```python
import re

from rapidfuzz import fuzz

from evidencerank.models import HallucinationReport, JudgeResult

DEFAULT_THRESHOLD = 85.0

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def check_evidence(
    judge_result: JudgeResult,
    raw_cv_text: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> HallucinationReport:
    normalized_cv_text = _normalize_whitespace(raw_cv_text)
    unverified = []
    for claim in judge_result.evidence:
        normalized_quote = _normalize_whitespace(claim.quote)
        score = fuzz.partial_ratio(normalized_quote, normalized_cv_text)
        if score < threshold:
            unverified.append(claim.quote)
    return HallucinationReport(candidate_id=judge_result.candidate_id, unverified_quotes=unverified)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/agents/test_hallucination_checker.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/agents/hallucination_checker.py tests/agents/test_hallucination_checker.py
git commit -m "fix: normalize whitespace before fuzzy-matching evidence quotes"
```

---

### Task 3: Make hallucination checking corrective — filter unverified evidence before calibration

**Files:**
- Modify: `src/evidencerank/agents/hallucination_checker.py` (add `filter_verified_evidence`)
- Modify: `src/evidencerank/graph.py` (reorder edges, update `hallucination_check_node`)
- Test: `tests/agents/test_hallucination_checker.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `HallucinationReport`, `JudgeResult` (from `evidencerank.models`, unchanged).
- Produces: `filter_verified_evidence(judge_result: JudgeResult, report: HallucinationReport) -> JudgeResult` — a new pure function in `hallucination_checker.py` that later tasks/consumers can rely on. Graph edges become `judge → hallucination_check → calibrate → END` (previously `judge → calibrate → hallucination_check → END`); `hallucination_check_node` now returns both `hallucination_reports` (audit trail, unchanged shape) and a filtered `judge_results` (evidence pruned) that overwrites the state key `calibrate_node` reads from.

- [ ] **Step 1: Write the failing test for `filter_verified_evidence`**

Add to `tests/agents/test_hallucination_checker.py`:

```python
from evidencerank.agents.hallucination_checker import check_evidence, filter_verified_evidence
```

(add `filter_verified_evidence` to the existing `from evidencerank.agents.hallucination_checker import check_evidence` import line at the top of the file)

```python
def test_filter_verified_evidence_removes_only_unverified_claims():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[
            EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience"),
            EvidenceClaim(claim="Led a team", quote="managed a team of 10 engineers"),
        ],
    )
    report = check_evidence(judge_result, RAW_CV_TEXT)

    filtered = filter_verified_evidence(judge_result, report)

    assert [claim.quote for claim in filtered.evidence] == ["5 years of Python experience"]
    assert filtered.candidate_id == "c1"
    assert filtered.tier == Tier.STRONG_FIT
    assert filtered.rating == 8


def test_filter_verified_evidence_keeps_all_claims_when_fully_verified():
    judge_result = JudgeResult(
        candidate_id="c1",
        tier=Tier.STRONG_FIT,
        rating=8,
        evidence=[EvidenceClaim(claim="Has Python experience", quote="5 years of Python experience")],
    )
    report = check_evidence(judge_result, RAW_CV_TEXT)

    filtered = filter_verified_evidence(judge_result, report)

    assert filtered.evidence == judge_result.evidence
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_hallucination_checker.py -v`
Expected: `test_filter_verified_evidence_*` FAIL with `ImportError`/`AttributeError` (`filter_verified_evidence` doesn't exist yet).

- [ ] **Step 3: Add `filter_verified_evidence`**

Append to `src/evidencerank/agents/hallucination_checker.py` (after `check_evidence`):

```python
def filter_verified_evidence(judge_result: JudgeResult, report: HallucinationReport) -> JudgeResult:
    unverified_quotes = set(report.unverified_quotes)
    verified_evidence = [
        claim for claim in judge_result.evidence if claim.quote not in unverified_quotes
    ]
    return judge_result.model_copy(update={"evidence": verified_evidence})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/test_hallucination_checker.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Write the failing graph test**

Replace the full contents of `tests/test_graph.py` with:

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

    def fake_extract_cv(candidate_id, raw_text):
        return CandidateProfile(
            candidate_id=candidate_id,
            raw_cv_text=raw_text,
            contact=ContactInfo(name=candidate_id),
            skills=["Python"] if candidate_id != "weak" else ["Photoshop"],
        )

    def fake_prefilter_candidate(candidate_id, jd_required_skills, candidate_skills, threshold):
        passed = candidate_id != "weak"
        return PrefilterResult(candidate_id=candidate_id, similarity=0.9 if passed else 0.1, passed=passed)

    def fake_judge_candidate(jd_requirements, profile):
        return JudgeResult(
            candidate_id=profile.candidate_id,
            tier=Tier.STRONG_FIT,
            rating=9,
            evidence=[
                EvidenceClaim(claim="Strong fit", quote="Python"),
                EvidenceClaim(claim="Fabricated claim", quote="FABRICATED unverifiable quote text"),
            ],
        )

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

    monkeypatch.setattr("evidencerank.graph.extract_cv", fake_extract_cv)
    monkeypatch.setattr("evidencerank.graph.prefilter_candidate", fake_prefilter_candidate)
    monkeypatch.setattr("evidencerank.graph.judge_candidate", fake_judge_candidate)
    monkeypatch.setattr("evidencerank.graph.calibrate_pool", fake_calibrate_pool)
    monkeypatch.setattr("evidencerank.graph.check_evidence", fake_check_evidence)

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
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAILS — with today's graph order (`calibrate` before `hallucination_check`), `calibrate_pool` receives the unfiltered evidence (both "Python" and the "FABRICATED..." quote), so regression guard 4 fails.

- [ ] **Step 7: Reorder the graph and make `hallucination_check_node` corrective**

In `src/evidencerank/graph.py`, change the import line (currently `from evidencerank.agents.hallucination_checker import DEFAULT_THRESHOLD, check_evidence`) to:

```python
from evidencerank.agents.hallucination_checker import (
    DEFAULT_THRESHOLD,
    check_evidence,
    filter_verified_evidence,
)
```

Replace `hallucination_check_node` (`graph.py:80-87`) with:

```python
def hallucination_check_node(state: PipelineState) -> dict:
    click.echo("Running stage: hallucination_check")
    threshold = state.get("hallucination_threshold", DEFAULT_THRESHOLD)
    reports = {}
    filtered_judge_results = {}
    for candidate_id, judge_result in state["judge_results"].items():
        raw_text = state["profiles"][candidate_id].raw_cv_text
        report = check_evidence(judge_result, raw_text, threshold=threshold)
        reports[candidate_id] = report
        filtered_judge_results[candidate_id] = filter_verified_evidence(judge_result, report)
    return {"hallucination_reports": reports, "judge_results": filtered_judge_results}
```

Replace the edge wiring in `build_graph()` (`graph.py:98-103`) — currently:

```python
    graph.set_entry_point("extract_profiles")
    graph.add_edge("extract_profiles", "prefilter")
    graph.add_edge("prefilter", "judge")
    graph.add_edge("judge", "calibrate")
    graph.add_edge("calibrate", "hallucination_check")
    graph.add_edge("hallucination_check", END)
```

with:

```python
    graph.set_entry_point("extract_profiles")
    graph.add_edge("extract_profiles", "prefilter")
    graph.add_edge("prefilter", "judge")
    graph.add_edge("judge", "hallucination_check")
    graph.add_edge("hallucination_check", "calibrate")
    graph.add_edge("calibrate", END)
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/evidencerank/agents/hallucination_checker.py src/evidencerank/graph.py tests/agents/test_hallucination_checker.py tests/test_graph.py
git commit -m "feat: strip unverified evidence before calibration, not just after"
```

---

### Task 4: Surface hallucination outcome in `report.md`

**Files:**
- Modify: `src/evidencerank/report.py`
- Modify: `README.md` (one-line mention of the new column)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `state["hallucination_reports"]: dict[str, HallucinationReport]` (already produced by the graph as of Task 3).
- Produces: `build_markdown_report(state: dict) -> str` — same signature, output table gains one column ("Hallucination Flags") between "Key Evidence" and "Calibration Notes".

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_report.py:84-192` (from `test_build_markdown_report_has_ranked_table_row` through the end of the file) with:

```python
def test_build_markdown_report_has_ranked_table_row():
    markdown = build_markdown_report(_sample_state())

    assert (
        "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags "
        "| Calibration Notes |" in markdown
    )
    assert (
        "| 1 | strong | Strong Fit | 9 | Strong Python background: 5 years Python "
        "| — | Only surviving candidate |" in markdown
    )


def test_write_markdown_report_writes_file(tmp_path):
    out_path = tmp_path / "report.md"
    write_markdown_report(_sample_state(), out_path)

    assert "strong" in out_path.read_text(encoding="utf-8")


def test_build_markdown_report_orders_rows_by_rank_ascending():
    state = _sample_state()
    state["calibrated_results"] = [
        CalibratedResult(
            candidate_id="third", final_rank=3, tier=Tier.WEAK_FIT,
            rating=4, calibration_notes="Ranked third",
        ),
        CalibratedResult(
            candidate_id="first", final_rank=1, tier=Tier.STRONG_FIT,
            rating=9, calibration_notes="Ranked first",
        ),
        CalibratedResult(
            candidate_id="second", final_rank=2, tier=Tier.MODERATE_FIT,
            rating=6, calibration_notes="Ranked second",
        ),
    ]

    markdown = build_markdown_report(state)

    lines = markdown.splitlines()
    header, separator, *row_lines = lines
    candidate_order = [line.split("|")[2].strip() for line in row_lines]

    assert candidate_order == ["first", "second", "third"]
    assert lines.index("| 1 | first | Strong Fit | 9 |  | — | Ranked first |") < \
        lines.index("| 2 | second | Moderate Fit | 6 |  | — | Ranked second |") < \
        lines.index("| 3 | third | Weak Fit | 4 |  | — | Ranked third |")


def test_build_markdown_report_escapes_pipes_and_newlines_in_notes():
    state = _sample_state()
    state["calibrated_results"] = [
        CalibratedResult(
            candidate_id="strong", final_rank=1, tier=Tier.STRONG_FIT, rating=9,
            calibration_notes="Great fit | but watch out\nfor gaps in employment",
        )
    ]

    markdown = build_markdown_report(state)
    lines = markdown.splitlines()

    # Header + separator + exactly one data row: no extra rows from the embedded newline.
    assert len(lines) == 3
    data_row = lines[2]

    # No embedded newline leaked into the output.
    assert "\n" not in data_row

    # The literal pipe from the notes text was escaped (backslash-pipe), not left
    # as a bare separator that would split the notes into extra table columns.
    assert "Great fit \\| but watch out for gaps in employment" in data_row
    assert "Great fit | but" not in data_row

    # Exactly 7 well-formed columns: splitting on the escaped-pipe-protected row
    # (only unescaped pipes act as separators) yields the 7 data fields plus the
    # two empty strings from the leading/trailing pipe.
    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 9  # "", rank, candidate, tier, rating, evidence, flags, notes, ""


def test_build_markdown_report_escapes_pipes_and_newlines_in_evidence():
    state = _sample_state()
    state["judge_results"] = {
        "strong": JudgeResult(
            candidate_id="strong", tier=Tier.STRONG_FIT, rating=9,
            evidence=[
                EvidenceClaim(claim="Led team", quote="Managed 5 | 10 person teams\nacross two years"),
                EvidenceClaim(claim="Shipped feature", quote="Delivered on time"),
            ],
        )
    }

    markdown = build_markdown_report(state)
    lines = markdown.splitlines()

    # Header + separator + exactly one data row: no extra rows from the embedded newline.
    assert len(lines) == 3
    data_row = lines[2]

    # No embedded newline leaked into the output.
    assert "\n" not in data_row

    # The literal pipe from the quote was escaped, and both evidence claims made it
    # into the same cell (joined, not split across rows/columns).
    assert "Led team: Managed 5 \\| 10 person teams across two years" in data_row
    assert "Shipped feature: Delivered on time" in data_row
    assert "Managed 5 | 10 person" not in data_row

    # Exactly 7 well-formed columns: only unescaped pipes act as separators.
    unescaped_split = data_row.replace("\\|", "").split("|")
    assert len(unescaped_split) == 9  # "", rank, candidate, tier, rating, evidence, flags, notes, ""


def test_build_markdown_report_shows_removed_count_for_flagged_candidate():
    state = _sample_state()
    state["hallucination_reports"] = {
        "strong": HallucinationReport(
            candidate_id="strong",
            unverified_quotes=["fabricated quote one", "fabricated quote two"],
        ),
    }

    markdown = build_markdown_report(state)

    assert "| 2 removed |" in markdown


def test_build_markdown_report_shows_dash_when_no_hallucination_report_present():
    state = _sample_state()
    state["hallucination_reports"] = {}

    markdown = build_markdown_report(state)

    assert "| — |" in markdown
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: the rewritten/new tests FAIL against the current 6-column table (header text mismatch, column-count mismatches, missing "removed"/"—" cells).

- [ ] **Step 3: Add the Hallucination Flags column**

In `src/evidencerank/report.py`, add this helper after `_format_evidence` (`report.py:43-49`):

```python
def _format_hallucination_flag(state: dict, candidate_id: str) -> str:
    """Render a candidate's hallucination-check outcome as one Markdown table cell."""
    report = state.get("hallucination_reports", {}).get(candidate_id)
    if report is None or not report.unverified_quotes:
        return "—"
    return f"{len(report.unverified_quotes)} removed"
```

Replace `build_markdown_report` (`report.py:52-64`) with:

```python
def build_markdown_report(state: dict) -> str:
    lines = [
        "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags | Calibration Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in sorted(state.get("calibrated_results", []), key=lambda r: r.final_rank):
        notes = _escape_table_cell(result.calibration_notes)
        evidence = _format_evidence(state, result.candidate_id)
        flags = _format_hallucination_flag(state, result.candidate_id)
        lines.append(
            f"| {result.final_rank} | {result.candidate_id} | {result.tier.value} "
            f"| {result.rating} | {evidence} | {flags} | {notes} |"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Update README**

In `README.md`, find the paragraph describing `report.md` (near `"report.md" (a ranked Markdown table)`) and add one sentence after it:

```
`report.md`'s table includes a "Hallucination Flags" column showing how many
evidence items were removed for that candidate before calibration (see
`report.json`'s `hallucination_reports` for the removed quotes themselves) — a
dash (`—`) means every quote verified.
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/evidencerank/report.py tests/test_report.py README.md
git commit -m "feat: surface hallucination-flag counts in report.md"
```

---

### Task 5: Capture per-stage timing in the pipeline and `report.json`

**Files:**
- Modify: `src/evidencerank/graph.py`
- Modify: `src/evidencerank/report.py`
- Test: `tests/test_graph.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PipelineState` gains `stage_timings: dict[str, float]` (seconds per stage, keyed by node name: `extract_profiles`, `prefilter`, `judge`, `hallucination_check`, `calibrate`). `build_json_report(state)` includes this under the `"stage_timings"` key (empty dict `{}` if absent from state, so old callers building from partial state don't break).

- [ ] **Step 1: Write the failing graph test**

Add to `tests/test_graph.py`, inside `test_graph_runs_extract_prefilter_judge_hallucination_calibrate` (right after the "Regression guard 4" block, before the function ends):

```python
    # Regression guard 5: every stage records a non-negative timing, keyed by
    # node name, so latency is visible in the eventual report.json.
    assert set(final_state["stage_timings"].keys()) == {
        "extract_profiles", "prefilter", "judge", "hallucination_check", "calibrate",
    }
    for seconds in final_state["stage_timings"].values():
        assert isinstance(seconds, float)
        assert seconds >= 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAILS with `KeyError: 'stage_timings'`.

- [ ] **Step 3: Add timing instrumentation to `graph.py`**

Add `import time` to the top of `src/evidencerank/graph.py` (alongside the existing `from typing import TypedDict` import).

Add `stage_timings: dict[str, float]` to the `PipelineState` TypedDict (`graph.py:21-31`), e.g. directly under the existing `hallucination_reports` field:

```python
class PipelineState(TypedDict, total=False):
    jd: JDRequirements
    raw_resumes: dict[str, str]
    profiles: dict[str, CandidateProfile]
    prefilter_results: dict[str, PrefilterResult]
    dropped: list[dict[str, str]]
    judge_results: dict[str, JudgeResult]
    calibrated_results: list[CalibratedResult]
    hallucination_reports: dict[str, HallucinationReport]
    stage_timings: dict[str, float]
    prefilter_threshold: float
    hallucination_threshold: float
```

Add a timing wrapper and use it in `build_graph()`. Insert the wrapper function right before `def build_graph():`:

```python
def _timed_node(name, node_fn):
    def wrapped(state: PipelineState) -> dict:
        start = time.perf_counter()
        result = dict(node_fn(state))
        elapsed = time.perf_counter() - start
        timings = dict(state.get("stage_timings", {}))
        timings[name] = elapsed
        result["stage_timings"] = timings
        return result
    return wrapped
```

Replace the `graph.add_node(...)` calls in `build_graph()` (`graph.py:92-96`) with:

```python
    graph.add_node("extract_profiles", _timed_node("extract_profiles", extract_profiles_node))
    graph.add_node("prefilter", _timed_node("prefilter", prefilter_node))
    graph.add_node("judge", _timed_node("judge", judge_node))
    graph.add_node("hallucination_check", _timed_node("hallucination_check", hallucination_check_node))
    graph.add_node("calibrate", _timed_node("calibrate", calibrate_node))
```

(The five node *functions* — `extract_profiles_node`, `prefilter_node`, `judge_node`, `hallucination_check_node`, `calibrate_node` — are unchanged; only how they're registered on the graph changes.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing report test**

Add to `tests/test_report.py`:

```python
def test_build_json_report_includes_stage_timings():
    state = _sample_state()
    state["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}

    report = build_json_report(state)

    assert report["stage_timings"] == {"extract_profiles": 1.5, "judge": 3.25}


def test_build_json_report_defaults_missing_stage_timings_to_empty_dict():
    report = build_json_report(_sample_state())

    assert report["stage_timings"] == {}
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAILS with `KeyError: 'stage_timings'`.

- [ ] **Step 7: Include `stage_timings` in `build_json_report`**

In `src/evidencerank/report.py`, add one line to the returned dict in `build_json_report` (`report.py:5-26`), right after the `"hallucination_reports"` entry:

```python
        "hallucination_reports": {
            candidate_id: report.model_dump()
            for candidate_id, report in state.get("hallucination_reports", {}).items()
        },
        "stage_timings": state.get("stage_timings", {}),
    }
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/evidencerank/graph.py src/evidencerank/report.py tests/test_graph.py tests/test_report.py
git commit -m "feat: record per-stage timing in report.json"
```

---

### Task 6: Render stage timings in the evaluation Markdown report

**Files:**
- Modify: `evaluation/report.py`
- Modify: `README.md` (one-line mention)
- Test: `evaluation/test_report.py`

**Interfaces:**
- Consumes: `stage_timings` key from a loaded `report.json` (produced by Task 5; may be absent for older report files).
- Produces: `build_eval_markdown_report(report_paths)` gains a `## Stage Timings` section, present only when the primary report's `stage_timings` is a non-empty dict.

- [ ] **Step 1: Write the failing tests**

Add to `evaluation/test_report.py`:

```python
def test_build_eval_markdown_report_includes_stage_timings_when_present(tmp_path, monkeypatch):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["stage_timings"] = {"extract_profiles": 1.5, "judge": 3.25}
    report_path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    markdown = build_eval_markdown_report([report_path])

    assert "## Stage Timings" in markdown
    assert "| extract_profiles | 1.500 |" in markdown
    assert "| judge | 3.250 |" in markdown


def test_build_eval_markdown_report_omits_stage_timings_when_absent(tmp_path, monkeypatch):
    from evaluation.report import build_eval_markdown_report

    report_path = tmp_path / "report.json"
    _write_calibrated_report(report_path, {"alice": 1})
    # NOTE: intentionally does NOT add "stage_timings" — simulates an older
    # report.json written before this key existed, to guard against a
    # KeyError regression.

    monkeypatch.setattr(groundedness_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(recruiter_alignment_metric, "measure", Mock(return_value=0.9))
    monkeypatch.setattr(evidence_relevancy_metric, "measure", Mock(return_value=0.9))

    markdown = build_eval_markdown_report([report_path])

    assert "## Stage Timings" not in markdown
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: `test_build_eval_markdown_report_includes_stage_timings_when_present` FAILS (no such section yet); `test_build_eval_markdown_report_omits_stage_timings_when_absent` already PASSES (nothing to regress yet, but confirms the baseline).

- [ ] **Step 3: Add the Stage Timings section**

In `evaluation/report.py`, in `build_eval_markdown_report` (`report.py:86-137`), add this block right after the `## GEval Metrics` table loop (after the `for name in (...)` loop, `report.py:118-123`) and before the `if len(report_paths) >= 2:` rank-stability block:

```python
    stage_timings = data.get("stage_timings") or {}
    if stage_timings:
        lines += [
            "",
            "## Stage Timings",
            "",
            "| Stage | Seconds |",
            "|---|---|",
        ]
        for stage_name, seconds in stage_timings.items():
            lines.append(f"| {stage_name} | {seconds:.3f} |")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest evaluation/test_report.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Update README**

In `README.md`, in the "Evaluation metric report" section describing what `eval_report.md` contains, add:

```
When the underlying `report.json` includes per-stage timing (`stage_timings`,
added by the production pipeline), the report also includes a "Stage Timings"
table showing wall-clock seconds per stage — absent for older `report.json`
files that predate this field.
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluation/report.py evaluation/test_report.py README.md
git commit -m "feat: render per-stage timing in the evaluation Markdown report"
```

---

### Task 7: Re-run the pipeline and validate the fixes against the original bad run

**Files:**
- None modified — this is a validation task using the existing CLI.

**Interfaces:**
- Consumes: everything built in Tasks 1–6.
- Produces: a fresh `report.json` / `report.md` / `eval_report.md` for comparison against the pre-fix numbers recorded in this plan's Goal section.

- [ ] **Step 1: Confirm Ollama is running with the required models**

Run: `ollama list`
Expected: `qwen2.5:7b-instruct` and `qwen2.5:14b-instruct` are present. If `ollama serve` isn't running, start it first (per README setup).

- [ ] **Step 2: Re-run the pipeline against the existing resumes**

Run:
```bash
uv run evidencerank \
  --jd machine_learning_engineer.txt \
  --resumes-dir resumes \
  --out-json report.json \
  --out-md report.md \
  --with-eval-report
```
Expected: exit code 0; `report.json`, `report.md`, `eval_report.md` all written; stdout shows `Running stage: <name>` for all 5 stages.

- [ ] **Step 3: Compare against the pre-fix baseline**

Open the new `eval_report.md` and compare against this plan's stated baseline (Groundedness 7.7% pass / RecruiterAlignment 11.5% pass / EvidenceRelevancy 0% pass / 14/26 hallucination-flagged). Expected direction: Groundedness pass rate should rise sharply (evidence is now pre-filtered before calibration, per Task 3's noted side effect in the spec); RecruiterAlignment and EvidenceRelevancy should also improve since evidence is both correctly scoped (Task 1) and claim-relevant (Task 1); hallucination-flagged count should drop (Tasks 1 and 2 both reduce false and true positives). Note the new "Stage Timings" section's numbers as the latency baseline going forward.

- [ ] **Step 4: Record the outcome**

No commit needed for this task (no source changes) — report the before/after numbers back in the conversation so the user can decide whether further iteration (e.g. the model-swap experiment noted as out-of-scope in the spec) is warranted.
