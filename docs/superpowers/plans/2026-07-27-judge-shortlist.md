# Judge Shortlist Before Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound `calibrate_pool()`'s input to the judge's top 10 candidates by rating (ties at the boundary included), instead of the entire judged pool, closing a context-window scalability risk and matching what `CALIBRATOR_PROMPT` already assumes ("a shortlisted candidate pool").

**Architecture:** A new pure selection function (`select_shortlist`) in a new `agents/shortlist.py` module, wired into the LangGraph pipeline as a new `shortlist` node between `judge` and `calibrate`. `calibrate_node` filters `judge_results` down to the shortlist before calling `calibrate_pool`. `report.json` gains a `not_shortlisted` list (same shape as the existing `dropped` list); `judge_results` and `report.md`'s rendering logic are otherwise unchanged.

**Tech Stack:** Python, pytest, LangGraph (`StateGraph`), Pydantic models already defined in `src/evidencerank/models.py`.

## Global Constraints

- Shortlist size is a hardcoded constant (`10`), not a CLI flag — no new CLI surface in this build.
- Ties at the 10th-place boundary are all included, even if that makes the shortlist larger than 10 — never arbitrarily cut a tied candidate.
- `judge_results` in `report.json` must still contain every judged candidate, shortlisted or not — only `calibrated_results` reflects the shortlist.
- `not_shortlisted` entries use the exact reason string: `"ranked outside judge's top 10 by rating"`.
- Follow the existing codebase pattern: pure logic lives in a small `agents/*.py` module (see `agents/prefilter.py`), the LangGraph node wrapper lives in `graph.py`.

---

### Task 1: `select_shortlist()` pure function

**Files:**
- Create: `src/evidencerank/agents/shortlist.py`
- Test: `tests/agents/test_shortlist.py`

**Interfaces:**
- Consumes: `evidencerank.models.JudgeResult` (existing; has `candidate_id: str`, `rating: int`, `tier: Tier`, `evidence: list[EvidenceClaim]`).
- Produces: `select_shortlist(judge_results: list[JudgeResult], size: int = 10) -> tuple[list[JudgeResult], list[dict[str, str]]]` — used by Task 2's `shortlist_node`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_shortlist.py`:

```python
from evidencerank.agents.shortlist import select_shortlist
from evidencerank.models import EvidenceClaim, JudgeResult, Tier


def _judge_result(candidate_id: str, rating: int) -> JudgeResult:
    return JudgeResult(
        candidate_id=candidate_id,
        tier=Tier.MODERATE_FIT,
        rating=rating,
        evidence=[EvidenceClaim(claim="c", quote="q")],
    )


def test_select_shortlist_keeps_everyone_when_pool_is_at_or_under_size():
    judge_results = [_judge_result(f"c{i}", rating=5) for i in range(10)]

    shortlisted, not_shortlisted = select_shortlist(judge_results)

    assert len(shortlisted) == 10
    assert not_shortlisted == []


def test_select_shortlist_keeps_top_10_by_rating_with_no_ties():
    judge_results = [_judge_result(f"c{i}", rating=20 - i) for i in range(12)]

    shortlisted, not_shortlisted = select_shortlist(judge_results)

    assert {r.candidate_id for r in shortlisted} == {f"c{i}" for i in range(10)}
    assert {entry["candidate_id"] for entry in not_shortlisted} == {"c10", "c11"}
    assert not_shortlisted[0]["reason"] == "ranked outside judge's top 10 by rating"


def test_select_shortlist_includes_ties_at_the_boundary():
    # 9 candidates at rating=9, then 3 candidates at rating=7. The 10th-ranked
    # slot lands on a rating=7 candidate, but there are 3 of them tied - all
    # 3 must be included, making the shortlist 12 long, not 10.
    judge_results = [_judge_result(f"c{i}", rating=9) for i in range(9)] + [
        _judge_result(f"t{i}", rating=7) for i in range(3)
    ]

    shortlisted, not_shortlisted = select_shortlist(judge_results)

    assert len(shortlisted) == 12
    assert not_shortlisted == []


def test_select_shortlist_handles_empty_input():
    assert select_shortlist([]) == ([], [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_shortlist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evidencerank.agents.shortlist'`

- [ ] **Step 3: Write minimal implementation**

Create `src/evidencerank/agents/shortlist.py`:

```python
from evidencerank.models import JudgeResult


def select_shortlist(
    judge_results: list[JudgeResult], size: int = 10
) -> tuple[list[JudgeResult], list[dict[str, str]]]:
    if len(judge_results) <= size:
        return list(judge_results), []

    ranked = sorted(judge_results, key=lambda result: result.rating, reverse=True)
    cutoff_rating = ranked[size - 1].rating

    shortlisted = [result for result in ranked if result.rating >= cutoff_rating]
    not_shortlisted = [
        {"candidate_id": result.candidate_id, "reason": "ranked outside judge's top 10 by rating"}
        for result in ranked
        if result.rating < cutoff_rating
    ]
    return shortlisted, not_shortlisted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_shortlist.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/agents/shortlist.py tests/agents/test_shortlist.py
git commit -m "feat: add select_shortlist for top-10-by-rating candidate selection"
```

---

### Task 2: Wire `shortlist` into the pipeline graph

**Files:**
- Modify: `src/evidencerank/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `select_shortlist` from Task 1 (`evidencerank.agents.shortlist`).
- Produces: `PipelineState["shortlisted_ids"]: set[str]`, `PipelineState["not_shortlisted"]: list[dict[str, str]]` — used by Task 3's `report.py` changes for `not_shortlisted`, and already consumed within this task by the modified `calibrate_node`.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_graph.py` (append after the existing
`test_graph_runs_extract_prefilter_judge_calibrate_hallucination` test):

```python
def test_graph_shortlists_top_10_by_rating_before_calibrating(monkeypatch):
    jd = JDRequirements(title="ML Engineer", required_skills=["Python"])
    raw_resumes = {f"c{i}": f"Python resume {i}" for i in range(12)}

    def fake_extract_cv(candidate_id, raw_text):
        return CandidateProfile(
            candidate_id=candidate_id,
            raw_cv_text=raw_text,
            contact=ContactInfo(name=candidate_id),
            skills=["Python"],
        )

    def fake_prefilter_candidate(candidate_id, jd_required_skills, candidate_skills, threshold):
        return PrefilterResult(candidate_id=candidate_id, similarity=0.9, passed=True)

    # Rating descends with candidate index (c0 highest at 20, c11 lowest at
    # 9) with no ties, so the top 10 by rating is exactly c0..c9.
    def fake_judge_candidate(jd_requirements, profile):
        index = int(profile.candidate_id[1:])
        return JudgeResult(
            candidate_id=profile.candidate_id,
            tier=Tier.STRONG_FIT,
            rating=20 - index,
            evidence=[EvidenceClaim(claim="Strong fit", quote="Python")],
        )

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

    monkeypatch.setattr("evidencerank.graph.cached_extract_cv", fake_extract_cv)
    monkeypatch.setattr("evidencerank.graph.prefilter_candidate", fake_prefilter_candidate)
    monkeypatch.setattr("evidencerank.graph.judge_candidate", fake_judge_candidate)
    monkeypatch.setattr("evidencerank.graph.calibrate_pool", fake_calibrate_pool)
    monkeypatch.setattr("evidencerank.graph.check_evidence", fake_check_evidence)

    graph = build_graph()
    final_state = graph.invoke({"jd": jd, "raw_resumes": raw_resumes})

    expected_shortlist = {f"c{i}" for i in range(10)}
    expected_cut = {f"c{i}" for i in range(10, 12)}

    assert len(calibrate_calls) == 1
    assert set(calibrate_calls[0]) == expected_shortlist
    assert final_state["shortlisted_ids"] == expected_shortlist
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

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py::test_graph_shortlists_top_10_by_rating_before_calibrating -v`
Expected: FAIL — `calibrate_calls[0]` contains all 12 candidate IDs, not just 10, so
`assert set(calibrate_calls[0]) == expected_shortlist` fails (and `final_state["shortlisted_ids"]`
raises `KeyError` since that state key doesn't exist yet).

- [ ] **Step 3: Implement the shortlist node and wire it in**

In `src/evidencerank/graph.py`, add the import (alongside the existing `agents` imports):

```python
from evidencerank.agents.shortlist import select_shortlist
```

Add two keys to `PipelineState` (after `judge_results: dict[str, JudgeResult]`):

```python
class PipelineState(TypedDict, total=False):
    jd: JDRequirements
    raw_resumes: dict[str, str]
    profiles: dict[str, CandidateProfile]
    prefilter_results: dict[str, PrefilterResult]
    dropped: list[dict[str, str]]
    judge_results: dict[str, JudgeResult]
    shortlisted_ids: set[str]
    not_shortlisted: list[dict[str, str]]
    calibrated_results: list[CalibratedResult]
    hallucination_reports: dict[str, HallucinationReport]
    prefilter_threshold: float
    hallucination_threshold: float
```

Add the new node function (after `judge_node`, before `calibrate_node`):

```python
def shortlist_node(state: PipelineState) -> dict:
    click.echo("Running stage: shortlist")
    shortlisted, not_shortlisted = select_shortlist(list(state["judge_results"].values()))
    return {
        "shortlisted_ids": {result.candidate_id for result in shortlisted},
        "not_shortlisted": not_shortlisted,
    }
```

Replace `calibrate_node` with:

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

Replace `build_graph()` with:

```python
def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract_profiles", extract_profiles_node)
    graph.add_node("prefilter", prefilter_node)
    graph.add_node("judge", judge_node)
    graph.add_node("shortlist", shortlist_node)
    graph.add_node("calibrate", calibrate_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    graph.set_entry_point("extract_profiles")
    graph.add_edge("extract_profiles", "prefilter")
    graph.add_edge("prefilter", "judge")
    graph.add_edge("judge", "shortlist")
    graph.add_edge("shortlist", "calibrate")
    graph.add_edge("calibrate", "hallucination_check")
    graph.add_edge("hallucination_check", END)

    return graph.compile()
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_graph.py::test_graph_shortlists_top_10_by_rating_before_calibrating -v`
Expected: PASS

- [ ] **Step 5: Run the full graph test file to check for regressions**

Run: `uv run pytest tests/test_graph.py -v`
Expected: both tests PASS — `test_graph_runs_extract_prefilter_judge_calibrate_hallucination` only
has 2 candidates (well under size 10), so the whole pool is still shortlisted unchanged and that
test's existing assertions hold without modification.

- [ ] **Step 6: Commit**

```bash
git add src/evidencerank/graph.py tests/test_graph.py
git commit -m "feat: wire shortlist stage between judge and calibrate"
```

---

### Task 3: Surface `not_shortlisted` in `report.json`

**Files:**
- Modify: `src/evidencerank/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `state["not_shortlisted"]` (produced by Task 2's `shortlist_node`, shape `list[dict[str, str]]`).
- Produces: `build_json_report(state)["not_shortlisted"]` — a new top-level key in the JSON report, no change to any existing key's shape.

- [ ] **Step 1: Write the failing tests**

In `tests/test_report.py`, modify `_sample_state()` to add a `not_shortlisted` key (insert
right after the existing `"dropped"` line):

```python
def _sample_state():
    return {
        "jd": JDRequirements(title="ML Engineer", required_skills=["Python"]),
        "profiles": {
            "strong": CandidateProfile(
                candidate_id="strong",
                raw_cv_text="Jane Doe, 5 years Python",
                contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
                skills=["Python"],
            )
        },
        "prefilter_results": {
            "strong": PrefilterResult(candidate_id="strong", similarity=0.9, passed=True),
            "weak": PrefilterResult(candidate_id="weak", similarity=0.1, passed=False),
        },
        "dropped": [{"candidate_id": "weak", "reason": "pre-filter: no relevant skill overlap"}],
        "not_shortlisted": [
            {"candidate_id": "cut", "reason": "ranked outside judge's top 10 by rating"}
        ],
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
```

Modify `test_build_json_report_contains_all_sections` to add one assertion (append at the end
of the existing assertion block):

```python
def test_build_json_report_contains_all_sections():
    report = build_json_report(_sample_state())

    assert report["jd"]["title"] == "ML Engineer"
    assert report["profiles"]["strong"]["candidate_id"] == "strong"
    assert report["profiles"]["strong"]["raw_cv_text"] == "Jane Doe, 5 years Python"
    assert report["profiles"]["strong"]["contact"]["name"] == "Jane Doe"
    assert report["prefilter_results"]["strong"]["similarity"] == 0.9
    assert report["prefilter_results"]["strong"]["passed"] is True
    assert report["prefilter_results"]["weak"]["passed"] is False
    assert report["dropped"][0]["candidate_id"] == "weak"
    assert report["not_shortlisted"][0]["candidate_id"] == "cut"
    assert report["judge_results"]["strong"]["rating"] == 9
    assert report["calibrated_results"][0]["final_rank"] == 1
    assert report["hallucination_reports"]["strong"]["unverified_quotes"] == []
```

Modify `test_build_json_report_defaults_missing_stages_to_empty` to add one assertion:

```python
def test_build_json_report_defaults_missing_stages_to_empty():
    minimal_state = {"jd": JDRequirements(title="ML Engineer", required_skills=["Python"])}

    report = build_json_report(minimal_state)

    assert report["profiles"] == {}
    assert report["prefilter_results"] == {}
    assert report["not_shortlisted"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `test_build_json_report_contains_all_sections` fails with `KeyError:
'not_shortlisted'`, and `test_build_json_report_defaults_missing_stages_to_empty` fails the
same way.

- [ ] **Step 3: Implement the change**

In `src/evidencerank/report.py`, modify `build_json_report` (add the `"not_shortlisted"` line
right after `"dropped"`):

```python
def build_json_report(state: dict) -> dict:
    return {
        "jd": state["jd"].model_dump(),
        "profiles": {
            candidate_id: profile.model_dump()
            for candidate_id, profile in state.get("profiles", {}).items()
        },
        "prefilter_results": {
            candidate_id: result.model_dump()
            for candidate_id, result in state.get("prefilter_results", {}).items()
        },
        "dropped": state.get("dropped", []),
        "not_shortlisted": state.get("not_shortlisted", []),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evidencerank/report.py tests/test_report.py
git commit -m "feat: surface not_shortlisted candidates in report.json"
```

---

### Task 4: Update README

**Files:**
- Modify: `README.md`

No tests — documentation only, depends on Tasks 2 and 3 being complete so the description is
accurate.

- [ ] **Step 1: Update the stage list and add shortlist behavior documentation**

In `README.md`, find this paragraph:

```
The pipeline prints a `Running stage: <name>` line to stdout as each of the 5 stages
(`extract_profiles`, `prefilter`, `judge`, `calibrate`, `hallucination_check`) starts,
so you can follow progress on longer runs.
```

Replace it with:

```
The pipeline prints a `Running stage: <name>` line to stdout as each of the 6 stages
(`extract_profiles`, `prefilter`, `judge`, `shortlist`, `calibrate`, `hallucination_check`)
starts, so you can follow progress on longer runs.

Only the judge's top 10 candidates by rating proceed to the `calibrate` stage (ties at the
10th-place boundary are all kept, so the shortlist can be slightly larger than 10 candidates).
Everyone judged is still fully recorded in `report.json`'s `profiles` and `judge_results`;
candidates cut before calibration are additionally listed in `report.json`'s `not_shortlisted`
with the reason `"ranked outside judge's top 10 by rating"`. `report.md`'s ranked table only
reflects the shortlist that reached calibration.
```

- [ ] **Step 2: Verify the change reads correctly**

Run: `grep -n "shortlist" README.md`
Expected: the new stage name and paragraph appear.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the judge shortlist stage in README"
```

---

### Task 5: Full regression run

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest -q`
Expected: all tests pass (75 existing + 4 new in Task 1 + 1 new in Task 2 = 80 total), no
warnings beyond the pre-existing `deepeval` `DeprecationWarning`s.

- [ ] **Step 2: If anything fails, fix forward**

Do not skip or delete a failing test to make the suite green — if Task 5 surfaces a failure,
return to the task that introduced it, diagnose with `superpowers:systematic-debugging`, and fix
the root cause before considering this plan complete.
