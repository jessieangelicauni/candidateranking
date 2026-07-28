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
