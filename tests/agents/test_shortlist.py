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
    # Create 12 candidates: 8 at rating 10, 2 at rating 9, 2 at rating 1.
    # Top 10 are the 8 at rating 10 plus the 2 at rating 9.
    # The 2 at rating 1 are excluded.
    judge_results = [_judge_result(f"c{i}", rating=10) for i in range(8)] + \
                    [_judge_result(f"c{i}", rating=9) for i in range(8, 10)] + \
                    [_judge_result(f"c{i}", rating=1) for i in range(10, 12)]

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
