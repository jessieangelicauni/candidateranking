from evidencerank.models import JudgeResult


def select_shortlist(
    judge_results: list[JudgeResult], size: int = 10
) -> tuple[list[JudgeResult], list[dict[str, str]]]:
    """
    Select the top candidates by rating, handling ties at the boundary.

    If the pool is at or under the size limit, all candidates are selected.
    Otherwise, the top `size` candidates by rating are selected, plus any
    candidates tied at the cutoff rating.

    Args:
        judge_results: List of JudgeResult objects from the judge agent.
        size: Maximum number of candidates to shortlist (default: 10).

    Returns:
        A tuple of (shortlisted, not_shortlisted) where:
        - shortlisted: List of JudgeResult objects selected for shortlist.
        - not_shortlisted: List of dicts with candidate_id and reason for exclusion.
    """
    if len(judge_results) <= size:
        return list(judge_results), []

    ranked = sorted(judge_results, key=lambda result: result.rating, reverse=True)
    cutoff_rating = ranked[size - 1].rating

    shortlisted = [result for result in ranked if result.rating >= cutoff_rating]
    not_shortlisted = [
        {"candidate_id": result.candidate_id, "reason": f"ranked outside judge's top {size} by rating"}
        for result in ranked
        if result.rating < cutoff_rating
    ]
    return shortlisted, not_shortlisted
