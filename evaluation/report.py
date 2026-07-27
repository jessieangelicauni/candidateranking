import json
import statistics
from pathlib import Path

from deepeval.test_case import LLMTestCase

from evaluation.metrics import (
    build_test_case,
    evidence_relevancy_metric,
    groundedness_metric,
    recruiter_alignment_metric,
)


def compute_pipeline_stats(report_path: str | Path) -> dict:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))

    total_candidates = len(data["profiles"])
    dropped_prefilter = len(data["dropped"])
    hallucination_flagged = sum(
        1
        for report in data["hallucination_reports"].values()
        if report["unverified_quotes"]
    )

    return {
        "total_candidates": total_candidates,
        "passed_prefilter": total_candidates - dropped_prefilter,
        "dropped_prefilter": dropped_prefilter,
        "evaluated_by_judge": len(data["judge_results"]),
        "hallucination_flagged": hallucination_flagged,
    }


_GEVAL_METRICS = [groundedness_metric, recruiter_alignment_metric, evidence_relevancy_metric]


def _format_jd_text(jd: dict) -> str:
    return "\n".join(
        [
            f"Title: {jd['title']}",
            f"Required skills: {', '.join(jd['required_skills'])}",
            f"Nice-to-have skills: {', '.join(jd['nice_to_have_skills'])}",
            f"Minimum experience years: {jd['min_experience_years']}",
            f"Education: {jd['education']}",
            f"Responsibilities: {', '.join(jd['responsibilities'])}",
        ]
    )


def _format_judge_result_text(judge_result: dict) -> str:
    lines = [f"Tier: {judge_result['tier']}", f"Rating: {judge_result['rating']}"]
    for claim in judge_result["evidence"]:
        lines.append(f'- {claim["claim"]}: "{claim["quote"]}"')
    return "\n".join(lines)


def _aggregate_scores(scores: list[float], threshold: float) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "pass_rate": None}
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if n >= 2 else None
    pass_rate = sum(1 for score in scores if score >= threshold) / n
    return {"n": n, "mean": mean, "std": std, "pass_rate": pass_rate}


def compute_geval_scores(report_path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    jd_text = _format_jd_text(data["jd"])

    test_cases: list[LLMTestCase] = []
    for candidate_id, judge_result in data["judge_results"].items():
        cv_text = data["profiles"][candidate_id]["raw_cv_text"]
        judge_text = _format_judge_result_text(judge_result)
        test_cases.append(build_test_case(jd_text, judge_text, cv_text))

    results: dict[str, dict] = {}
    for metric in _GEVAL_METRICS:
        scores = [metric.measure(test_case) for test_case in test_cases]
        results[metric.name] = _aggregate_scores(scores, metric.threshold)
    return results
