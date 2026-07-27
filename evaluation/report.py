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
from evaluation.rank_stability import rank_stability


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


def build_eval_markdown_report(report_paths: list[str | Path]) -> str:
    primary = report_paths[0]
    data = json.loads(Path(primary).read_text(encoding="utf-8"))
    stats = compute_pipeline_stats(primary)
    geval = compute_geval_scores(primary)

    lines = [
        "# Evaluation Metric Report",
        "",
        f"**JD:** {data['jd']['title']}",
        f"**Primary report:** {primary}",
    ]
    if len(report_paths) > 1:
        extra = ", ".join(str(path) for path in report_paths[1:])
        lines.append(f"**Additional reports (rank stability):** {extra}")
    lines += [
        "",
        "## Pipeline Stats",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total candidates | {stats['total_candidates']} |",
        f"| Passed pre-filter | {stats['passed_prefilter']} |",
        f"| Dropped by pre-filter | {stats['dropped_prefilter']} |",
        f"| Evaluated by Judge | {stats['evaluated_by_judge']} |",
        f"| Hallucination-flagged candidates | {stats['hallucination_flagged']} |",
        "",
        "## GEval Metrics",
        "",
        "| Metric | n | Mean | Std Dev | Pass Rate |",
        "|---|---|---|---|---|",
    ]
    for name in ("Groundedness", "RecruiterAlignment", "EvidenceRelevancy"):
        m = geval[name]
        mean_str = f"{m['mean']:.3f}" if m["mean"] is not None else "N/A"
        std_str = f"{m['std']:.3f}" if m["std"] is not None else "N/A"
        pass_str = f"{m['pass_rate']:.1%}" if m["pass_rate"] is not None else "N/A"
        lines.append(f"| {name} | {m['n']} | {mean_str} | {std_str} | {pass_str} |")

    lines.append(
        "\n_Note: since the hallucination checker now strips unverified evidence before "
        "calibration, Groundedness is expected to trend toward ~100% by construction "
        "(remaining quotes already passed fuzzy verification) — RecruiterAlignment and "
        "EvidenceRelevancy are the informative signals for judge quality._"
    )

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

    if len(report_paths) >= 2:
        stability = rank_stability([str(path) for path in report_paths])
        lines += [
            "",
            "## Rank Stability",
            "",
            "| Runs | Mean Spearman | Mean Kendall Tau |",
            "|---|---|---|",
            f"| {stability['n_runs']} | {stability['mean_spearman']:.3f} "
            f"| {stability['mean_kendall_tau']:.3f} |",
        ]

    return "\n".join(lines)


def write_eval_markdown_report(report_paths: list[str | Path], path: str | Path) -> None:
    Path(path).write_text(build_eval_markdown_report(report_paths), encoding="utf-8")
