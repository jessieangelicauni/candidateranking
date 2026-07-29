import json
from pathlib import Path

from evaluation.rank_stability import rank_stability


def compute_pipeline_stats(report_path: str | Path) -> dict:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))

    total_candidates = len(data["profiles"])
    dropped_prefilter = len(data["dropped"])
    evaluated_by_judge = len(data["judge_results"])
    hallucination_flagged = sum(
        1
        for report in data["hallucination_reports"].values()
        if report["unverified_quotes"]
    )
    hallucination_rate = hallucination_flagged / evaluated_by_judge if evaluated_by_judge else 0.0

    return {
        "total_candidates": total_candidates,
        "passed_prefilter": total_candidates - dropped_prefilter,
        "dropped_prefilter": dropped_prefilter,
        "evaluated_by_judge": evaluated_by_judge,
        "hallucination_rate": hallucination_rate,
    }


def build_eval_markdown_report(report_paths: list[str | Path]) -> str:
    primary = report_paths[0]
    data = json.loads(Path(primary).read_text(encoding="utf-8"))
    stats = compute_pipeline_stats(primary)

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
        f"| Hallucination Rate | {stats['hallucination_rate']:.1%} |",
    ]

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
