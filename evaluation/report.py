import json
from pathlib import Path

from evaluation.rank_stability import rank_stability


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
        "judge_results": {
            candidate_id: result.model_dump()
            for candidate_id, result in state.get("judge_results", {}).items()
        },
        "calibrated_results": [result.model_dump() for result in state.get("calibrated_results", [])],
        "hallucination_reports": {
            candidate_id: report.model_dump()
            for candidate_id, report in state.get("hallucination_reports", {}).items()
        },
        "stage_timings": state.get("stage_timings", {}),
    }


def write_json_report(state: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_json_report(state), indent=2), encoding="utf-8")


def _escape_table_cell(text: str) -> str:
    """Make free-text safe to interpolate into a Markdown table cell.

    Escapes literal pipe characters (which would otherwise be parsed as
    column separators) and collapses newlines to spaces (which would
    otherwise break the row onto multiple lines / rows).
    """
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _format_evidence(judge_result: dict | None) -> str:
    """Render a candidate's Judge evidence claims as one Markdown table cell."""
    if judge_result is None:
        return ""
    joined = "; ".join(f"{claim['claim']}: {claim['quote']}" for claim in judge_result["evidence"])
    return _escape_table_cell(joined)


def _format_hallucination_flag(hallucination_report: dict | None) -> str:
    """Render a candidate's hallucination-check outcome as one Markdown table cell."""
    if hallucination_report is None or not hallucination_report["unverified_quotes"]:
        return "—"
    return f"{len(hallucination_report['unverified_quotes'])} removed"


def _build_ranking_table(data: dict) -> str:
    lines = [
        "| Rank | Candidate | Tier | Rating | Key Evidence | Hallucination Flags | Calibration Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    judge_results = data.get("judge_results", {})
    hallucination_reports = data.get("hallucination_reports", {})
    for result in sorted(data.get("calibrated_results", []), key=lambda r: r["final_rank"]):
        candidate_id = result["candidate_id"]
        notes = _escape_table_cell(result["calibration_notes"])
        evidence = _format_evidence(judge_results.get(candidate_id))
        flags = _format_hallucination_flag(hallucination_reports.get(candidate_id))
        lines.append(
            f"| {result['final_rank']} | {candidate_id} | {result['tier']} "
            f"| {result['rating']} | {evidence} | {flags} | {notes} |"
        )
    return "\n".join(lines)


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


def build_markdown_report(report_paths: list[str | Path]) -> str:
    primary = report_paths[0]
    data = json.loads(Path(primary).read_text(encoding="utf-8"))
    stats = compute_pipeline_stats(primary)

    lines = [
        "# Candidate Ranking Report",
        "",
        f"**JD:** {data['jd']['title']}",
        f"**Primary report:** {primary}",
    ]
    if len(report_paths) > 1:
        extra = ", ".join(str(path) for path in report_paths[1:])
        lines.append(f"**Additional reports (rank stability):** {extra}")
    lines += [
        "",
        "## Rankings",
        "",
        _build_ranking_table(data),
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


def write_markdown_report(report_paths: list[str | Path], path: str | Path) -> None:
    Path(path).write_text(build_markdown_report(report_paths), encoding="utf-8")
