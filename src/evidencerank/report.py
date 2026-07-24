import json
from pathlib import Path


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


def build_markdown_report(state: dict) -> str:
    lines = ["| Rank | Candidate | Tier | Rating | Calibration Notes |", "|---|---|---|---|---|"]
    for result in sorted(state.get("calibrated_results", []), key=lambda r: r.final_rank):
        notes = _escape_table_cell(result.calibration_notes)
        lines.append(
            f"| {result.final_rank} | {result.candidate_id} | {result.tier.value} "
            f"| {result.rating} | {notes} |"
        )
    return "\n".join(lines)


def write_markdown_report(state: dict, path: str | Path) -> None:
    Path(path).write_text(build_markdown_report(state), encoding="utf-8")
