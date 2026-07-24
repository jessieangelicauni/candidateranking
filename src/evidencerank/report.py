import json
from pathlib import Path


def build_json_report(state: dict) -> dict:
    return {
        "jd": state["jd"].model_dump(),
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


def build_markdown_report(state: dict) -> str:
    lines = ["| Rank | Candidate | Tier | Rating | Calibration Notes |", "|---|---|---|---|---|"]
    for result in sorted(state.get("calibrated_results", []), key=lambda r: r.final_rank):
        lines.append(
            f"| {result.final_rank} | {result.candidate_id} | {result.tier.value} "
            f"| {result.rating} | {result.calibration_notes} |"
        )
    return "\n".join(lines)


def write_markdown_report(state: dict, path: str | Path) -> None:
    Path(path).write_text(build_markdown_report(state), encoding="utf-8")
