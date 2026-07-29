import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def load_rank_map(report_path: str | Path) -> dict[str, int]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {entry["candidate_id"]: entry["final_rank"] for entry in data["calibrated_results"]}


def rank_stability(report_paths: list[str]) -> dict:
    rank_maps = [load_rank_map(path) for path in report_paths]
    candidate_ids = sorted(set(rank_maps[0]).intersection(*rank_maps[1:]))
    if len(candidate_ids) < 2:
        raise ValueError(
            f"Only {len(candidate_ids)} candidate(s) were shortlisted in every run "
            f"({len(report_paths)} runs given) - rank correlation needs at least 2 "
            "candidates common to all runs to be meaningful."
        )

    spearman_scores = []
    kendall_scores = []
    for map_a, map_b in combinations(rank_maps, 2):
        ranks_a = [map_a[candidate_id] for candidate_id in candidate_ids]
        ranks_b = [map_b[candidate_id] for candidate_id in candidate_ids]
        spearman_scores.append(spearmanr(ranks_a, ranks_b).correlation)
        kendall_scores.append(kendalltau(ranks_a, ranks_b).correlation)

    return {
        "mean_spearman": sum(spearman_scores) / len(spearman_scores),
        "mean_kendall_tau": sum(kendall_scores) / len(kendall_scores),
        "n_runs": len(report_paths),
    }


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _format_evidence(judge_result: dict | None) -> str:
    if judge_result is None:
        return ""
    joined = "; ".join(f"{claim['claim']}: {claim['quote']}" for claim in judge_result["evidence"])
    return _escape_table_cell(joined)


def _format_hallucination_flag(hallucination_report: dict | None) -> str:
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


def _compute_mean_evidence_relevancy(data: dict) -> float:
    jd = data["jd"]
    reference_texts = [
        *jd.get("required_skills", []),
        *jd.get("nice_to_have_skills", []),
        *jd.get("responsibilities", []),
    ]
    candidates_with_evidence = [
        judge_result["evidence"]
        for judge_result in data.get("judge_results", {}).values()
        if judge_result.get("evidence")
    ]
    if not reference_texts or not candidates_with_evidence:
        return 0.0

    embedder = _get_embedder()
    all_texts = list(reference_texts)
    claim_spans = []
    for evidence in candidates_with_evidence:
        start = len(all_texts)
        all_texts.extend(claim["claim"] for claim in evidence)
        claim_spans.append((start, len(all_texts)))

    vectors = embedder.encode(all_texts)
    n_reference = len(reference_texts)
    reference_vecs = vectors[:n_reference]

    candidate_scores = []
    for start, end in claim_spans:
        claim_scores = [
            max(_cosine_similarity(claim_vec, ref_vec) for ref_vec in reference_vecs)
            for claim_vec in vectors[start:end]
        ]
        candidate_scores.append(sum(claim_scores) / len(claim_scores))

    return sum(candidate_scores) / len(candidate_scores)


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
        "mean_evidence_relevancy": _compute_mean_evidence_relevancy(data),
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
        f"| Evidence Relevancy | {stats['mean_evidence_relevancy']:.3f} |",
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
