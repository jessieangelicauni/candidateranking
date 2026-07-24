import json
from itertools import combinations
from pathlib import Path

from scipy.stats import kendalltau, spearmanr


def load_rank_map(report_path: str | Path) -> dict[str, int]:
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {entry["candidate_id"]: entry["final_rank"] for entry in data["calibrated_results"]}


def rank_stability(report_paths: list[str]) -> dict:
    rank_maps = [load_rank_map(path) for path in report_paths]
    candidate_ids = sorted(rank_maps[0].keys())

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
