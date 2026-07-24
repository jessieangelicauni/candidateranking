# EvidenceRank

Evidence-grounded multi-agent LLM pipeline for ranking IT job candidates against a job
description, using local Ollama models orchestrated with LangGraph. See
`docs/superpowers/specs/2026-07-24-evidencerank-design.md` for the full design.

## Setup

1. Install [Ollama](https://ollama.com) and start the server: `ollama serve`
2. Pull the default models:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull qwen2.5:14b-instruct
   ```
3. Install project dependencies: `uv sync`
4. Run the test suite: `uv run pytest`

## Running the pipeline

```bash
uv run evidencerank \
  --jd machine_learning_engineer.txt \
  --resumes-dir . \
  --out-json report.json \
  --out-md report.md
```

This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results) and `report.md` (a ranked Markdown table).

## Model configuration

Override any stage's model with an environment variable:

```bash
export EVIDENCERANK_MODEL_JUDGE=qwen2.5:32b-instruct
```

Valid stages: `jd_parser`, `cv_extractor`, `judge`, `calibrator`.

## Research evaluation harness

The `evaluation/` package is separate from the production pipeline (`src/evidencerank/`):

- `evaluation/metrics.py` — DeepEval `GEval` metrics (Groundedness, RecruiterAlignment,
  EvidenceRelevancy) to run against pipeline output.
- `evaluation/rank_stability.py` — computes Spearman/Kendall-tau rank correlation across
  repeated runs on the same input, to report LLM judgment consistency.

To measure rank stability, run the pipeline N times on the same JD/resumes with
different `--out-json` paths, then:

```python
from evaluation.rank_stability import rank_stability
print(rank_stability(["run1.json", "run2.json", "run3.json"]))
```
