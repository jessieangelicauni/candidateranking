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
4. Set required environment variables (see [Environment variables](#environment-variables) below)
5. Run the test suite: `uv run pytest`

## Running the pipeline

Put all candidate resumes (PDF) in one folder — the `resumes/` folder is included in
this repo for that purpose — then run:

```bash
uv run evidencerank \
  --jd machine_learning_engineer.txt \
  --resumes-dir resumes \
  --out-json report.json \
  --out-md report.md
```

The pipeline prints a `Running stage: <name>` line to stdout as each of the 5 stages
(`extract_profiles`, `prefilter`, `judge`, `calibrate`, `hallucination_check`) starts,
so you can follow progress on longer runs.

This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results) and `report.md` (a ranked Markdown table).

**Note:** `report.json`'s `profiles` section contains unredacted candidate identity
data (name, email, phone, location, raw CV text) — this is an intentional full
audit-trail artifact. It is not the same view the Judge model sees (that input is
redacted for blind evaluation; see the design spec's Fairness section). Don't treat or
share `report.json` as if it were already anonymized.

## Model configuration

Override any stage's model with an environment variable:

```bash
export EVIDENCERANK_MODEL_JUDGE=qwen2.5:32b-instruct
```

Valid stages: `jd_parser`, `cv_extractor`, `judge`, `calibrator`.

## Environment variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

`.env` is loaded automatically (via `python-dotenv`) whenever you run `uv run evidencerank`
or import `evaluation.metrics` — no manual `export` needed. `.env` is gitignored — never
commit real tokens.

| Variable | Required for | Notes |
| --- | --- | --- |
| `HF_TOKEN` | Downloading the `BAAI/bge-small-en-v1.5` embedding model used by the pre-filter stage | Only needed if you hit Hugging Face Hub rate limits/auth requirements on first download; the model is cached locally afterward. Get a token at https://huggingface.co/settings/tokens. |
| `EVIDENCERANK_EVAL_MODEL` | `evaluation/metrics.py` GEval metrics | Optional. Ollama model used as the GEval judge; defaults to `qwen2.5:14b-instruct` (same model already pulled for the production judge stage). Requires `ollama serve` running locally, same as the production pipeline. |

## Research evaluation harness

The `evaluation/` package is separate from the production pipeline (`src/evidencerank/`):

- `evaluation/metrics.py` — DeepEval `GEval` metrics (Groundedness, RecruiterAlignment,
  EvidenceRelevancy) to run against pipeline output.
- `evaluation/rank_stability.py` — computes Spearman/Kendall-tau rank correlation across
  repeated runs on the same input, to report LLM judgment consistency.

The three `GEval` metrics in `evaluation/metrics.py` use a local Ollama model as the
judge (`EVIDENCERANK_EVAL_MODEL`, see [Environment variables](#environment-variables)),
same as the production pipeline — no external API key required.

To measure rank stability, run the pipeline N times on the same JD/resumes with
different `--out-json` paths, then:

```python
from evaluation.rank_stability import rank_stability
print(rank_stability(["run1.json", "run2.json", "run3.json"]))
```
# candidateranking
