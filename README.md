# EvidenceRank

Evidence-grounded multi-agent LLM pipeline for ranking IT job candidates against a job
description, using local Ollama models orchestrated with LangGraph. See
`docs/superpowers/specs/2026-07-24-evidencerank-design.md` for the full design, and
`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md` /
`docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md` for the
evaluation-report tooling described below. See
`docs/superpowers/specs/2026-07-27-judge-grounding-and-hallucination-design.md` for a
later change that reorders the pipeline (Hallucination Checker now runs before the Pool
Calibrator, not after) and reworks Judge grounding — supersedes the pipeline diagram in
the original design doc.

## Setup

1. Install [Ollama](https://ollama.com) and start the server: `ollama serve`
2. Pull the default models:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull qwen2.5:14b-instruct
   ollama pull qwen3:14b  # GEval judge for the evaluation harness (evaluation-metric.md)
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
  --resumes-dir resumes --llm-concurrency 4
```

The pipeline prints a `Running stage: <name>` line to stdout as each of the 5 stages
(`extract_profiles`, `prefilter`, `judge`, `hallucination_check`, `calibrate`)
starts, so you can follow progress on longer runs. `hallucination_check` runs before
`calibrate`, not after — unverified evidence is stripped from a candidate's Judge result
before it's calibrated (see `hallucination_reports` in `report.json` for the original,
unstripped evidence).

Every candidate that passes the pre-filter and gets judged proceeds to `calibrate` — there
is no shortlist cap. For very large candidate pools this means the Calibrator's single LLM
call (every judge result embedded in one prompt) can grow large enough to approach its
context window; see the `CALIBRATOR_NUM_CTX` comment in `agents/calibrator.py` if you're
running against hundreds of candidates.

This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results), `report.md` (a ranked Markdown table), and
`evaluation-metric.md` (the evaluation metric report — see
[Evaluation metric report](#evaluation-metric-report) below), all written to the
directory you run `evidencerank` from. Each run overwrites the previous one's
`report.json`/`report.md`/`evaluation-metric.md` — rename them (e.g. `mv report.json
run1.json`) between runs if you need to keep more than one.

Extracted candidate profiles are cached at `.cache/evidencerank/extract_profiles/`
(relative to the directory you run `evidencerank` from), keyed by a hash of the
resume text, the CV-extractor prompt, the extractor model, and the extractor's
output schema — so re-running the pipeline against the same resumes skips
re-extracting a profile whose inputs haven't changed. Editing the extractor
prompt, switching the `EVIDENCERANK_MODEL_CV_EXTRACTOR` model, or changing the
extracted-fields schema all automatically invalidate the relevant cache entries;
deleting the directory forces full re-extraction. **Like `report.json`, cached
entries contain unredacted candidate contact info (name, email, phone,
location) — don't treat `.cache/` as anonymized, and don't share it casually.**

`report.md`'s table includes a "Hallucination Flags" column showing how many
evidence items were removed for that candidate before calibration (see
`report.json`'s `hallucination_reports` for the removed quotes themselves) — a
dash (`—`) means every quote verified.

Two thresholds are fixed (not CLI-configurable). The embedding pre-filter checks each
JD required skill individually against a candidate's skill list — a required skill
"exists" for that candidate if any one of their skills has cosine similarity `>= 0.7`
to it — and the candidate passes if at least `2` of the JD's required skills exist
for them (a fixed minimum count, not a majority — it doesn't scale with how many
required skills the JD lists). (This is a per-skill existence count, not a single
whole-list similarity score — a candidate whose skills are all generically
tech-adjacent but don't individually match specific required skills like "Machine
Learning" or "PyTorch" will fail even if a naive whole-list comparison would look
superficially similar.) The
hallucination checker's minimum fuzzy-match score for a quoted piece of evidence to
count as verified is `85.0`. Verification compares each quote against the candidate's
raw resume text (`profile.raw_cv_text`), not the CV-extractor's parsed fields — a
genuine, verbatim quote is only ever genuine because it's copied from the resume
itself, so this verifies a quote from any resume section regardless of how (or
whether) the extractor captured it, closing the false-positive gap the old
extracted-fields comparison had for sections the extractor didn't parse into a
structured field (e.g. a summary paragraph or a skills-category header line). The
trade-off: if the extractor mis-transcribes something and the Judge echoes that
error into a quote, it won't be caught here — but the Judge is prompted to quote
only from the resume text, not the extracted fields, so this only matters if the
Judge disobeys that instruction (e.g. by echoing a parsed/paraphrased field like
`education.degree` instead of the source text — the checker still correctly flags
that case, since it won't fuzzy-match the raw resume).

`--llm-concurrency` (default `4`) bounds how many candidates' `extract_profiles`
and `judge` LLM calls run concurrently, using LangChain's `Runnable.batch()`
instead of one sequential call per candidate. The default matches Ollama's own
default concurrent-request limit (`OLLAMA_NUM_PARALLEL`) on recent versions —
raising `--llm-concurrency` past what Ollama and your GPU's VRAM can actually
run at once adds contention overhead without speeding anything up, so tune it
alongside `OLLAMA_NUM_PARALLEL` rather than in isolation.

**Note:** `report.json`'s `profiles` section contains unredacted candidate identity
data (name, email, phone, location, raw CV text) — this is an intentional full
audit-trail artifact. It is not the same view the Judge model sees (that input is
redacted for blind evaluation; see the design spec's Fairness section). Don't treat or
share `report.json` as if it were already anonymized.

Every run also generates the evaluation metric report (`evaluation-metric.md`) — see
[Evaluation metric report](#evaluation-metric-report) below for what it contains. This
requires `ollama serve` running locally with the eval judge model available (same as the
standalone `evidencerank-eval-report` command).

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
| `EVIDENCERANK_EVAL_MODEL` | `evaluation/metrics.py` GEval metrics | Optional. Ollama model used as the GEval judge; defaults to `qwen3:14b` (`ollama pull qwen3:14b`). Requires `ollama serve` running locally, same as the production pipeline. |

## Research evaluation harness

The `evaluation/` package is separate from the production pipeline (`src/evidencerank/`):

- `evaluation/metrics.py` — DeepEval `GEval` metrics (RecruiterAlignment, EvidenceRelevancy)
  to run against pipeline output. There is deliberately no GEval "Groundedness" metric —
  quote authenticity is already measured deterministically by the production pipeline's
  hallucination checker (see Hallucination Rate below), which is strictly more reliable
  for that question than an LLM re-judging it (see the caveat below).
- `evaluation/rank_stability.py` — computes Spearman/Kendall-tau rank correlation across
  repeated runs on the same input, to report LLM judgment consistency.
- `evaluation/report.py` — aggregates the above (plus pipeline stats: candidates
  submitted, pre-filter pass/drop, hallucination rate) into a single Markdown
  evaluation report, suitable for a paper appendix.

The two `GEval` metrics in `evaluation/metrics.py` use a local Ollama model as the
judge (`EVIDENCERANK_EVAL_MODEL`, see [Environment variables](#environment-variables)) —
no external API key required, same as the production pipeline, though a different model
by default (`qwen3:14b`, a reasoning model, rather than the production judge's
`qwen2.5:14b-instruct`). A non-reasoning model of the same size was measurably less
reliable as an evaluator: it sometimes misjudged its own side-by-side text comparison
(e.g. claiming a quote didn't match the resume when it was in fact an exact match),
dragging down GEval scores with the eval judge's own errors rather than real
production-pipeline defects.

To measure rank stability, `uv run evidencerank-rank-stability` runs the pipeline
multiple times on the same JD/resumes automatically — no manual renaming needed:

```bash
uv run evidencerank-rank-stability --jd machine_learning_engineer.txt --resumes-dir resumes --runs 3
```

This runs the pipeline `--runs` times (default `3`, minimum `2`), writes each run's full
report as `run1.json`, `run2.json`, ... (never overwritten, so every run stays available
for inspection), and builds `evaluation-metric.md` from all of them — GEval scores and
pipeline stats from `run1.json`, rank stability (Spearman/Kendall-tau) across all of
them. `--llm-concurrency` and `--out` work the same as the other commands.

If you'd rather drive this manually (e.g. against runs you already have, or with
resumes/JD changing between runs), run the pipeline yourself N times, renaming
`report.json` after each run since every run overwrites it, then call
`evidencerank-eval-report` (below) with all the paths.

### Evaluation metric report

`uv run evidencerank-eval-report` builds a Markdown report combining GEval metric
aggregates, pipeline stats, and (when 2+ runs are given) rank stability, from one or
more existing `report.json` files:

```bash
uv run evidencerank-eval-report --reports report.json --out evaluation-metric.md
```

If you're evaluating a single run right after producing it, `evidencerank rank` (see
[Running the pipeline](#running-the-pipeline) above) already does this in the same run —
no separate command needed.

Pass `--reports` once per report path — repeat it for each additional run to also
include rank stability across runs:

```bash
uv run evidencerank-eval-report \
  --reports run1.json --reports run2.json --reports run3.json \
  --out evaluation-metric.md
```

GEval scores and pipeline stats are always computed from the first `--reports` path
given; every path is used for rank stability. This requires `ollama serve` running
locally (same GEval judge model as above) — the GEval calls are not mocked outside
of tests.

When the underlying `report.json` includes per-stage timing (`stage_timings`,
added by the production pipeline), the report also includes a "Stage Timings"
table showing wall-clock seconds per stage — absent for older `report.json`
files that predate this field.

Quote authenticity itself isn't a GEval metric (see Research evaluation harness above) —
it's already measured deterministically via Hallucination Rate in Pipeline Stats, since
the hallucination checker strips unverified evidence before calibration. RecruiterAlignment
and EvidenceRelevancy are the GEval signals for judge quality, since calibration and
claim-quote relevance have no deterministic equivalent.
