# EvidenceRank

Evidence-grounded multi-agent LLM pipeline for ranking IT job candidates against a job
description, using local Ollama models orchestrated with LangGraph. See
`docs/superpowers/specs/2026-07-24-evidencerank-design.md` for the full design, and
`docs/superpowers/specs/2026-07-27-eval-metric-report-design.md` /
`docs/superpowers/specs/2026-07-27-eval-report-cli-integration-design.md` for the
original evaluation-report tooling this section describes — see
`docs/superpowers/specs/2026-07-29-unify-report-output-design.md` for a later change
that merges that tooling's separate Markdown evaluation report into `report.md`, superseding
those two docs' description of a standalone evaluation-report output file. See
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
   ```
3. Install project dependencies: `uv sync`
4. Set required environment variables (see [Environment variables](#environment-variables) below)
5. Run the test suite: `uv run pytest`

## Running the pipeline

Put all candidate resumes (PDF) in one folder — the `resumes/` folder is included in
this repo for that purpose — then run:

```bash
uv run evidencerank rank \
  --jd ai_data_engineer.txt \
  --resumes-dir resumes --llm-concurrency 12
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
context window if you're running against hundreds of candidates.

This produces `report.json` (full evidence trail, including dropped candidates and
hallucination check results) and `report.md` (a ranked Markdown table plus pipeline
stats — see [Research evaluation harness](#research-evaluation-harness) below for what the
stats section contains), both written to the directory you run `evidencerank` from.
Each run overwrites the previous one's `report.json`/`report.md` — rename them (e.g.
`mv report.json run1.json`) between runs if you need to keep more than one.

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
"exists" for that candidate if any one of their skills has cosine similarity `>= 0.9`
to it — and the candidate passes if at least `50%` of the JD's required skills exist
for them (`MIN_REQUIRED_SKILLS_FRACTION = 0.5` in `prefilter.py`, a proportion of
however many required skills the JD lists, not a fixed headcount — a JD with 4
required skills needs 2 matched, a JD with 16 needs 8). (This is a per-skill existence
count feeding that proportion, not a single whole-list similarity score — a candidate
whose skills are all generically tech-adjacent but don't individually match specific
required skills like "Machine Learning" or "PyTorch" will fail even if a naive
whole-list comparison would look superficially similar.) The
hallucination checker's minimum fuzzy-match score for a quoted piece of evidence to
count as verified is `85.0`. Verification compares each quote against the candidate's
raw resume text (`profile.raw_cv_text`), not the CV-extractor's parsed fields — a
genuine, verbatim quote is only ever genuine because it's copied from the resume
itself, so this verifies a quote from any resume section regardless of how (or
whether) the extractor captured it, closing the false-positive gap the old
extracted-fields comparison had for sections the extractor didn't parse into a
structured field (e.g. a summary paragraph or a skills-category header line).

The Judge's own prompt (`JUDGE_PROMPT` in `agents/judge.py`) only ever contains the
raw job description text and the candidate's raw, identity-redacted resume text — no
pre-parsed `JDRequirements` JSON and no CV-extractor's structured fields (skills,
work_history, education, projects) are sent to it. Fit is judged directly from what
both source documents actually say, not from a summarized/derived intermediate
representation of either. (The parsed `JDRequirements` and extracted `CandidateProfile`
are still used elsewhere in the pipeline — the pre-filter's embedding match needs
`required_skills`, and the Calibrator's pool-reconciliation pass still works from the
Judge's tier/rating/evidence output — this only changes what the Judge itself sees.)
One consequence: since the Judge has no structured fields to echo from at all anymore,
the old failure mode of it quoting a parsed/paraphrased field (e.g. `education.degree`)
instead of the source text can't happen — every quote it could possibly produce already
comes from the same raw text the hallucination checker verifies against.

`--llm-concurrency` (default `12`) bounds how many candidates' `extract_profiles`
and `judge` LLM calls run concurrently, using LangChain's `Runnable` concurrency
support instead of one sequential call per candidate (`batch_as_completed()` for
`extract_profiles`, so each result is cached as soon as it finishes rather than
only after the whole batch completes; `batch()` for `judge`, which isn't cached).
This only actually parallelizes if Ollama's own concurrent-request limit is
raised to match — `ollama serve` defaults to a single request at a time
(`OLLAMA_NUM_PARALLEL=1` on older versions, or capped low based on available
VRAM on newer ones), so `--llm-concurrency` requests will just queue at the
server unless you start Ollama with a matching value:

```bash
OLLAMA_NUM_PARALLEL=12 ollama serve
```

Benchmarked on an RTX 4070 Ti Super (16GB VRAM) against `qwen2.5:7b-instruct`:
`OLLAMA_NUM_PARALLEL=1` (effectively no concurrency) measured 8.5s/candidate,
`=4` measured 3.1s/candidate, `=8` measured 2.0s/candidate, `=12` measured
1.86s/candidate — real but sharply diminishing returns past 4 (going 8→12 is
a 50% increase in slots for only a 7% speed gain), since throughput becomes
GPU-compute-bound rather than slot-count-bound; `=8` is close to the practical
ceiling on this GPU, with `=12` mostly trading VRAM/compute headroom for a
small extra edge. Raising `--llm-concurrency`/`OLLAMA_NUM_PARALLEL` further
only helps if your GPU's VRAM and compute can actually sustain that many
concurrent requests; tune the two together rather than in isolation, and
watch VRAM headroom for the larger `judge` model (14B by default) specifically.

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
— no manual `export` needed. `.env` is gitignored — never commit real tokens.

| Variable | Required for | Notes |
| --- | --- | --- |
| `HF_TOKEN` | Downloading the `BAAI/bge-small-en-v1.5` embedding model used by the pre-filter stage | Only needed if you hit Hugging Face Hub rate limits/auth requirements on first download; the model is cached locally afterward. Get a token at https://huggingface.co/settings/tokens. |

## Research evaluation harness

`evaluation/report.py` builds every report the pipeline produces — including
`report.json` and `report.md` themselves, called directly by `src/evidencerank/cli.py`'s
`rank` command — plus the research-only aggregates appended to `report.md`'s Pipeline
Stats/Rank Stability sections (candidates submitted, pre-filter pass/drop, hallucination
rate, evidence relevancy, rank correlation across repeated runs). `rank_stability()` in
that same file computes the Spearman/Kendall-tau rank correlation piece.

All three signals — hallucination rate, evidence relevancy, and rank stability — are
deterministic, with no LLM judging another LLM's output involved and no `ollama serve`
requirement beyond what the production pipeline itself already needs. Hallucination rate
uses fuzzy string-matching; rank stability uses rank-correlation statistics. Evidence
Relevancy measures whether the Judge's evidence claims are actually about the JD's
requirements — for each judged candidate, every evidence claim is embedded (via the same
`BAAI/bge-small-en-v1.5` model the pre-filter stage already uses) and compared by cosine
similarity against the JD's required/nice-to-have skills and responsibilities, taking the
best match per claim, then averaging per candidate and across all judged candidates. This
is the deterministic, non-LLM counterpart to what a framework like RAGAS/DeepEval would
normally compute via an LLM judge (e.g. "Answer Relevancy") — same evaluation concept,
without the self-referential LLM-grading-LLM signal. Unlike the hallucination-rate and
rank-stability signals, computing this one does load a local embedding model — the same
one the pipeline's pre-filter stage already requires, not a new dependency, but worth
knowing if you're building a report purely from existing `report.json` files via
`--reports` with no other project setup done yet.

To measure rank stability, pass `--runs` (default `1`, use `2+` to get a stability
comparison) to `evidencerank rank` — it runs the pipeline multiple times on the same
JD/resumes automatically, no manual renaming needed:

```bash
uv run evidencerank rank --jd ai_data_engineer.txt --resumes-dir resumes --runs 3
```

This runs the pipeline `--runs` times, writes each run's full report as `run1.json`,
`run2.json`, ... (never overwritten, so every run stays available for inspection —
`report.json` itself is not written in this mode), and builds `report.md` from all of
them — rankings from `run1.json`, Pipeline Stats broken down per run plus a mean row
(see below), rank stability (Spearman/Kendall-tau) across all of them.
`--llm-concurrency` and `--out` work the same as a single run.

### Folding in past runs

`evidencerank rank` always runs the pipeline (that's the point of the command), but you
can also fold in existing `report.json` files from past runs — e.g. from before you
changed the JD or resumes — into the same rank-stability comparison, via `--reports`
(repeatable):

```bash
uv run evidencerank rank \
  --jd ai_data_engineer.txt --resumes-dir resumes \
  --reports run1.json --reports run2.json \
  --out report.md
```

This run's own output (`report.json`, or `run1.json..runN.json` if `--runs` > 1) is
always used for the Rankings section specifically; every `--reports` path plus this
run's own output(s) together feed Pipeline Stats and the Rank Stability comparison
whenever there are 2+ reports in total. `--reports` alone doesn't skip the pipeline —
it's additive input on top of a real run, not a replacement for one.

With a single report, Pipeline Stats is the familiar `Metric | Value` table. With 2+
reports, it switches to one row per report (`Run | Total candidates | Passed
pre-filter | ...`) plus a final `**Mean**` row averaging each metric across all of
them — so you can see both each run's own numbers and the aggregate at a glance, not
just one or the other. Candidate counts (`Total candidates`, `Passed pre-filter`,
etc.) in the `Mean` row display as a whole number when every report agrees, or with
one decimal place when they don't (e.g. pre-filter/extraction results varying
slightly between runs).

When the underlying `report.json` includes per-stage timing (`stage_timings`,
added by the production pipeline), the report also includes a "Stage Timings"
table showing wall-clock seconds per stage — absent for older `report.json`
files that predate this field.

Quote authenticity is measured deterministically via Hallucination Rate in Pipeline
Stats — the hallucination checker strips unverified evidence before calibration, so
this is a direct count from that check, not a judgment call by any model. Whether the
surviving evidence is actually relevant to the JD (as opposed to just verbatim-accurate)
is the separate Evidence Relevancy score, also in Pipeline Stats.
