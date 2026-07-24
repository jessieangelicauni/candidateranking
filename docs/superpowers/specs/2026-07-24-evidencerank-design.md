# EvidenceRank Design

**Date:** 2026-07-24
**Status:** Approved for planning

## Purpose

EvidenceRank is a multi-agent LLM pipeline that ranks IT job candidates against a job
description (JD). It exists to support Q1-Scopus-track research into LLM-based
applicant tracking: the goal is a system that reasons about fit the way an experienced
recruiter would — weighing company reputation, experience depth, measurable impact, and
technical alignment — rather than keyword-matching, while grounding every judgment in
quoted evidence from the source resume.

The target evaluation set is on the order of 200 IT candidates against a single JD. The
current project folder has 5 sample resumes (`Daniel Taylor.pdf`, `Danny Morgan.pdf`,
`James morgan.pdf`, `Katie Hoover.pdf`, `Michael Burton.pdf`) and one JD
(`machine_learning_engineer.txt`) used to build and validate the pipeline before it is
pointed at the full candidate pool.

**Hard constraint:** no deterministic, rule-based scoring formula (e.g., weighted sums of
sub-scores) may determine a candidate's fit or rank. All fit judgments are LLM semantic
reasoning. Deterministic logic is permitted only for two non-judgment purposes: cutting
compute (the embedding pre-filter) and verifying grounding (the hallucination checker) —
neither one decides how well a candidate fits the role.

## Architecture

Orchestration uses **LangGraph**, modeling the pipeline as a stateful graph rather than a
linear LangChain chain. This is required by the pipeline's real branching structure:
per-candidate work fans out in parallel, narrows at the pre-filter, fans out again for
independent judging, then fans back in for pool-wide calibration.

```
                    ┌──────────────┐
                    │  JD Parser   │  (once, upfront)
                    └──────┬───────┘
                           │ JDRequirements
                           ▼
   ┌───────────────────────────────────────────┐
   │         per candidate (parallel)           │
   │  ┌───────────────┐                         │
   │  │ CV Extractor   │  raw PDF → profile +    │
   │  │                │  raw CV text retained   │
   │  └──────┬─────────┘                         │
   │         ▼                                   │
   │  ┌───────────────┐                          │
   │  │ Embedding      │  cosine sim vs threshold │
   │  │ Pre-filter     │  (deterministic cut)     │
   │  └──────┬─────────┘                          │
   │      pass │  fail → dropped, logged          │
   │         ▼                                   │
   │  ┌───────────────┐                          │
   │  │ Candidate Judge│  tier + 1-10 rating +    │
   │  │ (LLM)          │  quoted evidence         │
   │  └──────┬─────────┘                          │
   └─────────┼───────────────────────────────────┘
             ▼ (fan-in: all surviving candidates)
      ┌───────────────┐
      │ Pool Calibrator│  reconciles relative
      │ (LLM)          │  ordering across pool
      └──────┬─────────┘
             ▼
      ┌───────────────┐
      │ Hallucination  │  fuzzy-match every quote
      │ Checker        │  against source CV text
      └──────┬─────────┘
             ▼
      Final ranked report (JSON + Markdown)
```

## Components

### 1. JD Parser (LLM agent)
- Input: JD text file.
- Output: `JDRequirements` — required skills, nice-to-have skills, minimum experience
  years, education requirements, core responsibilities.
- Runs once per pipeline run.

### 2. CV Extractor (LLM agent)
- Input: one resume (PDF, parsed to text).
- Output: `CandidateProfile` — skills, work history (title, company, dates,
  achievements), education, projects — **plus the raw extracted CV text retained
  alongside the structured profile.**
- The raw text is kept so the Judge can quote directly from source and the
  Hallucination Checker can verify against source, rather than everything being
  filtered through the extractor's own (possibly lossy) interpretation.

### 3. Embedding Pre-filter (deterministic, non-LLM)
- Uses `bge-small-en-v1.5` via the `sentence-transformers` library (not Ollama — Ollama
  does not host this exact model).
- Embeds JD required-skills text and each candidate's skills text, computes cosine
  similarity, drops candidates below a configurable threshold before they reach the
  (expensive) LLM Judge stage.
- This step only decides which candidates *get evaluated*, not how they rank —
  it does not touch final scoring.
- Dropped candidates are still recorded in the output with the reason "pre-filter: no
  relevant skill overlap" for auditability.

### 4. Candidate Judge (LLM agent, one call per surviving candidate)
- Input: `JDRequirements`, candidate's `CandidateProfile`, raw CV text.
- Output: qualitative tier (`Strong Fit` / `Moderate Fit` / `Weak Fit` / `Not a Fit`), a
  1–10 holistic rating **assigned by the model as its own judgment** (not computed from
  sub-scores by a formula), and a rationale where every factual claim is backed by a
  verbatim quote from the CV.
- Reasoning should reflect recruiter-style holistic judgment: longer relevant experience
  increases confidence, measurable impact matters more than job titles, and technical
  skill alignment with JD requirements is weighted meaningfully — as contextual
  reasoning, not keyword matching.
- Before this stage runs, candidate identity fields (name, email, phone, photo,
  location) are stripped from what the Judge sees — blind evaluation, to reduce
  exposure to bias unrelated to job fit (see Fairness below).

### 5. Pool Calibrator (LLM agent, one call, whole surviving pool)
- Input: all surviving candidates' Judge outputs (tier, rating, evidence) together.
- Output: final relative rank order, with brief calibration notes explaining any
  adjustments (e.g., "Judge rated candidate 3 as Strong Fit but pool comparison shows
  weaker experience depth than candidate 1, also Strong Fit — reordered").
- Purpose: correct for anchoring/leniency drift between independent per-candidate Judge
  calls, since resumes are not evaluated in true isolation in the final ranking.

### 6. Hallucination Checker (deterministic, non-LLM)
- For every quoted evidence string in the Judge/Calibrator output, fuzzy-matches
  (`rapidfuzz`) it against that candidate's raw CV text.
- Quotes that fail to match above a similarity threshold are flagged in the output as
  unverified warnings by default (the candidate's other results still stand; the report
  surfaces which specific claims are unverified). An opt-in flag enables automatic
  re-judging of candidates with unverified quotes instead, for stricter research runs.
- An LLM is deliberately **not** used to check another LLM's grounding — an LLM judging
  hallucination is itself prone to hallucinating that check. String-level verification
  against source is the more defensible technique for a research paper.

## Fairness

Blind evaluation: identity fields (name, email, phone, photo, location) are stripped
from the profile before the Judge stage. Company names and university names are
retained, since the design intentionally wants the Judge to reason about e.g. "reputable
multinational company" exposure — this is an explicit, disclosed reasoning input rather
than an incidental bias, and is reported as such in the evidence trail.

## Model Configuration

All models run locally via Ollama, sized for a 16GB VRAM GPU (RTX 4070):

| Stage | Model | Rationale |
|---|---|---|
| JD Parser | `qwen2.5:7b-instruct` | Fast, strong structured JSON extraction |
| CV Extractor | `qwen2.5:7b-instruct` | Same — extraction, not deep reasoning |
| Candidate Judge | `qwen2.5:14b-instruct` (q4, ~9GB) | Most reasoning-heavy stage gets the largest model that comfortably fits VRAM |
| Pool Calibrator | `qwen2.5:14b-instruct` | Same reasoning demands as Judge |

Model names are configurable (not hardcoded) since comparing model choices is expected
to be part of the research work.

## Data Flow & Schemas (indicative)

```
JDRequirements:
  required_skills: [str]
  nice_to_have_skills: [str]
  min_experience_years: int
  education: str
  responsibilities: [str]

CandidateProfile:
  candidate_id: str
  raw_cv_text: str
  skills: [str]
  work_history: [{title, company, start, end, achievements: [str]}]
  education: [{degree, institution, year}]
  projects: [{name, description, tech}]

JudgeResult:
  candidate_id: str
  tier: enum(Strong Fit, Moderate Fit, Weak Fit, Not a Fit)
  rating: int (1-10)
  evidence: [{claim: str, quote: str}]

CalibratedResult:
  candidate_id: str
  final_rank: int
  tier: ...
  rating: ...
  calibration_notes: str

HallucinationReport:
  candidate_id: str
  unverified_quotes: [str]
```

## Evaluation Harness (research validation, separate from the production pipeline)

Lives in `evaluation/`, run against pipeline outputs — not embedded in the pipeline
itself, keeping the production path and the research measurement path cleanly separate.

Uses DeepEval:
- Custom Faithfulness/Groundedness metric on Judge evidence (does the rationale follow
  from the quoted evidence).
- G-Eval "recruiter alignment" rubric metric (LLM-graded comparison against defined
  fit criteria).
- Answer Relevancy metric (does cited evidence actually relate to the JD requirement it
  supports).
- Custom rank-stability metric: run the pipeline N times on the same input and report
  Spearman/Kendall-tau correlation across runs, as a measure of LLM judgment
  consistency for the paper.

## Output

- Full JSON per run: every stage's intermediate output, full evidence trail, dropped
  candidates with reasons, hallucination check results — for reproducibility.
- Markdown ranking report: rank, candidate, tier, rating, key evidence, calibration
  notes — formatted for direct use in a paper appendix.

## Project Structure

```
src/evidencerank/
  agents/
    jd_parser.py
    cv_extractor.py
    prefilter.py
    judge.py
    calibrator.py
    hallucination_checker.py
  graph.py       # LangGraph pipeline definition
  models.py      # Pydantic schemas
  llm.py         # Ollama client wrappers, per-stage model config
  io.py          # PDF/text loading for resumes and JD
  report.py      # JSON + Markdown report generation
evaluation/       # DeepEval suite (separate from production pipeline)
cli.py            # entry point
tests/            # pytest, mocked LLM calls, schema/parsing unit tests
```

## Out of Scope (this build)

- No API-based LLM fallback — Ollama-only, per explicit decision.
- No web UI — CLI-driven, JSON/Markdown output only.
- No live scaling test against the full ~200-candidate pool as part of this design; the
  architecture is sized for it, but validation happens on the 5 current resumes first.
