import json
from pathlib import Path

from evidencerank.cache import compute_cache_key, load_cached_json, save_cached_json
from evidencerank.llm import get_chat_model, resolve_model_name
from evidencerank.models import CandidateProfile, ExtractedProfileFields

CV_EXTRACTOR_PROMPT = """You are an expert technical recruiter. Read the resume below and \
extract the candidate's contact info, skills, work history, education, and projects exactly \
as stated. Do not infer skills or experience that are not explicitly present in the text.

A resume's work history is not always under a heading literally called "Work History" or \
"Experience" - treat any section describing paid work, internships, or contract roles as work \
history regardless of its heading wording (e.g. "Employment History", "Career History", or a \
combined heading like "Internships / Experience"), and extract every entry in it.

Resume:
{cv_text}
"""

DEFAULT_CACHE_DIR = Path(".cache/evidencerank/extract_profiles")


def _build_cv_extractor_prompt(cv_text: str) -> str:
    return CV_EXTRACTOR_PROMPT.format(cv_text=cv_text)


def _cache_key_for(cv_text: str) -> str:
    return compute_cache_key(
        cv_text,
        CV_EXTRACTOR_PROMPT,
        resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )


def extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile:
    model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
    fields = model.invoke(_build_cv_extractor_prompt(cv_text))
    return CandidateProfile(
        candidate_id=candidate_id,
        raw_cv_text=cv_text,
        **fields.model_dump(),
    )


def cached_extract_cv(
    candidate_id: str,
    cv_text: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> CandidateProfile:
    key = _cache_key_for(cv_text)
    cached = load_cached_json(cache_dir, key)
    if cached is not None:
        return CandidateProfile(candidate_id=candidate_id, raw_cv_text=cv_text, **cached)

    profile = extract_cv(candidate_id, cv_text)
    save_cached_json(
        cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
    )
    return profile


def cached_extract_cvs(
    candidates: dict[str, str],
    max_concurrency: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, CandidateProfile]:
    results: dict[str, CandidateProfile] = {}
    misses: list[tuple[str, str, str]] = []
    for candidate_id, cv_text in candidates.items():
        key = _cache_key_for(cv_text)
        cached = load_cached_json(cache_dir, key)
        if cached is not None:
            results[candidate_id] = CandidateProfile(
                candidate_id=candidate_id, raw_cv_text=cv_text, **cached
            )
        else:
            misses.append((candidate_id, cv_text, key))

    if misses:
        model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
        prompts = [_build_cv_extractor_prompt(cv_text) for _, cv_text, _ in misses]
        fields_list = model.batch(prompts, config={"max_concurrency": max_concurrency})
        for (candidate_id, cv_text, key), fields in zip(misses, fields_list):
            profile = CandidateProfile(
                candidate_id=candidate_id, raw_cv_text=cv_text, **fields.model_dump()
            )
            save_cached_json(
                cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
            )
            results[candidate_id] = profile

    return {candidate_id: results[candidate_id] for candidate_id in candidates}
