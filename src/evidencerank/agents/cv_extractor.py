import json
from pathlib import Path

from evidencerank.cache import compute_cache_key, load_cached_json, save_cached_json
from evidencerank.llm import get_chat_model, resolve_model_name
from evidencerank.models import CandidateProfile, ExtractedProfileFields

CV_EXTRACTOR_PROMPT = """You are an expert technical recruiter. Read the resume below and \
extract the candidate's contact info, skills, work history, education, and projects exactly \
as stated. Do not infer skills or experience that are not explicitly present in the text.

Resume:
{cv_text}
"""

DEFAULT_CACHE_DIR = Path(".cache/evidencerank/extract_profiles")


def extract_cv(candidate_id: str, cv_text: str) -> CandidateProfile:
    model = get_chat_model("cv_extractor").with_structured_output(ExtractedProfileFields)
    fields = model.invoke(CV_EXTRACTOR_PROMPT.format(cv_text=cv_text))
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
    key = compute_cache_key(
        cv_text,
        CV_EXTRACTOR_PROMPT,
        resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )
    cached = load_cached_json(cache_dir, key)
    if cached is not None:
        return CandidateProfile(candidate_id=candidate_id, raw_cv_text=cv_text, **cached)

    profile = extract_cv(candidate_id, cv_text)
    save_cached_json(
        cache_dir, key, profile.model_dump(exclude={"candidate_id", "raw_cv_text"})
    )
    return profile
