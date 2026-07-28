import json
from unittest.mock import MagicMock

from evidencerank.agents.cv_extractor import CV_EXTRACTOR_PROMPT, cached_extract_cv, cached_extract_cvs, extract_cv
from evidencerank.cache import compute_cache_key, save_cached_json
from evidencerank.llm import resolve_model_name
from evidencerank.models import ContactInfo, ExtractedProfileFields


def test_extract_cv_assembles_candidate_profile(monkeypatch):
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
        skills=["Python", "SQL"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = extract_cv("c1", "Jane Doe resume text...")

    assert profile.candidate_id == "c1"
    assert profile.raw_cv_text == "Jane Doe resume text..."
    assert profile.contact.name == "Jane Doe"
    assert profile.skills == ["Python", "SQL"]
    fake_chat_model.with_structured_output.assert_called_once_with(ExtractedProfileFields)


def test_cached_extract_cv_skips_llm_on_cache_hit(tmp_path, monkeypatch):
    cv_text = "Jane Doe resume text..."
    cached_fields = {
        "contact": {"name": "Jane Doe", "email": "jane@example.com", "phone": "", "location": ""},
        "skills": ["Python", "SQL"],
        "work_history": [],
        "education": [],
        "projects": [],
    }
    key = compute_cache_key(
        cv_text,
        CV_EXTRACTOR_PROMPT,
        resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )
    save_cached_json(tmp_path, key, cached_fields)

    fake_chat_model = MagicMock()
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = cached_extract_cv("c1", cv_text, cache_dir=tmp_path)

    assert profile.candidate_id == "c1"
    assert profile.raw_cv_text == cv_text
    assert profile.contact.name == "Jane Doe"
    assert profile.skills == ["Python", "SQL"]
    fake_chat_model.with_structured_output.assert_not_called()


def test_cached_extract_cv_calls_llm_and_writes_cache_on_miss(tmp_path, monkeypatch):
    cv_text = "John Smith resume text..."
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="John Smith", email="john@example.com"),
        skills=["Go", "Rust"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    profile = cached_extract_cv("c2", cv_text, cache_dir=tmp_path)

    assert profile.candidate_id == "c2"
    assert profile.skills == ["Go", "Rust"]
    fake_chat_model.with_structured_output.assert_called_once_with(ExtractedProfileFields)

    key = compute_cache_key(
        cv_text,
        CV_EXTRACTOR_PROMPT,
        resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )
    cache_file = tmp_path / f"{key}.json"
    assert cache_file.exists()
    cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached_data["skills"] == ["Go", "Rust"]
    assert cached_data["contact"]["name"] == "John Smith"


def test_cached_extract_cv_second_call_with_same_input_skips_llm(tmp_path, monkeypatch):
    cv_text = "Repeat candidate resume text..."
    extracted = ExtractedProfileFields(
        contact=ContactInfo(name="Repeat Candidate"),
        skills=["Java"],
    )
    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = extracted
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model",
        lambda stage: fake_chat_model,
    )

    first = cached_extract_cv("c3", cv_text, cache_dir=tmp_path)
    second = cached_extract_cv("c3", cv_text, cache_dir=tmp_path)

    assert first.skills == second.skills == ["Java"]
    fake_structured_model.invoke.assert_called_once()


def test_cached_extract_cvs_all_hits_skips_llm_entirely(tmp_path, monkeypatch):
    candidates = {
        "c1": "Jane Doe resume text...",
        "c2": "John Smith resume text...",
    }
    cached_fields = {
        "c1": {
            "contact": {"name": "Jane Doe", "email": "", "phone": "", "location": ""},
            "skills": ["Python"], "work_history": [], "education": [], "projects": [],
        },
        "c2": {
            "contact": {"name": "John Smith", "email": "", "phone": "", "location": ""},
            "skills": ["Go"], "work_history": [], "education": [], "projects": [],
        },
    }
    for candidate_id, cv_text in candidates.items():
        key = compute_cache_key(
            cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"),
            json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
        )
        save_cached_json(tmp_path, key, cached_fields[candidate_id])

    fake_chat_model = MagicMock()
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model", lambda stage: fake_chat_model
    )

    profiles = cached_extract_cvs(candidates, max_concurrency=4, cache_dir=tmp_path)

    assert set(profiles.keys()) == {"c1", "c2"}
    assert profiles["c1"].skills == ["Python"]
    assert profiles["c2"].skills == ["Go"]
    fake_chat_model.with_structured_output.assert_not_called()


def test_cached_extract_cvs_all_misses_batches_and_caches_each(tmp_path, monkeypatch):
    candidates = {
        "c1": "Jane Doe resume text...",
        "c2": "John Smith resume text...",
    }
    extracted_c1 = ExtractedProfileFields(contact=ContactInfo(name="Jane Doe"), skills=["Python"])
    extracted_c2 = ExtractedProfileFields(contact=ContactInfo(name="John Smith"), skills=["Go"])
    fake_structured_model = MagicMock()
    fake_structured_model.batch.return_value = [extracted_c1, extracted_c2]
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model", lambda stage: fake_chat_model
    )

    profiles = cached_extract_cvs(candidates, max_concurrency=4, cache_dir=tmp_path)

    assert profiles["c1"].skills == ["Python"]
    assert profiles["c2"].skills == ["Go"]
    call_args, call_kwargs = fake_structured_model.batch.call_args
    assert len(call_args[0]) == 2
    assert call_kwargs["config"] == {"max_concurrency": 4}

    for candidate_id, cv_text in candidates.items():
        key = compute_cache_key(
            cv_text, CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"),
            json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
        )
        assert (tmp_path / f"{key}.json").exists()


def test_cached_extract_cvs_mixed_hits_and_misses_only_batches_misses(tmp_path, monkeypatch):
    candidates = {
        "c1": "Jane Doe resume text...",
        "c2": "John Smith resume text...",
    }
    cached_fields_c1 = {
        "contact": {"name": "Jane Doe", "email": "", "phone": "", "location": ""},
        "skills": ["Python"], "work_history": [], "education": [], "projects": [],
    }
    key_c1 = compute_cache_key(
        candidates["c1"], CV_EXTRACTOR_PROMPT, resolve_model_name("cv_extractor"),
        json.dumps(ExtractedProfileFields.model_json_schema(), sort_keys=True),
    )
    save_cached_json(tmp_path, key_c1, cached_fields_c1)

    extracted_c2 = ExtractedProfileFields(contact=ContactInfo(name="John Smith"), skills=["Go"])
    fake_structured_model = MagicMock()
    fake_structured_model.batch.return_value = [extracted_c2]
    fake_chat_model = MagicMock()
    fake_chat_model.with_structured_output.return_value = fake_structured_model
    monkeypatch.setattr(
        "evidencerank.agents.cv_extractor.get_chat_model", lambda stage: fake_chat_model
    )

    profiles = cached_extract_cvs(candidates, max_concurrency=4, cache_dir=tmp_path)

    assert profiles["c1"].skills == ["Python"]
    assert profiles["c2"].skills == ["Go"]
    call_args, _ = fake_structured_model.batch.call_args
    prompts_sent = call_args[0]
    assert len(prompts_sent) == 1
    assert candidates["c2"] in prompts_sent[0]
