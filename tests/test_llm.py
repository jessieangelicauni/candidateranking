import pytest

from evidencerank.llm import DEFAULT_MODELS, resolve_model_name


def test_resolve_model_name_returns_default():
    assert resolve_model_name("judge") == DEFAULT_MODELS["judge"]


def test_resolve_model_name_respects_env_override(monkeypatch):
    monkeypatch.setenv("EVIDENCERANK_MODEL_JUDGE", "custom-model:latest")
    assert resolve_model_name("judge") == "custom-model:latest"


def test_resolve_model_name_rejects_unknown_stage():
    with pytest.raises(ValueError):
        resolve_model_name("not_a_stage")
