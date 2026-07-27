from unittest.mock import MagicMock

import pytest

from evidencerank.llm import DEFAULT_MODELS, get_chat_model, resolve_model_name


def test_resolve_model_name_returns_default():
    assert resolve_model_name("judge") == DEFAULT_MODELS["judge"]


def test_resolve_model_name_respects_env_override(monkeypatch):
    monkeypatch.setenv("EVIDENCERANK_MODEL_JUDGE", "custom-model:latest")
    assert resolve_model_name("judge") == "custom-model:latest"


def test_resolve_model_name_rejects_unknown_stage():
    with pytest.raises(ValueError):
        resolve_model_name("not_a_stage")


def test_get_chat_model_passes_num_ctx_when_given(monkeypatch):
    fake_chat_ollama = MagicMock()
    monkeypatch.setattr("evidencerank.llm.ChatOllama", fake_chat_ollama)

    get_chat_model("judge", num_ctx=32768)

    fake_chat_ollama.assert_called_once_with(
        model=DEFAULT_MODELS["judge"], temperature=0.0, num_ctx=32768
    )


def test_get_chat_model_defaults_num_ctx_to_none(monkeypatch):
    fake_chat_ollama = MagicMock()
    monkeypatch.setattr("evidencerank.llm.ChatOllama", fake_chat_ollama)

    get_chat_model("judge")

    fake_chat_ollama.assert_called_once_with(
        model=DEFAULT_MODELS["judge"], temperature=0.0, num_ctx=None
    )
