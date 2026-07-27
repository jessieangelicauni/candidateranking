import os

from langchain_ollama import ChatOllama

DEFAULT_MODELS: dict[str, str] = {
    "jd_parser": "qwen2.5:7b-instruct",
    "cv_extractor": "qwen2.5:7b-instruct",
    "judge": "qwen2.5:14b-instruct",
    "calibrator": "qwen2.5:14b-instruct",
}

_ENV_PREFIX = "EVIDENCERANK_MODEL_"


def resolve_model_name(stage: str) -> str:
    if stage not in DEFAULT_MODELS:
        raise ValueError(f"Unknown stage: {stage!r}. Known stages: {sorted(DEFAULT_MODELS)}")
    env_key = f"{_ENV_PREFIX}{stage.upper()}"
    return os.environ.get(env_key, DEFAULT_MODELS[stage])


def get_chat_model(stage: str, temperature: float = 0.0, num_ctx: int | None = None) -> ChatOllama:
    return ChatOllama(model=resolve_model_name(stage), temperature=temperature, num_ctx=num_ctx)
