import hashlib
import json
from pathlib import Path


def compute_cache_key(*parts: str) -> str:
    # \x1f (ASCII unit separator) can't appear in ordinary text, so it can't
    # cause a boundary collision between adjacent parts the way a plain
    # join or a printable delimiter like "|" could.
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_cached_json(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_cached_json(cache_dir: Path, key: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
