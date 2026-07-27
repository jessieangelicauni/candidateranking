from evidencerank.cache import compute_cache_key, load_cached_json, save_cached_json


def test_compute_cache_key_same_inputs_same_key():
    key_a = compute_cache_key("resume text", "prompt text", "model-name")
    key_b = compute_cache_key("resume text", "prompt text", "model-name")

    assert key_a == key_b


def test_compute_cache_key_different_inputs_different_key():
    key_a = compute_cache_key("resume text", "prompt text", "model-name")
    key_b = compute_cache_key("different resume text", "prompt text", "model-name")

    assert key_a != key_b


def test_compute_cache_key_avoids_boundary_collision():
    # Without a delimiter between parts, ("ab", "c") and ("a", "bc") would
    # hash identically (both concatenate to "abc").
    key_a = compute_cache_key("ab", "c")
    key_b = compute_cache_key("a", "bc")

    assert key_a != key_b


def test_load_cached_json_returns_none_when_missing(tmp_path):
    result = load_cached_json(tmp_path, "nonexistent-key")

    assert result is None


def test_save_and_load_cached_json_round_trips(tmp_path):
    data = {"skills": ["Python", "SQL"], "contact": {"name": "Jane Doe"}}

    save_cached_json(tmp_path, "some-key", data)
    result = load_cached_json(tmp_path, "some-key")

    assert result == data


def test_save_cached_json_creates_cache_dir_if_missing(tmp_path):
    cache_dir = tmp_path / "nested" / "cache" / "dir"

    save_cached_json(cache_dir, "some-key", {"a": 1})

    assert (cache_dir / "some-key.json").exists()


def test_load_cached_json_returns_none_for_corrupt_file(tmp_path):
    path = tmp_path / "some-key.json"
    path.write_text("{not valid json", encoding="utf-8")

    result = load_cached_json(tmp_path, "some-key")

    assert result is None
