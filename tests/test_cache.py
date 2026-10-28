"""Tests for SQLite caching layer, cache modes, and Meta compliance checks."""

import tempfile
from pathlib import Path
import pytest

from stepproof.cache import SQLiteCache, make_cache_key
from stepproof.provenance import create_provenance


@pytest.fixture
def temp_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.db"
        yield SQLiteCache(db_path=db_path)


def test_cache_put_and_get(temp_cache: SQLiteCache):
    prov = create_provenance("mock", "mock-model", meta_compliant=False)
    key = make_cache_key("mock", "mock-model", "v1.0", {"query": "hello"})

    temp_cache.put(
        cache_key=key,
        provider="mock",
        model_name="mock-model",
        prompt_version="v1.0",
        input_data={"query": "hello"},
        response_data={"reply": "world"},
        provenance=prov,
    )

    hit = temp_cache.get(key, require_meta=False)
    assert hit is not None
    data, hit_prov = hit
    assert data["reply"] == "world"
    assert hit_prov.model_name == "mock-model"
    assert hit_prov.meta_compliant is False


def test_cache_requires_meta_refusal(temp_cache: SQLiteCache):
    prov_mock = create_provenance("mock", "mock-model", meta_compliant=False)
    key = make_cache_key("mock", "mock-model", "v1.0", "sample")

    temp_cache.put(
        cache_key=key,
        provider="mock",
        model_name="mock-model",
        prompt_version="v1.0",
        input_data="sample",
        response_data={"status": "ok"},
        provenance=prov_mock,
    )

    # When require_meta is True, get() MUST raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        temp_cache.get(key, require_meta=True)
    assert "refuses non-Meta models" in str(exc_info.value)

    # Compliant entry passes
    prov_meta = create_provenance("local_sam", "sam-3.1", meta_compliant=True)
    key_meta = make_cache_key("local_sam", "sam-3.1", "v1.0", "sample2")
    temp_cache.put(
        cache_key=key_meta,
        provider="local_sam",
        model_name="sam-3.1",
        prompt_version="v1.0",
        input_data="sample2",
        response_data={"status": "meta_ok"},
        provenance=prov_meta,
    )
    hit_meta = temp_cache.get(key_meta, require_meta=True)
    assert hit_meta is not None
    assert hit_meta[0]["status"] == "meta_ok"


def test_cache_modes(temp_cache: SQLiteCache):
    calls = 0

    def mock_caller():
        nonlocal calls
        calls += 1
        prov = create_provenance("mock", "mock-model")
        return {"count": calls}, prov

    # 1. LIVE mode: first call executes caller_fn
    res1, _ = temp_cache.execute_with_cache(
        provider_name="mock",
        model_name="mock-model",
        prompt_version="v1",
        input_data="inputA",
        caller_fn=mock_caller,
        mode="live",
        require_meta=False,
    )
    assert calls == 1
    assert res1["count"] == 1

    # LIVE mode second call hits cache
    res2, _ = temp_cache.execute_with_cache(
        provider_name="mock",
        model_name="mock-model",
        prompt_version="v1",
        input_data="inputA",
        caller_fn=mock_caller,
        mode="live",
        require_meta=False,
    )
    assert calls == 1
    assert res2["count"] == 1

    # 2. CACHE_ONLY mode: hit returns result, miss raises KeyError
    res3, _ = temp_cache.execute_with_cache(
        provider_name="mock",
        model_name="mock-model",
        prompt_version="v1",
        input_data="inputA",
        caller_fn=mock_caller,
        mode="cache_only",
        require_meta=False,
    )
    assert calls == 1
    assert res3["count"] == 1

    with pytest.raises(KeyError):
        temp_cache.execute_with_cache(
            provider_name="mock",
            model_name="mock-model",
            prompt_version="v1",
            input_data="missing_key",
            caller_fn=mock_caller,
            mode="cache_only",
            require_meta=False,
        )

    # 3. REFRESH mode: executes caller_fn and updates cache
    res4, _ = temp_cache.execute_with_cache(
        provider_name="mock",
        model_name="mock-model",
        prompt_version="v1",
        input_data="inputA",
        caller_fn=mock_caller,
        mode="refresh",
        require_meta=False,
    )
    assert calls == 2
    assert res4["count"] == 2
