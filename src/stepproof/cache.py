"""SQLite-backed response cache with provenance tracking and Meta compliance guards."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable

from stepproof.config import settings
from stepproof.provenance import ProvenanceRecord, assert_meta_compliant

logger = logging.getLogger(__name__)


def compute_input_hash(data: Any) -> str:
    """Compute deterministic SHA256 hash of arbitrary input data."""
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, (dict, list)):
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
    else:
        raw = str(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_cache_key(
    provider: str,
    model_name: str,
    prompt_version: str,
    input_data: Any,
) -> str:
    """Construct cache key = sha256(provider:model:prompt_version:input_hash)."""
    input_hash = compute_input_hash(input_data)
    combined = f"{provider}:{model_name}:{prompt_version}:{input_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class SQLiteCache:
    """Persistent SQLite cache storing model outputs and their provenance."""

    def __init__(self, db_path: Path | str | None = None):
        path = Path(db_path or settings.cache_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create cache table if not already present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    meta_compliant INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def get(
        self,
        cache_key: str,
        require_meta: bool | None = None,
    ) -> tuple[Any, ProvenanceRecord] | None:
        """Retrieve cached output and provenance by cache_key.

        Args:
            cache_key: Unique cache key string.
            require_meta: If True, refuses non-meta compliant cached entries.

        Raises:
            RuntimeError: If require_meta is True and the cached entry is not meta-compliant.
        """
        must_be_meta = (
            require_meta
            if require_meta is not None
            else settings.stepproof_require_meta
        )

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT response_json, provenance_json, meta_compliant FROM model_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        prov_data = json.loads(row["provenance_json"])
        provenance = ProvenanceRecord.model_validate(prov_data)

        if must_be_meta:
            assert_meta_compliant(provenance)

        data = json.loads(row["response_json"])
        return data, provenance

    def put(
        self,
        cache_key: str,
        provider: str,
        model_name: str,
        prompt_version: str,
        input_data: Any,
        response_data: Any,
        provenance: ProvenanceRecord,
    ) -> None:
        """Store response data and provenance under cache_key."""
        input_hash = compute_input_hash(input_data)
        response_json = json.dumps(response_data)
        provenance_json = provenance.model_dump_json()
        meta_flag = 1 if provenance.meta_compliant else 0

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_cache (
                    cache_key, provider, model_name, prompt_version,
                    input_hash, response_json, provenance_json,
                    meta_compliant, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    provider,
                    model_name,
                    prompt_version,
                    input_hash,
                    response_json,
                    provenance_json,
                    meta_flag,
                    provenance.timestamp,
                ),
            )
            conn.commit()

    def execute_with_cache(
        self,
        provider_name: str,
        model_name: str,
        prompt_version: str,
        input_data: Any,
        caller_fn: Callable[[], tuple[Any, ProvenanceRecord]],
        mode: str | None = None,
        require_meta: bool | None = None,
    ) -> tuple[Any, ProvenanceRecord]:
        """Wrap an expensive model call with caching logic honoring mode.

        Modes:
        - 'live': return cached result if hit, otherwise call caller_fn and store.
        - 'cache_only': return cached result; raise KeyError if missing.
        - 'refresh': always call caller_fn and overwrite cache.
        """
        active_mode = (mode or settings.stepproof_mode or "live").lower().strip()
        key = make_cache_key(provider_name, model_name, prompt_version, input_data)

        if active_mode in ("live", "cache_only"):
            hit = self.get(key, require_meta=require_meta)
            if hit is not None:
                logger.debug("Cache hit for key %s (mode=%s)", key[:8], active_mode)
                return hit
            if active_mode == "cache_only":
                raise KeyError(
                    f"Cache miss in cache_only mode for key {key} "
                    f"(provider={provider_name}, model={model_name})"
                )

        # live on miss or refresh
        result, provenance = caller_fn()
        self.put(
            cache_key=key,
            provider=provider_name,
            model_name=model_name,
            prompt_version=prompt_version,
            input_data=input_data,
            response_data=result,
            provenance=provenance,
        )
        return result, provenance
