"""Provenance tracking and Meta compliance verification."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# Canonical set of recognized Meta model identifiers and families
META_MODEL_PREFIXES = (
    "sam-3",
    "sam3",
    "muse-glimmer",
    "muse-spark",
    "muse-wonder",
    "muse-image",
    "muse-code",
    "meta/muse-glimmer",
    "meta/muse-spark",
    "facebookresearch/sam3",
)

META_HOSTS = (
    "api.meta.ai",
    "dev.meta.ai",
)


def evaluate_meta_compliance(
    provider: str,
    model_name: str,
    base_url_host: str = "",
) -> bool:
    """Evaluate whether a provider/model combination qualifies as a Meta model.

    Compliant models include:
    - SAM 3.1 (via local weights or official Meta Model API)
    - Muse Glimmer / Muse Spark / Muse Wonder
    - Any model hosted directly on api.meta.ai

    Free dev stand-ins (Ollama Qwen, Groq Llama text, Mocks, etc.) are NOT meta_compliant.
    """
    provider_clean = (provider or "").lower().strip()
    model_clean = (model_name or "").lower().strip()
    host_clean = (base_url_host or "").lower().strip()

    # Mocks are strictly dev stand-ins
    if provider_clean in ("mock", "dev"):
        return False

    # Official Meta Model API host
    if any(mh in host_clean for mh in META_HOSTS):
        return True

    # Check for known Meta model prefixes (e.g. Muse family, SAM 3.1)
    if any(model_clean.startswith(prefix) or prefix in model_clean for prefix in META_MODEL_PREFIXES):
        # Local SAM or Meta API provider
        if provider_clean in ("local_sam", "meta_api", "openai_compat"):
            return True

    return False


class ProvenanceRecord(BaseModel):
    """Metadata tracking the origin, runtime, and compliance status of model inference."""

    provider: str = Field(description="Provider identifier (e.g. mock, openai_compat, local_sam, meta_api)")
    model_name: str = Field(description="Name or path of model used")
    base_url_host: str = Field(default="localhost", description="Hostname of API endpoint or 'local'")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of execution",
    )
    meta_compliant: bool = Field(
        default=False,
        description="True ONLY for Meta Model API, Muse Open weights, and SAM 3.1",
    )

    @property
    def badge(self) -> str:
        """Return user-facing compliance badge."""
        return "Meta model" if self.meta_compliant else "Dev stand-in"


def create_provenance(
    provider: str,
    model_name: str,
    base_url: str = "",
    meta_compliant: bool | None = None,
) -> ProvenanceRecord:
    """Factory helper to create a verified ProvenanceRecord."""
    host = "local"
    if base_url:
        try:
            parsed = urlparse(base_url)
            host = parsed.netloc or parsed.path or "local"
        except Exception:
            host = base_url

    if meta_compliant is None:
        meta_compliant = evaluate_meta_compliance(
            provider=provider,
            model_name=model_name,
            base_url_host=host,
        )

    return ProvenanceRecord(
        provider=provider,
        model_name=model_name,
        base_url_host=host,
        timestamp=datetime.now(timezone.utc).isoformat(),
        meta_compliant=meta_compliant,
    )


def assert_meta_compliant(record: ProvenanceRecord) -> None:
    """Raise RuntimeError if record does not satisfy Meta compliance."""
    if not record.meta_compliant:
        raise RuntimeError(
            f"Meta Compliance Guard: Provider '{record.provider}' with model '{record.model_name}' "
            f"on '{record.base_url_host}' is marked as a Dev stand-in (meta_compliant=False). "
            f"STEPPROOF_REQUIRE_META=1 refuses non-Meta models."
        )
