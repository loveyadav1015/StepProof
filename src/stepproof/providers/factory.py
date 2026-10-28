"""Factory for instantiating LLM and Segmentation providers with Meta compliance enforcement."""

from __future__ import annotations

import logging
import os

from stepproof.config import settings
from stepproof.provenance import assert_meta_compliant
from stepproof.providers.base import LLMProvider, SegmentationProvider
from stepproof.providers.llm_mock import LLMMockProvider
from stepproof.providers.llm_openai_compat import LLMOpenAICompatProvider
from stepproof.providers.seg_local_sam import SegLocalSAMProvider
from stepproof.providers.seg_meta_api import SegMetaAPIProvider
from stepproof.providers.seg_mock import SegMockProvider

logger = logging.getLogger(__name__)


def get_llm_provider(
    provider_name: str | None = None,
    require_meta: bool | None = None,
) -> LLMProvider:
    """Instantiate and return the configured LLM provider.

    Args:
        provider_name: 'mock' or 'openai_compat'. Defaults to settings.stepproof_llm.
        require_meta: If True, raises RuntimeError if provider is not meta-compliant.
                      Defaults to settings.stepproof_require_meta.

    Raises:
        ValueError: On unknown provider name.
        RuntimeError: If require_meta is True and provider is not meta_compliant.
    """
    selected = (provider_name or settings.stepproof_llm or "mock").lower().strip()
    must_be_meta = (
        require_meta
        if require_meta is not None
        else (settings.stepproof_require_meta or os.getenv("STEPPROOF_REQUIRE_META", "0") in ("1", "true"))
    )

    provider: LLMProvider
    if selected == "mock":
        provider = LLMMockProvider()
    elif selected in ("openai_compat", "openai"):
        provider = LLMOpenAICompatProvider()
    else:
        raise ValueError(f"Unknown STEPPROOF_LLM provider: '{selected}'. Options: mock, openai_compat")

    prov_record = provider.get_provenance()
    if must_be_meta:
        assert_meta_compliant(prov_record)

    logger.info("Initialized LLMProvider: %s (meta_compliant=%s)", selected, prov_record.meta_compliant)
    return provider


def get_seg_provider(
    provider_name: str | None = None,
    require_meta: bool | None = None,
) -> SegmentationProvider:
    """Instantiate and return the configured Segmentation provider.

    Args:
        provider_name: 'mock', 'local_sam', or 'meta_api'. Defaults to settings.stepproof_seg.
        require_meta: If True, raises RuntimeError if provider is not meta-compliant.
                      Defaults to settings.stepproof_require_meta.

    Raises:
        ValueError: On unknown provider name.
        RuntimeError: If require_meta is True and provider is not meta_compliant.
    """
    selected = (provider_name or settings.stepproof_seg or "mock").lower().strip()
    must_be_meta = (
        require_meta
        if require_meta is not None
        else (settings.stepproof_require_meta or os.getenv("STEPPROOF_REQUIRE_META", "0") in ("1", "true"))
    )

    provider: SegmentationProvider
    if selected == "mock":
        provider = SegMockProvider()
    elif selected in ("local_sam", "sam_local", "sam3"):
        provider = SegLocalSAMProvider()
    elif selected in ("meta_api", "sam_meta_api"):
        provider = SegMetaAPIProvider()
    else:
        raise ValueError(
            f"Unknown STEPPROOF_SEG provider: '{selected}'. Options: mock, local_sam, meta_api"
        )

    prov_record = provider.get_provenance()
    if must_be_meta:
        assert_meta_compliant(prov_record)

    logger.info("Initialized SegmentationProvider: %s (meta_compliant=%s)", selected, prov_record.meta_compliant)
    return provider
