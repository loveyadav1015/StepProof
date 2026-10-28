"""Stub provider for Meta Model API hosted segmentation."""

from __future__ import annotations

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, SegmentationProvider
from stepproof.schemas import Evidence


class SegMetaAPIProvider(SegmentationProvider):
    """Meta Model API SAM 3.1 segmentation stub.

    TODO (Meta Model API integration):
    - Documentation: https://dev.meta.ai/docs/overview
    - Media segmentation guide: https://dev.meta.ai/models/sam-3-1
    - DO NOT guess endpoint paths or payload formats.
    - When API documentation is reviewed:
      1. Configure authorization using META_API_KEY.
      2. Upload or encode keyframe payload following the official Media segmentation guide.
      3. Call the verified endpoint (e.g. https://api.meta.ai/v1/media/segmentation).
      4. Parse returned polygons/masks into StepProof Evidence instances.
    """

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "sam-3.1",
        base_url: str = "https://api.meta.ai/v1",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.provenance = create_provenance(
            provider="meta_api",
            model_name=self.model_name,
            base_url=self.base_url,
            meta_compliant=True,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def find(
        self,
        image: ImageInput,
        phrase: str,
        step_id: int | str = 0,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[list[Evidence], ProvenanceRecord]:
        """Raise NotImplementedError pointing to official Meta developer documentation."""
        raise NotImplementedError(
            "SegMetaAPIProvider is a stub awaiting official documentation integration. "
            "Please refer to https://dev.meta.ai/docs/overview and the SAM 3.1 Media segmentation guide. "
            "Do not guess the payload shape. For offline development and tests, set STEPPROOF_SEG=mock."
        )
