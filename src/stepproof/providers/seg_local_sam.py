"""Stub provider for local SAM 3.1 segmentation weights."""

from __future__ import annotations

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, SegmentationProvider
from stepproof.schemas import Evidence


class SegLocalSAMProvider(SegmentationProvider):
    """Local SAM 3.1 segmentation stub.

    TODO (Phase 2):
    1. Apply for gated weights at https://huggingface.co/facebook/sam3.1 or
       official Meta Research repository: https://github.com/facebookresearch/sam3
    2. Once access is approved, clone weights to a dedicated local directory
       (e.g. weights/sam3.1/). Do NOT commit weights to git.
    3. Install SAM 3 runtime dependencies:
       pip install git+https://github.com/facebookresearch/segment-anything-2.git
       or the SAM 3 package once published.
    4. Implement keyframe promptable inference using:
       predictor.set_image(image)
       masks, scores, logits = predictor.predict(text=phrase)
    5. Convert output masks to bounding box, area in pixels, and confidence.
    """

    def __init__(self, model_path: str = "facebookresearch/sam3.1"):
        self.model_path = model_path
        self.provenance = create_provenance(
            provider="local_sam",
            model_name=self.model_path,
            base_url="local",
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
        """Raise NotImplementedError with setup instructions."""
        raise NotImplementedError(
            "SegLocalSAMProvider is a stub for facebookresearch/sam3.1. "
            "SAM 3.1 weights are gated and require manual approval. "
            "To use local SAM 3.1: download authorized weights to weights/sam3.1, "
            "install the SAM 3 inference package, and implement the local inference call. "
            "For offline development and tests, set STEPPROOF_SEG=mock."
        )
