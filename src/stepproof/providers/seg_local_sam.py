"""Local SAM 3.1 segmentation provider for image mode (requires weights in weights/sam3.1/)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, SegmentationProvider
from stepproof.schemas import Evidence

logger = logging.getLogger(__name__)


def _load_image_bgr(image_input: ImageInput) -> np.ndarray:
    """Load ImageInput into a BGR numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_input}")
        return img
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from raw bytes")
        return img
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


class SegLocalSAMProvider(SegmentationProvider):
    """Local SAM 3.1 segmentation provider using facebookresearch/sam3.

    Runs in image mode with fp16 precision and batch size 1 to target ~4 GB VRAM.
    Detects missing weights at weights/sam3.1/ and falls back to a clear error.
    """

    def __init__(
        self,
        weights_dir: Path | str = "weights/sam3.1",
        model_name: str = "facebookresearch/sam3.1",
        device: str | None = None,
    ):
        self.weights_dir = Path(weights_dir)
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._predictor: Any = None

        self.provenance = create_provenance(
            provider="local_sam",
            model_name=self.model_name,
            base_url="local",
            meta_compliant=True,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def _has_weights(self) -> bool:
        """Check if SAM 3.1 weight files exist in weights directory."""
        if not self.weights_dir.exists() or not self.weights_dir.is_dir():
            return False
        weight_patterns = ["*.pt", "*.pth", "*.bin", "*.safetensors", "*.checkpoint"]
        for pattern in weight_patterns:
            if any(self.weights_dir.glob(pattern)):
                return True
        return False

    def _ensure_model_loaded(self) -> None:
        """Verify weights presence and initialize local SAM 3.1 model in fp16."""
        if not self._has_weights():
            raise FileNotFoundError(
                f"SAM 3.1 local weights not found in '{self.weights_dir}'. "
                "SAM 3.1 weights are gated and require manual approval from Meta. "
                "Once approved, place the model weights in weights/sam3.1/ to use local_sam. "
                "For offline development and tests without weights, set STEPPROOF_SEG=mock."
            )

        if self._predictor is not None:
            return

        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for local SAM 3.1 execution. Run 'pip install torch'."
            ) from e

        # Target about 4 GB VRAM with FP16 and batch size 1
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev == "cuda" else torch.float32

        logger.info(
            "Loading local SAM 3.1 from %s on %s (%s, batch_size=1)...",
            self.weights_dir,
            dev,
            dtype,
        )

        try:
            # Check for sam3 or segment_anything packages
            import sam3  # type: ignore

            self._model = sam3.build_sam(checkpoint_dir=str(self.weights_dir))
            self._model.to(device=dev, dtype=dtype)
            self._model.eval()
            self._predictor = sam3.SAM3Predictor(self._model)
        except ImportError:
            # Fallback for sam2/sam-based wrappers
            try:
                from sam2.build_sam import build_sam2  # type: ignore
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

                chk = next(self.weights_dir.glob("*.pt*"))
                self._model = build_sam2(model_cfg="sam2_hiera_l.yaml", ckpt_path=str(chk))
                self._model.to(device=dev, dtype=dtype)
                self._predictor = SAM2ImagePredictor(self._model)
            except Exception as ex:
                raise RuntimeError(
                    f"Found weights in '{self.weights_dir}', but failed to import local SAM 3 library: {ex}. "
                    "Please install facebookresearch/sam3 per official repository instructions."
                ) from ex

    def find(
        self,
        image: ImageInput,
        phrase: str,
        step_id: int | str = 0,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[list[Evidence], ProvenanceRecord]:
        """Perform local promptable image segmentation using SAM 3.1."""
        self._ensure_model_loaded()

        img_bgr = _load_image_bgr(image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self._predictor.set_image(img_rgb)
        # Prompt model with phrase
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            text=phrase,
            multimask_output=True,
        )

        evidence_list: list[Evidence] = []
        if masks is not None and len(masks) > 0:
            for i, mask in enumerate(masks):
                score = float(scores[i]) if scores is not None and len(scores) > i else 0.95
                mask_uint8 = (mask > 0).astype(np.uint8)
                area_px = int(np.sum(mask_uint8))
                if area_px <= 0:
                    continue

                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                else:
                    x, y, w, h = 0, 0, 0, 0

                evidence_list.append(
                    Evidence(
                        step_id=step_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        object=phrase,
                        count=1,
                        mask_area_px=area_px,
                        bbox=[int(x), int(y), int(w), int(h)],
                        confidence=score,
                    )
                )

        return evidence_list, self.provenance
