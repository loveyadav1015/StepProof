"""Configurable HSV color range segmentation provider for development."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, SegmentationProvider
from stepproof.schemas import Evidence

logger = logging.getLogger(__name__)


def load_color_config(config_path: str | Path = "config/colors.yaml") -> dict[str, Any]:
    """Load HSV color boundaries from YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("objects", {}) if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load color config from %s: %s", path, e)
        return {}


class SegDevColorProvider(SegmentationProvider):
    """HSV color-range segmenter for dev testing on real footage.

    Marked strictly as a Dev stand-in (meta_compliant=False).
    """

    def __init__(
        self,
        config_path: str | Path = "config/colors.yaml",
        min_area_px: int = 100,
    ):
        self.config_path = Path(config_path)
        self.min_area_px = min_area_px
        self.color_defs = load_color_config(self.config_path)
        self.provenance = create_provenance(
            provider="dev_color",
            model_name="hsv-color-segmenter-dev",
            base_url="local",
            meta_compliant=False,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def _get_image_bgr(self, image_input: ImageInput) -> np.ndarray:
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
        if isinstance(image_input, np.ndarray):
            return image_input
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def find(
        self,
        image: ImageInput,
        phrase: str,
        step_id: int | str = 0,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[list[Evidence], ProvenanceRecord]:
        """Locate target objects in frame using calibrated HSV ranges."""
        img = self._get_image_bgr(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        phrase_clean = phrase.lower().strip()

        matched_def = None
        for obj_key, obj_val in self.color_defs.items():
            if obj_key.lower() in phrase_clean or phrase_clean in obj_key.lower():
                matched_def = obj_val
                break

        if not matched_def:
            # Fallback: check if standard colors are in phrase
            return [], self.provenance

        ranges = matched_def.get("ranges", [])
        min_area = int(matched_def.get("min_area_px", self.min_area_px))

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            lower = np.array(r[0], dtype=np.uint8)
            upper = np.array(r[1], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        evidence_list: list[Evidence] = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            evidence_list.append(
                Evidence(
                    step_id=step_id,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    object=phrase,
                    count=1,
                    mask_area_px=area,
                    bbox=[int(x), int(y), int(w), int(h)],
                    confidence=0.85,
                )
            )

        return evidence_list, self.provenance
