"""OpenCV-based color and contour segmentation mock for synthetic video testing."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, SegmentationProvider
from stepproof.schemas import Evidence


class SegMockProvider(SegmentationProvider):
    """Detects colored objects via HSV thresholding and contour extraction without neural models."""

    # HSV color boundaries for synthetic demonstration blocks
    COLOR_RANGES = {
        "red": [
            (np.array([0, 80, 50]), np.array([10, 255, 255])),
            (np.array([170, 80, 50]), np.array([180, 255, 255])),
        ],
        "green": [
            (np.array([35, 80, 50]), np.array([85, 255, 255])),
        ],
        "blue": [
            (np.array([100, 80, 50]), np.array([135, 255, 255])),
        ],
        "yellow": [
            (np.array([20, 80, 50]), np.array([35, 255, 255])),
        ],
        "purple": [
            (np.array([135, 80, 50]), np.array([165, 255, 255])),
        ],
        "orange": [
            (np.array([11, 80, 50]), np.array([19, 255, 255])),
        ],
    }

    def __init__(self, min_area_px: int = 200):
        self.min_area_px = min_area_px
        self.provenance = create_provenance(
            provider="mock",
            model_name="opencv-color-contour-mock",
            base_url="local",
            meta_compliant=False,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def _load_image(self, image_input: ImageInput) -> np.ndarray:
        """Helper to ensure image is a BGR numpy array."""
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

    def find(
        self,
        image: ImageInput,
        phrase: str,
        step_id: int | str = 0,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[list[Evidence], ProvenanceRecord]:
        """Find objects matching noun phrase (e.g. 'red block', 'green block')."""
        img = self._load_image(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        phrase_lower = phrase.lower().strip()

        matched_color = None
        for color_name in self.COLOR_RANGES:
            if color_name in phrase_lower:
                matched_color = color_name
                break

        results: list[Evidence] = []

        if not matched_color:
            # Color not recognized by mock segmenter, return empty list
            return results, self.provenance

        ranges = self.COLOR_RANGES[matched_color]
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area >= self.min_area_px:
                x, y, w, h = cv2.boundingRect(cnt)
                results.append(
                    Evidence(
                        step_id=step_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        object=phrase,
                        count=1,
                        mask_area_px=area,
                        bbox=[int(x), int(y), int(w), int(h)],
                        confidence=0.98,
                    )
                )

        return results, self.provenance
