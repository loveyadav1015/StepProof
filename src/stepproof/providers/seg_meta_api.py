"""Meta Model API hosted segmentation provider (SAM 3.1)."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from meta_sam_parser import SegmentationMask
from meta_sam_parser._mask_codec import decode_mask_to_raster

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, SegmentationProvider
from stepproof.schemas import Evidence

logger = logging.getLogger(__name__)

BOX_REGEX = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([0-9\.]+))?\|>"
)
MASK_REGEX = re.compile(
    r"<\|mask;x=0;y=0(?:;c=([0-9\.]+))?;data=(\d+),(\d+),([~!]?[^|>]+)\|>"
)


def _image_to_base64_jpeg(image_input: ImageInput) -> tuple[str, np.ndarray]:
    """Convert ImageInput to base64 JPEG data and BGR numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_input}")
        with open(image_input, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, img
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from raw bytes")
        return base64.b64encode(image_input).decode("utf-8"), img
    if isinstance(image_input, np.ndarray):
        success, encoded = cv2.imencode(".jpg", image_input)
        if not success:
            raise ValueError("Failed to encode cv2 frame array to JPEG")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return b64, image_input
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def parse_sam_output_text(
    output_text: str,
    phrase: str,
    step_id: int | str = 0,
    frame_index: int = 0,
    timestamp_s: float = 0.0,
) -> list[Evidence]:
    """Parse SAM special token output string into Evidence models with decoded masks."""
    boxes = list(BOX_REGEX.finditer(output_text))
    masks = list(MASK_REGEX.finditer(output_text))

    evidence_list: list[Evidence] = []

    for b_m, m_m in zip(boxes, masks):
        x1, y1, x2, y2, fw, fh, bc = b_m.groups()
        mc, h, w, payload = m_m.groups()

        x1_i, y1_i = int(x1), int(y1)
        x2_i, y2_i = int(x2), int(y2)
        w_box = max(0, x2_i - x1_i)
        h_box = max(0, y2_i - y1_i)

        h_mask, w_mask = int(h), int(w)
        confidence = float(bc or mc or 0.98)

        # Decode mask raster using meta-sam-parser
        encoding = "lossless" if payload.startswith("~") else "one_bit"
        mask_obj = SegmentationMask(encoding, payload, w_mask, h_mask)
        try:
            raster = decode_mask_to_raster(mask_obj)
            mask_area_px = sum(1 for b in raster if b != 0)
        except Exception as e:
            logger.warning("Failed to decode SAM mask raster: %s", e)
            mask_area_px = w_box * h_box

        evidence_list.append(
            Evidence(
                step_id=step_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                object=phrase,
                count=1,
                mask_area_px=mask_area_px,
                bbox=[x1_i, y1_i, w_box, h_box],
                confidence=confidence,
            )
        )

    return evidence_list


class SegMetaAPIProvider(SegmentationProvider):
    """Hosted SAM 3.1 segmentation provider calling Meta Model API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "sam-3.1",
        base_url: str | None = None,
        out_dir: Path | str | None = None,
    ):
        self.api_key = api_key or os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY") or ""
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("META_API_BASE_URL", "https://api.meta.ai/v1")).rstrip("/")
        self.out_dir = Path(out_dir) if out_dir else None

        self.provenance = create_provenance(
            provider="meta_api",
            model_name=self.model_name,
            base_url=self.base_url,
            meta_compliant=True,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def _execute_sam_request(self, phrase: str, b64_img: str) -> str:
        """Call Meta Responses API for SAM 3.1 with retries and exponential backoff on 429/5xx."""
        if not self.api_key:
            raise ValueError(
                "Meta API key not configured. Set META_API_KEY or MODEL_API_KEY environment variable."
            )

        endpoint = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.model_name,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": phrase},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    ],
                }
            ],
            "stream": True,
        }

        max_attempts = 4
        delay = 1.0

        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                        if response.status_code in (429, 500, 502, 503, 504):
                            response.read()
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        response.raise_for_status()

                        output_chunks: list[str] = []
                        for line in response.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                event = json.loads(data_str)
                                if event.get("type") == "response.output_text.delta":
                                    output_chunks.append(event.get("delta", ""))
                            except json.JSONDecodeError:
                                continue

                        return "".join(output_chunks)

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning(
                    "SAM 3.1 request attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt,
                    max_attempts,
                    e,
                    delay,
                )
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"SAM 3.1 API request failed after {max_attempts} attempts: {e}"
                    ) from e
                time.sleep(delay)
                delay *= 2.0

        return ""

    def find(
        self,
        image: ImageInput,
        phrase: str,
        step_id: int | str = 0,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> tuple[list[Evidence], ProvenanceRecord]:
        """Ground and segment target object in image using hosted SAM 3.1."""
        b64_img, img_cv = _image_to_base64_jpeg(image)

        output_text = self._execute_sam_request(phrase=phrase, b64_img=b64_img)
        evidence_list = parse_sam_output_text(
            output_text=output_text,
            phrase=phrase,
            step_id=step_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )

        # Save visual overlay if output directory is configured
        if self.out_dir and evidence_list:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            overlay = img_cv.copy()
            for ev in evidence_list:
                x, y, w, h = ev.bbox
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 220, 40), 2)
                cv2.putText(
                    overlay,
                    f"{ev.object} ({ev.confidence:.2f})",
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (40, 220, 40),
                    1,
                    cv2.LINE_AA,
                )
            out_file = self.out_dir / f"step_{step_id}_sam_{frame_index}.jpg"
            cv2.imwrite(str(out_file), overlay)

        return evidence_list, self.provenance
