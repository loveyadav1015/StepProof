"""OpenAI-compatible LLM provider working with Ollama, OpenRouter, Groq, and Meta Model API."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
import cv2
import httpx
import numpy as np
from openai import OpenAI

from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, LLMProvider
from stepproof.schemas import Evidence, SOP, Step, validate_with_retry

logger = logging.getLogger(__name__)


def _image_to_base64_jpeg(image_input: ImageInput) -> str:
    """Encode an image path, bytes, or numpy BGR array to a base64 JPEG string."""
    if isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if isinstance(image_input, bytes):
        return base64.b64encode(image_input).decode("utf-8")
    if isinstance(image_input, np.ndarray):
        success, encoded = cv2.imencode(".jpg", image_input)
        if not success:
            raise ValueError("Failed to encode cv2 frame array to JPEG")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


class LLMOpenAICompatProvider(LLMProvider):
    """Generic OpenAI-compatible client configured strictly via environment variables."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_style: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ):
        # Configured strictly by env vars with optional constructor overrides
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.getenv("LLM_MODEL", "meta/muse-glimmer-30b")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "not-needed")
        self.api_style = (api_style or os.getenv("LLM_API_STYLE", "chat")).lower().strip()

        temp_env = os.getenv("LLM_TEMPERATURE")
        self.temperature = (
            temperature
            if temperature is not None
            else (float(temp_env) if temp_env else None)
        )
        self.reasoning_effort = reasoning_effort or os.getenv("LLM_REASONING_EFFORT") or None

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(timeout=90.0),
        )

        self.provenance = create_provenance(
            provider="openai_compat",
            model_name=self.model,
            base_url=self.base_url,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def _call_completion(self, messages: list[dict[str, Any]]) -> str:
        """Call the API requesting JSON response output."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self.api_style == "responses" and hasattr(self.client, "responses"):
            # If responses API endpoint is available
            resp = self.client.responses.create(**kwargs)  # type: ignore
            return resp.output_text or ""

        # Default standard /v1/chat/completions
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return choice.message.content or "{}"

    def generate_sop(
        self,
        frames: list[ImageInput],
        prompt: str,
        video_hash: str = "unknown",
    ) -> tuple[SOP, ProvenanceRecord]:
        """Generate structured SOP from video frames with JSON schema validation and retry."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for frame in frames:
            b64_jpeg = _image_to_base64_jpeg(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert industrial engineering AI. Analyze keyframes of an expert "
                    "demonstration and generate a structured SOP adhering strictly to the JSON schema."
                ),
            },
            {"role": "user", "content": content},
        ]

        def _fetch_and_parse() -> SOP:
            raw_text = self._call_completion(messages)
            data = json.loads(raw_text)
            # Ensure source_video_hash and created_with are present
            data["source_video_hash"] = video_hash
            data["created_with"] = self.provenance.model_dump()
            return SOP.model_validate(data)

        sop = validate_with_retry(SOP, _fetch_and_parse, max_retries=1)
        return sop, self.provenance

    def judge_step(
        self,
        frame: ImageInput,
        step: Step,
        evidence: list[Evidence],
        learner_action_desc: str = "",
    ) -> tuple[dict[str, Any], ProvenanceRecord]:
        """Ask LLM to judge whether step is complete, incomplete, or wrong."""
        b64_jpeg = _image_to_base64_jpeg(frame)
        evidence_summary = [
            {
                "object": ev.object,
                "count": ev.count,
                "bbox": ev.bbox,
                "mask_area_px": ev.mask_area_px,
                "confidence": ev.confidence,
            }
            for ev in evidence
        ]

        judge_prompt = (
            f"Step Specification:\n"
            f"- Step ID: {step.id}\n"
            f"- Instruction: {step.instruction}\n"
            f"- Expected End State: {step.expected_end_state}\n"
            f"- Objects: {', '.join(step.objects)}\n"
            f"- Grounding Evidence: {json.dumps(evidence_summary)}\n"
            f"{'- Learner Action: ' + learner_action_desc if learner_action_desc else ''}\n\n"
            f"Evaluate if this step has been completed correctly according to the end state.\n"
            f"Return JSON strictly with the format:\n"
            f'{{"passed": true|false, "reason": "concise explanation", "confidence": float 0.0-1.0, "failed_rules": ["rule description"]}}'
        )

        messages = [
            {
                "role": "system",
                "content": "You are a quality assurance judge reviewing step completion from video evidence.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": judge_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                    },
                ],
            },
        ]

        def _fetch_judgment() -> dict[str, Any]:
            raw_text = self._call_completion(messages)
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict) or "passed" not in parsed:
                raise ValueError(f"Malformed judgment response: {raw_text}")
            return parsed

        verdict_dict = validate_with_retry(dict, _fetch_judgment, max_retries=1)  # type: ignore
        return verdict_dict, self.provenance
