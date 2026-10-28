"""OpenAI-compatible LLM provider working with Ollama, OpenRouter, Groq, and Meta Model API."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import httpx
import numpy as np
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from stepproof.config import settings
from stepproof.json_repair import repair_json
from stepproof.pricing import calculate_cost
from stepproof.provenance import ProvenanceRecord, create_provenance
from stepproof.providers.base import ImageInput, LLMProvider
from stepproof.schemas import SOP, Evidence, Step, validate_with_retry

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

        self.request_count: int = 0
        self.total_cost_usd: float = 0.0

        self.provenance = create_provenance(
            provider="openai_compat",
            model_name=self.model,
            base_url=self.base_url,
        )

    def get_provenance(self) -> ProvenanceRecord:
        return self.provenance

    def _check_budget(self) -> None:
        """Enforce request count and cost budget limits before making API call."""
        if self.request_count >= settings.max_requests_per_run:
            raise RuntimeError(
                f"Request budget exceeded: MAX_REQUESTS_PER_RUN limit reached ({settings.max_requests_per_run})"
            )
        if self.total_cost_usd >= settings.max_cost_usd:
            raise RuntimeError(
                f"Cost budget exceeded: MAX_COST_USD limit reached (${settings.max_cost_usd:.2f})"
            )

    def _execute_with_retry(self, call_fn: Callable[[], tuple[str, dict[str, int]]]) -> str:
        """Execute API request with budget check and exponential backoff retry on 429/5xx."""
        self._check_budget()

        max_attempts = 3
        delay = 1.0

        for attempt in range(1, max_attempts + 1):
            try:
                raw_text, usage = call_fn()
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                cost = calculate_cost(self.model, in_tok, out_tok, settings.prices_path)

                self.request_count += 1
                self.total_cost_usd += cost

                self.provenance.input_tokens = in_tok
                self.provenance.output_tokens = out_tok
                self.provenance.total_tokens = in_tok + out_tok
                self.provenance.cost_usd = cost

                return raw_text

            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                logger.warning(
                    "LLM request error (%s) on attempt %d/%d: %s. Backing off %.1fs...",
                    type(e).__name__,
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay *= 2.0
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "LLM HTTP %d on attempt %d/%d. Backing off %.1fs...",
                        e.response.status_code,
                        attempt,
                        max_attempts,
                    )
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise

        return "{}"

    def _call_completion(self, messages: list[dict[str, Any]], json_mode: bool = True) -> str:
        """Execute request using either chat completions or responses API."""
        def _invoke() -> tuple[str, dict[str, int]]:
            if self.api_style == "responses" and hasattr(self.client, "responses"):
                # Meta Model API Responses endpoint
                instructions = None
                input_items: list[dict[str, Any]] = []

                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        instructions = content if isinstance(content, str) else json.dumps(content)
                        continue

                    if isinstance(content, str):
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": [{"type": "input_text", "text": content}],
                        })
                    elif isinstance(content, list):
                        part_items = []
                        for part in content:
                            p_type = part.get("type")
                            if p_type == "text":
                                part_items.append({"type": "input_text", "text": part.get("text", "")})
                            elif p_type == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                part_items.append({"type": "input_image", "image_url": url})
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": part_items,
                        })

                # If single simple text message, input can be a string
                resp_input: Any = input_items
                if len(input_items) == 1 and len(input_items[0].get("content", [])) == 1:
                    first_part = input_items[0]["content"][0]
                    if first_part.get("type") == "input_text":
                        resp_input = first_part.get("text", "")

                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": resp_input,
                }
                if instructions:
                    kwargs["instructions"] = instructions
                if self.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if json_mode:
                    kwargs["text"] = {"format": {"type": "json_object"}}

                resp = self.client.responses.create(**kwargs)
                out_text = getattr(resp, "output_text", "") or ""
                usage_obj = getattr(resp, "usage", None)
                usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
                    "output_tokens": getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
                    "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
                }
                return out_text, usage

            # Default standard /v1/chat/completions
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            out_text = choice.message.content or "{}"
            usage_obj = getattr(resp, "usage", None)
            usage = {
                "input_tokens": getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
                "output_tokens": getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
            }
            return out_text, usage

        return self._execute_with_retry(_invoke)

    def complete(
        self,
        prompt: str,
        images: list[ImageInput] | None = None,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, int], ProvenanceRecord]:
        """Send completion request (text only or text+images) and return text, token usage, provenance."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if images:
            # Enforce max images limit per request
            limited_images = images[: settings.max_images_per_request]
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in limited_images:
                b64 = _image_to_base64_jpeg(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        raw_text = self._call_completion(messages, json_mode=json_mode)
        usage = {
            "input_tokens": self.provenance.input_tokens,
            "output_tokens": self.provenance.output_tokens,
            "total_tokens": self.provenance.total_tokens,
        }
        return raw_text, usage, self.provenance

    def generate_sop(
        self,
        frames: list[ImageInput],
        prompt: str,
        video_hash: str = "unknown",
    ) -> tuple[SOP, ProvenanceRecord]:
        """Generate structured SOP from video frames with JSON schema validation and retry."""
        limited_frames = frames[: settings.max_images_per_request]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for frame in limited_frames:
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
            raw_text = self._call_completion(messages, json_mode=True)
            repaired = repair_json(raw_text)
            data = json.loads(repaired)
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
            raw_text = self._call_completion(messages, json_mode=True)
            repaired = repair_json(raw_text)
            parsed = json.loads(repaired)
            if not isinstance(parsed, dict) or "passed" not in parsed:
                raise ValueError(f"Malformed judgment response: {raw_text}")
            return parsed

        verdict_dict = validate_with_retry(dict, _fetch_judgment, max_retries=1)  # type: ignore
        return verdict_dict, self.provenance
