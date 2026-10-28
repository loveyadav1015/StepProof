#!/usr/bin/env python3
"""Meta API Smoke Test: runs one text request and one image request.

Reads configuration strictly from environment variables:
- LLM_BASE_URL (e.g. https://api.meta.ai/v1 or http://localhost:11434/v1)
- LLM_MODEL (e.g. muse-wonder-1.0, muse-spark-1.3, meta/muse-glimmer-30b)
- LLM_API_KEY
- LLM_API_STYLE (chat or responses)
- LLM_TEMPERATURE
- LLM_REASONING_EFFORT

Never logs or prints the API key.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure src is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from stepproof.config import settings
from stepproof.providers.llm_openai_compat import LLMOpenAICompatProvider


def get_test_image() -> np.ndarray:
    """Generate a clean 320x240 test image containing simple visual geometry."""
    img = np.full((240, 320, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (120, 120), (30, 30, 220), -1)  # Red square
    cv2.putText(
        img,
        "StepProof Test",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return img


def run_smoke_test() -> int:
    base_url = os.getenv("LLM_BASE_URL", settings.llm_base_url)
    model = os.getenv("LLM_MODEL", settings.llm_model)
    api_style = os.getenv("LLM_API_STYLE", settings.llm_api_style).lower().strip()
    api_key_set = bool(os.getenv("LLM_API_KEY", settings.llm_api_key))

    print("=" * 60)
    print("STEPPROOF META API SMOKE TEST")
    print("=" * 60)
    print(f"Base URL:   {base_url}")
    print(f"Model ID:   {model}")
    print(f"API Style:  {api_style}")
    print(f"API Key:    {'[CONFIGURED]' if api_key_set else '[NOT SET / USING DEFAULT]'}")
    print("=" * 60)

    try:
        provider = LLMOpenAICompatProvider(
            base_url=base_url,
            model=model,
            api_style=api_style,
        )
    except Exception as e:
        print(f"❌ Failed to initialize LLM provider: {e}")
        return 1

    # --- Call (a): One text request ---
    print("\n--- [1/2] Text Request ---")
    text_prompt = "What are the two primary safety rules when assembling a small breadboard LED circuit? Answer in two brief bullet points."
    print(f"Prompt: {text_prompt}")

    t0 = time.time()
    try:
        text_resp, text_usage, text_prov = provider.complete(prompt=text_prompt, json_mode=False)
        t_text = time.time() - t0
        print(f"✅ Text request succeeded in {t_text:.2f}s")
        print(f"Model:        {text_prov.model_name}")
        print(f"Latency:      {t_text:.2f}s")
        print(f"Token Usage:  input={text_usage.get('input_tokens', 0)}, output={text_usage.get('output_tokens', 0)}, total={text_usage.get('total_tokens', 0)}")
        print(f"Meta-Compliant: {text_prov.meta_compliant}")
        print(f"Response:\n{text_resp.strip()}")
    except Exception as e:
        t_text = time.time() - t0
        print(f"❌ Text request failed after {t_text:.2f}s: {e}")
        return 1

    # --- Call (b): One request with a single image ---
    print("\n--- [2/2] Single Image Request ---")
    test_image = get_test_image()
    image_prompt = "Describe the colored shape visible in this image in one sentence."
    print(f"Prompt: {image_prompt}")

    t0 = time.time()
    try:
        img_resp, img_usage, img_prov = provider.complete(
            prompt=image_prompt,
            images=[test_image],
            json_mode=False,
        )
        t_img = time.time() - t0
        print(f"✅ Image request succeeded in {t_img:.2f}s")
        print(f"Model:        {img_prov.model_name}")
        print(f"Latency:      {t_img:.2f}s")
        print(f"Token Usage:  input={img_usage.get('input_tokens', 0)}, output={img_usage.get('output_tokens', 0)}, total={img_usage.get('total_tokens', 0)}")
        print(f"Meta-Compliant: {img_prov.meta_compliant}")
        print(f"Response:\n{img_resp.strip()}")
    except Exception as e:
        t_img = time.time() - t0
        print(f"❌ Image request failed after {t_img:.2f}s: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✅ Meta API smoke test completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
