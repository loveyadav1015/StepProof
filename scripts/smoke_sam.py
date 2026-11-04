#!/usr/bin/env python3
"""Smoke test for hosted SAM 3.1 (SegMetaAPIProvider).

Runs one real call when META_API_KEY or MODEL_API_KEY is present in environment.
If key is absent, prints clear instructions without failing.
Never prints or logs the API key.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from stepproof.providers.seg_meta_api import SegMetaAPIProvider  # noqa: E402


def get_test_image() -> np.ndarray:
    """Create a 320x240 image with a distinct red block for segmentation."""
    img = np.full((240, 320, 3), 245, dtype=np.uint8)
    # Red block (BGR: 30, 30, 220)
    cv2.rectangle(img, (70, 60), (170, 160), (30, 30, 220), -1)
    return img


def run_smoke_sam() -> int:
    api_key = os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY") or ""
    base_url = os.getenv("META_API_BASE_URL", "https://api.meta.ai/v1")
    api_style = os.getenv("SAM_API_STYLE", "responses").lower().strip()

    print("=" * 60)
    print("STEPPROOF HOSTED SAM 3.1 SMOKE TEST")
    print("=" * 60)
    print(f"Base URL:  {base_url}")
    print(f"API Style: {api_style} (responses stream or chat/completions)")
    print(f"API Key:   {'[CONFIGURED]' if api_key else '[NOT PRESENT]'}")

    if not api_key:
        print("\nℹ️  No META_API_KEY or MODEL_API_KEY detected in environment.")
        print("To run a live call against Meta's hosted SAM 3.1:")
        print("  export META_API_KEY=your_key_here")
        print("  export SAM_API_STYLE=responses  # or 'chat'")
        print("  python scripts/smoke_sam.py\n")
        print("Skipping live network call (no-op).")
        print("=" * 60)
        return 0

    out_dir = repo_root / "data" / "outputs" / "sam_smoke"
    provider = SegMetaAPIProvider(
        api_key=api_key,
        base_url=base_url,
        out_dir=out_dir,
        api_style=api_style,
    )

    img = get_test_image()
    phrase = "red square"
    print(f"\nSending image segmentation request for: '{phrase}' using style '{api_style}'...")

    t0 = time.time()
    try:
        evs, prov = provider.find(img, phrase)
        elapsed = time.time() - t0
        print(f"✅ SAM 3.1 request succeeded in {elapsed:.2f}s using style '{api_style}'!")
        print(f"Model:          {prov.model_name}")
        print(f"Meta-Compliant: {prov.meta_compliant}")
        print(f"Style Working:  {api_style}")
        print(f"Objects found:  {len(evs)}")
        for i, ev in enumerate(evs, 1):
            print(f"  [{i}] BBox: {ev.bbox}, Area: {ev.mask_area_px}px, Conf: {ev.confidence:.2f}")
        if out_dir.exists():
            print(f"Overlays saved in: {out_dir}")
        print("=" * 60)
        return 0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ SAM 3.1 attempt failed after {elapsed:.2f}s using primary style '{api_style}': {e}")
        alt_style = "chat" if api_style == "responses" else "responses"
        print(f"🔄 Retrying once automatically using alternative style '{alt_style}'...")

        provider_alt = SegMetaAPIProvider(
            api_key=api_key,
            base_url=base_url,
            out_dir=out_dir,
            api_style=alt_style,
        )
        t1 = time.time()
        try:
            evs_alt, prov_alt = provider_alt.find(img, phrase)
            elapsed_alt = time.time() - t1
            print(f"✅ Fallback to SAM 3.1 style '{alt_style}' SUCCEEDED in {elapsed_alt:.2f}s!")
            print(f"Model:          {prov_alt.model_name}")
            print(f"Meta-Compliant: {prov_alt.meta_compliant}")
            print(f"Style Working:  {alt_style}")
            print(f"Objects found:  {len(evs_alt)}")
            print("=" * 60)
            return 0
        except Exception as e2:
            elapsed_alt = time.time() - t1
            print(f"❌ SAM 3.1 retry with style '{alt_style}' also failed after {elapsed_alt:.2f}s: {e2}")
            print("=" * 60)
            return 1


if __name__ == "__main__":
    sys.exit(run_smoke_sam())
