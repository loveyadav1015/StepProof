"""Video quality diagnostic and doctor tool for real-world footage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def check_video_quality(video_path: Path | str) -> dict[str, Any]:
    """Inspect video file for duration, resolution, fps, camera shake, and exposure drift."""
    v_path = Path(video_path)
    if not v_path.exists():
        return {
            "exists": False,
            "path": str(v_path),
            "warnings": [f"File does not exist: {v_path}"],
        }

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        return {
            "exists": False,
            "path": str(v_path),
            "warnings": [f"OpenCV could not open video: {v_path}"],
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    warnings: list[str] = []
    advice: list[str] = []

    if width < 640 or height < 360:
        warnings.append(f"Low resolution: {width}x{height} (recommend >= 1280x720)")
        advice.append("Record in at least 720p or 1080p for reliable component segmentation.")

    if fps < 10 or fps > 60:
        warnings.append(f"Unusual frame rate: {fps:.1f} fps")

    # Sample up to 60 evenly spaced frames to measure motion and exposure
    sample_count = min(60, total_frames)
    indices = np.linspace(0, max(0, total_frames - 1), sample_count, dtype=int)

    mean_brightness = []
    frame_diffs = []
    prev_gray = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness.append(float(np.mean(gray)))

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            frame_diffs.append(float(np.mean(diff) / 255.0))
        prev_gray = gray

    cap.release()

    # Analyze exposure drift
    exposure_std = float(np.std(mean_brightness)) if mean_brightness else 0.0
    if exposure_std > 20.0:
        warnings.append(f"High exposure drift detected (std={exposure_std:.1f})")
        advice.append("Lock exposure/white balance on your camera and use constant room lighting.")

    # Analyze camera motion / jitter
    avg_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
    max_diff = float(np.max(frame_diffs)) if frame_diffs else 0.0
    if avg_diff > 0.15 or max_diff > 0.35:
        warnings.append(f"Noticeable camera motion / shake detected (avg_diff={avg_diff:.2f}, max={max_diff:.2f})")
        advice.append("Mount camera on a rigid tripod/stand. A stationary view is critical for spatial bounding boxes.")

    return {
        "exists": True,
        "path": str(v_path),
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "exposure_std": round(exposure_std, 2),
        "avg_frame_diff": round(avg_diff, 3),
        "warnings": warnings,
        "advice": list(set(advice)),
    }


def run_doctor(manifest_path: Path | str) -> dict[str, Any]:
    """Run doctor diagnostic on all videos defined in a manifest YAML."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    runs = data.get("runs", [])
    results = []

    print("=" * 70)
    print(f"🩺 STEPPROOF DOCTOR: Quality Check for {m_path.name}")
    print("=" * 70)

    for run_info in runs:
        scenario = run_info.get("scenario", "Unknown")
        video_file = run_info.get("video", "")
        expected_status = run_info.get("status", "ready")
        v_path = Path(video_file)

        print(f"\nChecking [{scenario}] -> {video_file}...")
        diag = check_video_quality(v_path)
        diag["scenario"] = scenario
        diag["status"] = expected_status

        if not diag["exists"]:
            print(f"  ⚠️  File status: MISSING (status in manifest: {expected_status})")
        else:
            print(f"  Resolution: {diag['width']}x{diag['height']} @ {diag['fps']} fps | Duration: {diag['duration_s']}s")
            print(f"  Exposure std: {diag['exposure_std']} | Camera motion diff: {diag['avg_frame_diff']}")
            if diag["warnings"]:
                for w in diag["warnings"]:
                    print(f"  ⚠️  Warning: {w}")
                for a in diag["advice"]:
                    print(f"     💡 Tip: {a}")
            else:
                print("  ✅ Quality check passed cleanly! Video is steady and well-exposed.")

        results.append(diag)

    print("\n" + "=" * 70)
    print("🩺 Doctor check complete.")
    print("=" * 70 + "\n")
    return {"manifest": str(m_path), "reports": results}
