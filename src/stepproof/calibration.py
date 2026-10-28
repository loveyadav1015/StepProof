"""Per-object calibration loader and statistical threshold profiling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from stepproof.config import settings
from stepproof.ingest.sampler import sample_video
from stepproof.providers.base import SegmentationProvider
from stepproof.schemas import SOP

logger = logging.getLogger(__name__)


def load_calibration_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load calibration thresholds from YAML configuration."""
    path = Path(config_path or settings.calibration_path)
    if not path.exists():
        return {
            "default": {
                "min_mask_area_px": 200,
                "min_confidence": 0.50,
                "expected_count": 1,
                "position_tolerance_px": 20,
            },
            "objects": {},
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not read calibration file at %s: %s", path, e)
        return {"default": {}, "objects": {}}


def get_object_thresholds(object_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrieve thresholds for a specific object with fallback to default."""
    cfg = config or load_calibration_config()
    default_vals = cfg.get("default", {
        "min_mask_area_px": 200,
        "min_confidence": 0.50,
        "expected_count": 1,
        "position_tolerance_px": 20,
    })

    objects_map = cfg.get("objects", {})
    obj_clean = object_name.lower().strip()

    # Exact or substring match
    for name, thresholds in objects_map.items():
        if name.lower().strip() in obj_clean or obj_clean in name.lower().strip():
            merged = dict(default_vals)
            merged.update(thresholds)
            return merged

    return default_vals


def profile_video_evidence(
    video_path: Path | str,
    sop: SOP,
    segmenter: SegmentationProvider,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Sample video and compute per-step object evidence statistics for calibration."""
    v_path = Path(video_path)
    frames_dir = Path(output_dir or v_path.parent / "calibration_frames")
    sampled = sample_video(v_path, output_dir=frames_dir)

    stats: dict[str, Any] = {}

    for step in sop.steps:
        step_id_str = str(step.id)
        stats[step_id_str] = {
            "instruction": step.instruction,
            "objects": {},
        }

        for obj in step.objects:
            areas: list[int] = []
            confidences: list[float] = []
            frames_detected = 0

            for frame in sampled.frames:
                evs, _ = segmenter.find(
                    image=frame.file_path,
                    phrase=obj,
                    step_id=step.id,
                    frame_index=frame.frame_index,
                    timestamp_s=frame.timestamp_s,
                )
                if evs:
                    for ev in evs:
                        if ev.mask_area_px > 0:
                            areas.append(ev.mask_area_px)
                            confidences.append(ev.confidence)
                    frames_detected += 1

            if areas:
                stats[step_id_str]["objects"][obj] = {
                    "frames_detected": frames_detected,
                    "total_frames": len(sampled.frames),
                    "min_area": min(areas),
                    "max_area": max(areas),
                    "avg_area": int(sum(areas) / len(areas)),
                    "min_conf": round(min(confidences), 2),
                    "max_conf": round(max(confidences), 2),
                    "avg_conf": round(sum(confidences) / len(confidences), 2),
                    "recommended_min_area": int(min(areas) * 0.8),
                    "recommended_min_conf": round(max(0.5, min(confidences) * 0.9), 2),
                }
            else:
                stats[step_id_str]["objects"][obj] = {
                    "frames_detected": 0,
                    "total_frames": len(sampled.frames),
                    "min_area": 0,
                    "max_area": 0,
                    "avg_area": 0,
                    "min_conf": 0.0,
                    "max_conf": 0.0,
                    "avg_conf": 0.0,
                    "recommended_min_area": 200,
                    "recommended_min_conf": 0.50,
                }

    return stats
