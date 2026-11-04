"""Step window alignment and time boundary proposal derived from video evidence."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from stepproof.ingest.sampler import sample_video
from stepproof.providers.base import SegmentationProvider
from stepproof.schemas import SOP

logger = logging.getLogger(__name__)


def load_step_windows(windows_path: str | Path = "data/sop/windows.yaml") -> dict[int, dict[str, float]]:
    """Load hand-edited or proposed step timing windows."""
    path = Path(windows_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            windows = {}
            for k, v in data.get("steps", {}).items():
                step_id = int(k) if str(k).isdigit() else k
                windows[step_id] = {
                    "start_s": float(v.get("start_s", 0.0)),
                    "end_s": float(v.get("end_s", 0.0)),
                }
            return windows
    except Exception as e:
        logger.warning("Could not read windows file at %s: %s", path, e)
        return {}


def propose_step_windows(
    video_path: Path | str,
    sop: SOP,
    segmenter: SegmentationProvider,
    output_path: Path | str = "data/sop/windows.yaml",
) -> dict[int, dict[str, float]]:
    """Derive each step's start and end times when its objects first appear and become stable."""
    v_path = Path(video_path)
    frames_dir = v_path.parent / "window_frames"
    sampled = sample_video(v_path, output_dir=frames_dir)
    total_duration = sampled.frames[-1].timestamp_s if sampled.frames else 8.0

    step_appearance: dict[int, list[float]] = {s.id: [] for s in sop.steps}

    for frame in sampled.frames:
        t_s = frame.timestamp_s
        for step in sop.steps:
            for obj in step.objects:
                evs, _ = segmenter.find(
                    image=frame.file_path,
                    phrase=obj,
                    step_id=step.id,
                    frame_index=frame.frame_index,
                    timestamp_s=t_s,
                )
                if evs and any(e.mask_area_px > 0 for e in evs):
                    step_appearance[step.id].append(t_s)

    proposed_windows: dict[int, dict[str, float]] = {}

    print("\n" + "=" * 70)
    print(f"⏱️ STEP TIMING WINDOW PROPOSAL: {v_path.name}")
    print("=" * 70)

    prev_end = 0.0
    for idx, step in enumerate(sop.steps):
        times = step_appearance.get(step.id, [])
        if times:
            # First stable appearance
            start_s = max(prev_end, round(times[0], 2))
            # End is either when next step appears or last appearance of this step
            next_step_times = step_appearance.get(sop.steps[idx + 1].id, []) if idx + 1 < len(sop.steps) else []
            if next_step_times:
                end_s = round(next_step_times[0], 2)
            else:
                end_s = round(max(start_s + 1.0, times[-1]), 2)
        else:
            # Fallback uniform spacing if objects were never seen
            start_s = round(prev_end, 2)
            end_s = round(start_s + (total_duration / len(sop.steps)), 2)

        prev_end = end_s
        proposed_windows[step.id] = {
            "start_s": start_s,
            "end_s": end_s,
            "instruction": step.instruction,
        }
        print(f"Step {step.id}: [{start_s:.2f}s -> {end_s:.2f}s] - {step.instruction[:45]}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    yaml_data = {
        "video": str(v_path),
        "sop_task": sop.task,
        "steps": {
            s_id: {
                "start_s": data["start_s"],
                "end_s": data["end_s"],
                "instruction": data["instruction"],
            }
            for s_id, data in proposed_windows.items()
        },
    }

    with open(out_p, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print("\n" + "=" * 70)
    print(f"✅ Proposed windows written to: {out_p}")
    print("Edit this file by hand to fine-tune exact step boundaries.")
    print("=" * 70 + "\n")

    return {k: {"start_s": v["start_s"], "end_s": v["end_s"]} for k, v in proposed_windows.items()}
