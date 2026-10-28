"""Ground SOP objects on expert demonstration keyframes using segmentation provider."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import cv2

from stepproof.ingest.sampler import FrameInfo, SampledVideo
from stepproof.provenance import ProvenanceRecord
from stepproof.providers.base import SegmentationProvider
from stepproof.schemas import Evidence, SOP, Step

logger = logging.getLogger(__name__)


def find_step_keyframe(step: Step, frames: list[FrameInfo]) -> FrameInfo:
    """Select keyframe closest to step's expected end timestamp."""
    if not frames:
        raise ValueError("Cannot select keyframe from empty frame list")

    target_t = step.end_s if step.end_s > 0 else step.start_s
    best_frame = frames[0]
    min_delta = abs(frames[0].timestamp_s - target_t)

    for f in frames[1:]:
        delta = abs(f.timestamp_s - target_t)
        if delta < min_delta:
            min_delta = delta
            best_frame = f

    return best_frame


def ground_step_objects(
    sop: SOP,
    sampled_video: SampledVideo,
    segmenter: SegmentationProvider,
    output_dir: Path | str,
) -> tuple[dict[int | str, list[Evidence]], ProvenanceRecord]:
    """Segment and ground physical objects for each step in the SOP on expert keyframes.

    Args:
        sop: SOP containing steps and object lists.
        sampled_video: Extracted frames from the expert demonstration video.
        segmenter: Segmentation provider (e.g. SegMockProvider, SAM 3.1).
        output_dir: Directory to save grounded overlay images.

    Returns:
        Tuple of (dict mapping step_id to Evidence list, ProvenanceRecord).
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    grounded_evidence: dict[int | str, list[Evidence]] = {}
    prov = segmenter.get_provenance()

    for step in sop.steps:
        keyframe = find_step_keyframe(step, sampled_video.frames)
        img = cv2.imread(keyframe.file_path)
        if img is None:
            logger.warning("Could not load keyframe image at %s", keyframe.file_path)
            continue

        step_evs: list[Evidence] = []
        for obj_phrase in step.objects:
            evs, _ = segmenter.find(
                image=keyframe.file_path,
                phrase=obj_phrase,
                step_id=step.id,
                frame_index=keyframe.frame_index,
                timestamp_s=keyframe.timestamp_s,
            )
            step_evs.extend(evs)

        grounded_evidence[step.id] = step_evs

        # Draw grounding overlay
        overlay = img.copy()
        for ev in step_evs:
            x, y, w, h = ev.bbox
            if w > 0 and h > 0:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.putText(
                    overlay,
                    f"Ref: {ev.object} (area={ev.mask_area_px}px)",
                    (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA,
                )

        overlay_path = dest / f"step_{step.id}_grounding.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        logger.info("Saved grounding overlay for Step %s to %s", step.id, overlay_path)

    # Save summary json
    summary_path = dest / "grounding_reference.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        serializable = {
            str(k): [ev.model_dump() for ev in v]
            for k, v in grounded_evidence.items()
        }
        json.dump(serializable, f, indent=2)

    return grounded_evidence, prov
