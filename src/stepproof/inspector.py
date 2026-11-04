"""Inspect and tune segmentation thresholds and step timelines on real footage."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stepproof.calibration import load_calibration_config
from stepproof.ingest.sampler import sample_video
from stepproof.pipeline.checker import evaluate_step_rules
from stepproof.providers.base import SegmentationProvider
from stepproof.schemas import SOP, Evidence

logger = logging.getLogger(__name__)


def inspect_video_for_tuning(
    video_path: Path | str,
    sop: SOP,
    segmenter: SegmentationProvider,
    out_dir: Path | str,
    calibration_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample video frames, run segmenter on active/next step objects, write CSV and generate visual summaries."""
    v_path = Path(video_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "sampled_frames"
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    sampled = sample_video(v_path, output_dir=frames_dir)
    cfg = calibration_cfg or load_calibration_config()

    csv_path = output_dir / "evidence_log.csv"
    csv_rows = []

    object_areas: dict[str, list[int]] = {}
    object_confs: dict[str, list[float]] = {}
    step_timelines: dict[int, list[dict[str, Any]]] = {s.id: [] for s in sop.steps}

    contact_sheet_frames = []

    for f_idx, frame_info in enumerate(sampled.frames):
        t_s = frame_info.timestamp_s
        frame_img = cv2.imread(str(frame_info.file_path))
        overlay_img = frame_img.copy() if frame_img is not None else None

        for step in sop.steps:
            step_evidence: list[Evidence] = []
            for obj_phrase in step.objects:
                evs, _ = segmenter.find(
                    image=frame_info.file_path,
                    phrase=obj_phrase,
                    step_id=step.id,
                    frame_index=f_idx,
                    timestamp_s=t_s,
                )
                step_evidence.extend(evs)
                if obj_phrase not in object_areas:
                    object_areas[obj_phrase] = []
                    object_confs[obj_phrase] = []

                if evs:
                    for ev in evs:
                        object_areas[obj_phrase].append(ev.mask_area_px)
                        object_confs[obj_phrase].append(ev.confidence)
                        if overlay_img is not None:
                            x, y, w, h = ev.bbox
                            cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (40, 220, 40), 2)
                            cv2.putText(
                                overlay_img,
                                f"{ev.object} {ev.mask_area_px}px",
                                (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (40, 220, 40),
                                1,
                            )
                else:
                    object_areas[obj_phrase].append(0)
                    object_confs[obj_phrase].append(0.0)

            passed, failed_rules = evaluate_step_rules(step, step_evidence, cfg)
            step_timelines[step.id].append({
                "t": t_s,
                "passed": passed,
                "failed_rules": failed_rules,
                "evidence_count": len(step_evidence),
            })

            # Record in CSV rows
            for obj_phrase in step.objects:
                matched_evs = [e for e in step_evidence if obj_phrase.lower() in e.object.lower()]
                count = sum(e.count for e in matched_evs)
                area = sum(e.mask_area_px for e in matched_evs)
                conf = max([e.confidence for e in matched_evs], default=0.0)
                csv_rows.append({
                    "timestamp_s": round(t_s, 2),
                    "frame_index": f_idx,
                    "step_id": step.id,
                    "object": obj_phrase,
                    "count": count,
                    "area_px": area,
                    "confidence": round(conf, 2),
                    "rules_passed": passed,
                    "failed_rules": "; ".join(failed_rules),
                })

        # Save annotated keyframe overlay
        if overlay_img is not None and f_idx % max(1, len(sampled.frames) // 6) == 0:
            frame_overlay_file = overlays_dir / f"frame_{f_idx:03d}.jpg"
            cv2.imwrite(str(frame_overlay_file), overlay_img)
            # Resize for contact sheet
            thumb = cv2.resize(overlay_img, (320, 180))
            contact_sheet_frames.append(thumb)

    # Write CSV evidence log
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp_s", "frame_index", "step_id", "object",
            "count", "area_px", "confidence", "rules_passed", "failed_rules"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Build contact sheet image
    if contact_sheet_frames:
        # Arrange thumbnails in grid of up to 3 columns
        cols = min(3, len(contact_sheet_frames))
        rows = (len(contact_sheet_frames) + cols - 1) // cols
        sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
        for i, thumb in enumerate(contact_sheet_frames):
            r = i // cols
            c = i % cols
            sheet[r * 180:(r + 1) * 180, c * 320:(c + 1) * 320] = thumb
        cv2.imwrite(str(output_dir / "contact_sheet.jpg"), sheet)

    # Generate HTML timeline report
    timeline_html_path = output_dir / "timeline.html"
    generate_html_timeline(sop, step_timelines, timeline_html_path)

    # Compute object reliability metrics
    unreliable_objects = []
    print("\n" + "=" * 70)
    print("🔬 STEPPROOF INSPECTION & TUNING REPORT")
    print("=" * 70)
    for obj, areas in object_areas.items():
        non_zeros = [a for a in areas if a > 0]
        seen_pct = (len(non_zeros) / max(1, len(areas))) * 100
        mean_area = float(np.mean(non_zeros)) if non_zeros else 0.0
        std_area = float(np.std(non_zeros)) if non_zeros else 0.0
        cv_val = (std_area / mean_area) if mean_area > 0 else 0.0

        is_unreliable = seen_pct < 25 or cv_val > 0.60
        status_tag = "⚠️ UNRELIABLE" if is_unreliable else "✅ STABLE"
        if is_unreliable:
            unreliable_objects.append(obj)

        print(f"Object: {obj:<18} | Seen: {seen_pct:4.1f}% | Avg Area: {mean_area:6.0f}px (std={std_area:4.0f}) | Status: {status_tag}")
        if is_unreliable:
            print(f"   💡 Tuning Tip: High variance ({cv_val:.2f}) or missing detections. Calibrate HSV/prompts or lower min_area.")

    print("\nOutputs generated:")
    print(f"  - Evidence CSV:     {csv_path}")
    print(f"  - Timeline Visual:  {timeline_html_path}")
    print(f"  - Contact Sheet:    {output_dir / 'contact_sheet.jpg'}")
    print("=" * 70 + "\n")

    return {
        "csv_path": str(csv_path),
        "timeline_html": str(timeline_html_path),
        "unreliable_objects": unreliable_objects,
    }


def generate_html_timeline(sop: SOP, timelines: dict[int, list[dict[str, Any]]], out_path: Path) -> None:
    """Generate interactive standalone HTML visual timeline of step rule pass/fail over time."""
    rows_html = []
    for step in sop.steps:
        records = timelines.get(step.id, [])
        blocks = []
        for rec in records:
            color = "#2e7d32" if rec["passed"] else "#d32f2f"
            title = f"t={rec['t']}s: {'PASS' if rec['passed'] else 'FAIL: ' + '; '.join(rec['failed_rules'])}"
            blocks.append(
                f'<span style="display:inline-block; width:12px; height:24px; background:{color}; '
                f'margin-right:2px; border-radius:2px;" title="{title}"></span>'
            )
        step_bar = "".join(blocks)
        rows_html.append(f"""
        <div style="margin-bottom:16px;">
            <h4>Step {step.id}: {step.instruction}</h4>
            <div>{step_bar}</div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>StepProof Verification Timeline: {sop.task}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background:#f9f9f9; }}
        .card {{ background:white; padding:20px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="card">
        <h2>⏱️ Step Verification Timeline: {sop.task}</h2>
        <p>Green = All checks pass | Red = Check failed / Object missing</p>
        {"".join(rows_html)}
    </div>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
