"""Rule-based verification checker and step state machine with visual evidence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np

from stepproof.ingest.sampler import FrameInfo
from stepproof.provenance import ProvenanceRecord
from stepproof.providers.base import LLMProvider, SegmentationProvider
from stepproof.schemas import Check, Evidence, FlagType, SOP, Step, StepState, StepVerdict

logger = logging.getLogger(__name__)


class StepTrackingState:
    """Internal state tracker for a single SOP step during a verification run."""

    def __init__(self, step: Step):
        self.step = step
        self.state: StepState = "PENDING"
        self.flag_type: FlagType = "none"
        self.reason: str = ""
        self.failed_rules: list[str] = []
        self.consecutive_passes: int = 0
        self.completion_frame_idx: Optional[int] = None
        self.completion_timestamp_s: Optional[float] = None
        self.evidence_frame_path: Optional[str] = None
        self.confidence: float = 1.0
        self.latest_evidence: list[Evidence] = []


def evaluate_step_rules(
    step: Step,
    evidence_list: list[Evidence],
) -> tuple[bool, list[str]]:
    """Evaluate whether segmented evidence satisfies all checks defined on the step."""
    failed: list[str] = []

    for check in step.checks:
        target_obj = check.object.lower().strip()
        matched = [
            ev for ev in evidence_list
            if target_obj in ev.object.lower() and ev.mask_area_px > 0
        ]

        if check.type == "present":
            if not matched:
                failed.append(f"Missing required object '{check.object}'")
        elif check.type == "count":
            expected_count = int(check.value) if check.value is not None else 1
            total_count = sum(ev.count for ev in matched)
            if total_count < expected_count:
                failed.append(
                    f"Count mismatch for '{check.object}': expected {expected_count}, found {total_count}"
                )
        elif check.type == "position":
            if not matched:
                failed.append(f"Cannot verify position: '{check.object}' not detected")
            # If bounding box bounds are provided as [min_x, min_y, max_x, max_y]
            elif isinstance(check.value, (list, tuple)) and len(check.value) == 4:
                min_x, min_y, max_x, max_y = check.value
                for ev in matched:
                    x, y, w, h = ev.bbox
                    if not (min_x <= x <= max_x and min_y <= y <= max_y):
                        failed.append(
                            f"Object '{check.object}' position ({x},{y}) outside expected bounds"
                        )
        elif check.type == "order":
            pass

    return (len(failed) == 0), failed


def draw_evidence_overlay(
    image_path: str | Path,
    output_path: str | Path,
    step: Step,
    state: StepState,
    flag_type: FlagType,
    evidence_list: list[Evidence],
    reason: str = "",
) -> str:
    """Draw bounding boxes and status banner on keyframe image and save."""
    img = cv2.imread(str(image_path))
    if img is None:
        return str(image_path)

    # Color scheme: Green for DONE, Red for FLAGGED wrong/skipped, Yellow for needs_review
    if state == "DONE":
        banner_color = (40, 180, 40)  # BGR green
    elif flag_type == "needs_review":
        banner_color = (30, 200, 240)  # Yellow
    else:
        banner_color = (40, 40, 220)  # Red

    # Draw evidence bounding boxes
    for ev in evidence_list:
        x, y, w, h = ev.bbox
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), banner_color, 2)
            cv2.putText(
                img,
                f"{ev.object} ({ev.confidence:.2f})",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                banner_color,
                1,
                cv2.LINE_AA,
            )

    # Draw top banner
    h_img, w_img = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w_img, 45), (20, 20, 20), -1)
    status_text = f"Step {step.id} [{state} - {flag_type.upper()}]"
    cv2.putText(
        img,
        status_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    if reason:
        cv2.putText(
            img,
            reason[:75],
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), img)
    return str(out_file)


class StepChecker:
    """Executes step state machine against learner video frames."""

    def __init__(
        self,
        sop: SOP,
        segmenter: SegmentationProvider,
        llm_judge: LLMProvider | None = None,
        consecutive_frames_required: int = 3,
        out_dir: Path | str | None = None,
    ):
        self.sop = sop
        self.segmenter = segmenter
        self.llm_judge = llm_judge
        self.consecutive_frames_required = consecutive_frames_required
        self.out_dir = Path(out_dir) if out_dir else None
        self.trackers: list[StepTrackingState] = [StepTrackingState(s) for s in sop.steps]

    def verify_run(
        self,
        sampled_frames: list[FrameInfo],
    ) -> list[StepVerdict]:
        """Run verification over sampled frames and produce final verdicts."""
        for frame_info in sampled_frames:
            frame_path = frame_info.file_path
            t_s = frame_info.timestamp_s
            f_idx = frame_info.frame_index

            # 1. Segment objects needed for pending/active steps
            for i, tracker in enumerate(self.trackers):
                if tracker.state in ("DONE", "FLAGGED"):
                    continue

                step = tracker.step
                frame_evidence: list[Evidence] = []
                for obj_phrase in step.objects:
                    evs, _ = self.segmenter.find(
                        image=frame_path,
                        phrase=obj_phrase,
                        step_id=step.id,
                        frame_index=f_idx,
                        timestamp_s=t_s,
                    )
                    frame_evidence.extend(evs)

                tracker.latest_evidence = frame_evidence
                rules_pass, failed_rules = evaluate_step_rules(step, frame_evidence)

                if rules_pass:
                    if tracker.state == "PENDING":
                        tracker.state = "ACTIVE"
                    tracker.consecutive_passes += 1
                    tracker.failed_rules = []

                    # Step is DONE when passing in consecutive sampled frames
                    if tracker.consecutive_passes >= self.consecutive_frames_required:
                        # Call LLM judge if provided
                        if self.llm_judge is not None:
                            judge_res, _ = self.llm_judge.judge_step(
                                frame=frame_path,
                                step=step,
                                evidence=frame_evidence,
                            )
                            passed = judge_res.get("passed", True)
                            conf = float(judge_res.get("confidence", 0.95))
                            reason = judge_res.get("reason", "Step verified.")

                            # Merge rule: judge says wrong with high confidence but rules pass -> needs_review
                            if not passed and conf >= 0.85:
                                tracker.state = "FLAGGED"
                                tracker.flag_type = "needs_review"
                                tracker.reason = f"Rules passed but judge flagged: {reason}"
                                tracker.confidence = conf
                            else:
                                tracker.state = "DONE"
                                tracker.reason = f"Verified: {reason}"
                                tracker.confidence = conf
                        else:
                            tracker.state = "DONE"
                            tracker.reason = (
                                f"All checks passed across {self.consecutive_frames_required} frames."
                            )

                        tracker.completion_frame_idx = f_idx
                        tracker.completion_timestamp_s = t_s

                        if self.out_dir:
                            overlay_path = self.out_dir / f"step_{step.id}_done.jpg"
                            tracker.evidence_frame_path = draw_evidence_overlay(
                                image_path=frame_path,
                                output_path=overlay_path,
                                step=step,
                                state=tracker.state,
                                flag_type=tracker.flag_type,
                                evidence_list=frame_evidence,
                                reason=tracker.reason,
                            )

                        # Check if any earlier steps were skipped or out of order
                        for earlier_idx in range(i):
                            earlier_tracker = self.trackers[earlier_idx]
                            if earlier_tracker.state in ("PENDING", "ACTIVE"):
                                earlier_tracker.state = "FLAGGED"
                                earlier_tracker.flag_type = "skipped"
                                earlier_tracker.reason = (
                                    f"Skipped: step {step.id} completed at t={t_s:.1f}s while "
                                    f"step {earlier_tracker.step.id} was not completed."
                                )
                                earlier_tracker.confidence = 0.98
                                if self.out_dir:
                                    overlay_path = self.out_dir / f"step_{earlier_tracker.step.id}_skipped.jpg"
                                    earlier_tracker.evidence_frame_path = draw_evidence_overlay(
                                        image_path=frame_path,
                                        output_path=overlay_path,
                                        step=earlier_tracker.step,
                                        state="FLAGGED",
                                        flag_type="skipped",
                                        evidence_list=frame_evidence,
                                        reason=earlier_tracker.reason,
                                    )
                else:
                    tracker.consecutive_passes = 0
                    tracker.failed_rules = failed_rules

        # 2. End of video: finalize any remaining steps
        last_frame_path = sampled_frames[-1].file_path if sampled_frames else ""
        for tracker in self.trackers:
            if tracker.state in ("PENDING", "ACTIVE"):
                tracker.state = "FLAGGED"
                if tracker.consecutive_passes > 0 or tracker.latest_evidence:
                    # Attempted or wrong part
                    tracker.flag_type = "wrong"
                    reasons = tracker.failed_rules or ["End-state not achieved before run ended"]
                    tracker.reason = f"Failed at end of run: {', '.join(reasons)}"
                else:
                    tracker.flag_type = "skipped"
                    tracker.reason = f"Step {tracker.step.id} was never executed during the demonstration."

                tracker.confidence = 0.95
                if self.out_dir and last_frame_path:
                    overlay_path = self.out_dir / f"step_{tracker.step.id}_{tracker.flag_type}.jpg"
                    tracker.evidence_frame_path = draw_evidence_overlay(
                        image_path=last_frame_path,
                        output_path=overlay_path,
                        step=tracker.step,
                        state="FLAGGED",
                        flag_type=tracker.flag_type,
                        evidence_list=tracker.latest_evidence,
                        reason=tracker.reason,
                    )

        # 3. Detect Out-of-Order completions among DONE steps
        done_trackers = [t for t in self.trackers if t.state == "DONE" and t.completion_timestamp_s is not None]
        for idx in range(1, len(done_trackers)):
            curr = done_trackers[idx]
            prev = done_trackers[idx - 1]
            if curr.completion_timestamp_s < prev.completion_timestamp_s:
                curr.state = "FLAGGED"
                curr.flag_type = "out_of_order"
                curr.reason = (
                    f"Out of order: step {curr.step.id} completed at t={curr.completion_timestamp_s:.1f}s, "
                    f"before step {prev.step.id} completed at t={prev.completion_timestamp_s:.1f}s."
                )

        # Build output StepVerdict models
        verdicts: list[StepVerdict] = []
        for tracker in self.trackers:
            verdicts.append(
                StepVerdict(
                    step_id=tracker.step.id,
                    state=tracker.state,
                    reason=tracker.reason,
                    failed_rules=tracker.failed_rules,
                    evidence_frame_path=tracker.evidence_frame_path,
                    confidence=tracker.confidence,
                    flag_type=tracker.flag_type,
                )
            )

        return verdicts
