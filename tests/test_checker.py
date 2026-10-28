"""Tests for step state machine checker under all failure and pass scenarios."""

import pytest

from stepproof.pipeline.checker import StepChecker
from stepproof.providers.llm_mock import LLMMockProvider
from stepproof.providers.seg_mock import SegMockProvider
from stepproof.schemas import SOP, Check, Evidence, Step


@pytest.fixture
def base_sop():
    prov = LLMMockProvider().get_provenance()
    steps = [
        Step(
            id=1,
            instruction="Place red block in left slot",
            objects=["red block"],
            expected_end_state="Red block resting in left slot",
            checks=[Check(type="present", object="red block", value=1)],
            start_s=0.0,
            end_s=2.0,
        ),
        Step(
            id=2,
            instruction="Place green block in center slot",
            objects=["green block"],
            expected_end_state="Green block resting in center slot",
            checks=[Check(type="present", object="green block", value=1)],
            start_s=2.0,
            end_s=4.0,
        ),
        Step(
            id=3,
            instruction="Place blue block in right slot",
            objects=["blue block"],
            expected_end_state="Blue block resting in right slot",
            checks=[Check(type="present", object="blue block", value=1)],
            start_s=4.0,
            end_s=6.0,
        ),
    ]
    return SOP(
        task="Block Assembly Task",
        steps=steps,
        source_video_hash="testhash123",
        created_with=prov,
    )


def test_checker_correct_run(base_sop, monkeypatch):
    """Scenario 1: Correct execution: all steps pass in consecutive frames."""
    segmenter = SegMockProvider()

    # Create frames where objects appear in order
    frames = [
        # Step 1 passes (3 frames)
        {"t": 0.5, "present": ["red block"]},
        {"t": 1.0, "present": ["red block"]},
        {"t": 1.5, "present": ["red block"]},
        # Step 2 passes (3 frames)
        {"t": 2.0, "present": ["red block", "green block"]},
        {"t": 2.5, "present": ["red block", "green block"]},
        {"t": 3.0, "present": ["red block", "green block"]},
        # Step 3 passes (3 frames)
        {"t": 3.5, "present": ["red block", "green block", "blue block"]},
        {"t": 4.0, "present": ["red block", "green block", "blue block"]},
        {"t": 4.5, "present": ["red block", "green block", "blue block"]},
    ]

    def mock_find(image, phrase, step_id=0, frame_index=0, timestamp_s=0.0):
        frame_data = frames[frame_index]
        found = []
        for present_obj in frame_data["present"]:
            if present_obj == phrase:
                found.append(
                    Evidence(
                        step_id=step_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        object=phrase,
                        count=1,
                        mask_area_px=1500,
                        bbox=[100, 100, 50, 50],
                        confidence=0.99,
                    )
                )
        return found, segmenter.get_provenance()

    monkeypatch.setattr(segmenter, "find", mock_find)

    from stepproof.ingest.sampler import FrameInfo
    sampled_frames = [
        FrameInfo(frame_index=i, original_frame_number=i * 5, timestamp_s=f["t"], file_path=f"dummy_{i}.jpg")
        for i, f in enumerate(frames)
    ]

    checker = StepChecker(base_sop, segmenter=segmenter, consecutive_frames_required=3)
    verdicts = checker.verify_run(sampled_frames)

    assert len(verdicts) == 3
    assert verdicts[0].step_id == 1 and verdicts[0].state == "DONE" and verdicts[0].flag_type == "none"
    assert verdicts[1].step_id == 2 and verdicts[1].state == "DONE" and verdicts[1].flag_type == "none"
    assert verdicts[2].step_id == 3 and verdicts[2].state == "DONE" and verdicts[2].flag_type == "none"


def test_checker_skipped_step(base_sop, monkeypatch):
    """Scenario 2: Step 2 is skipped: step 3 finishes while step 2 is pending."""
    segmenter = SegMockProvider()

    # Step 1 passes, then step 3 appears directly (no green block ever)
    frames = [
        {"t": 0.5, "present": ["red block"]},
        {"t": 1.0, "present": ["red block"]},
        {"t": 1.5, "present": ["red block"]},
        {"t": 2.0, "present": ["red block", "blue block"]},
        {"t": 2.5, "present": ["red block", "blue block"]},
        {"t": 3.0, "present": ["red block", "blue block"]},
    ]

    def mock_find(image, phrase, step_id=0, frame_index=0, timestamp_s=0.0):
        frame_data = frames[frame_index]
        found = []
        for present_obj in frame_data["present"]:
            if present_obj == phrase:
                found.append(
                    Evidence(
                        step_id=step_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        object=phrase,
                        count=1,
                        mask_area_px=1500,
                        bbox=[100, 100, 50, 50],
                        confidence=0.99,
                    )
                )
        return found, segmenter.get_provenance()

    monkeypatch.setattr(segmenter, "find", mock_find)

    from stepproof.ingest.sampler import FrameInfo
    sampled_frames = [
        FrameInfo(frame_index=i, original_frame_number=i * 5, timestamp_s=f["t"], file_path=f"dummy_{i}.jpg")
        for i, f in enumerate(frames)
    ]

    checker = StepChecker(base_sop, segmenter=segmenter, consecutive_frames_required=3)
    verdicts = checker.verify_run(sampled_frames)

    v1, v2, v3 = verdicts
    assert v1.step_id == 1 and v1.state == "DONE" and v1.flag_type == "none"
    assert v2.step_id == 2 and v2.state == "FLAGGED" and v2.flag_type == "skipped"
    assert "Skipped" in v2.reason or "skipped" in v2.reason
    assert v3.step_id == 3 and v3.state == "DONE" and v3.flag_type == "none"


def test_checker_wrong_part(base_sop, monkeypatch):
    """Scenario 3: Wrong part placed (purple block instead of green block)."""
    segmenter = SegMockProvider()

    frames = [
        {"t": 0.5, "present": ["red block"]},
        {"t": 1.0, "present": ["red block"]},
        {"t": 1.5, "present": ["red block"]},
        {"t": 2.0, "present": ["red block", "purple block"]},
        {"t": 2.5, "present": ["red block", "purple block"]},
        {"t": 3.0, "present": ["red block", "purple block"]},
    ]

    def mock_find(image, phrase, step_id=0, frame_index=0, timestamp_s=0.0):
        frame_data = frames[frame_index]
        found = []
        for present_obj in frame_data["present"]:
            if present_obj == phrase:
                found.append(
                    Evidence(
                        step_id=step_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        object=phrase,
                        count=1,
                        mask_area_px=1500,
                        bbox=[100, 100, 50, 50],
                        confidence=0.99,
                    )
                )
        return found, segmenter.get_provenance()

    monkeypatch.setattr(segmenter, "find", mock_find)

    from stepproof.ingest.sampler import FrameInfo
    sampled_frames = [
        FrameInfo(frame_index=i, original_frame_number=i * 5, timestamp_s=f["t"], file_path=f"dummy_{i}.jpg")
        for i, f in enumerate(frames)
    ]

    checker = StepChecker(base_sop, segmenter=segmenter, consecutive_frames_required=3)
    verdicts = checker.verify_run(sampled_frames)

    v1, v2, v3 = verdicts
    assert v1.step_id == 1 and v1.state == "DONE" and v1.flag_type == "none"
    assert v2.step_id == 2 and v2.state == "FLAGGED" and v2.flag_type == "wrong"
    assert len(v2.failed_rules) > 0
    assert any("green block" in r for r in v2.failed_rules)


def test_checker_out_of_order(base_sop, monkeypatch):
    """Scenario 4: Step 3 completes before Step 2 completes."""
    segmenter = SegMockProvider()

    frames = [
        {"t": 0.5, "present": ["red block"]},
        {"t": 1.0, "present": ["red block"]},
        {"t": 1.5, "present": ["red block"]},
        # Step 3 completes at t=2.0-3.0
        {"t": 2.0, "present": ["red block", "blue block"]},
        {"t": 2.5, "present": ["red block", "blue block"]},
        {"t": 3.0, "present": ["red block", "blue block"]},
        # Step 2 completes late at t=3.5-4.5
        {"t": 3.5, "present": ["red block", "blue block", "green block"]},
        {"t": 4.0, "present": ["red block", "blue block", "green block"]},
        {"t": 4.5, "present": ["red block", "blue block", "green block"]},
    ]

    def mock_find(image, phrase, step_id=0, frame_index=0, timestamp_s=0.0):
        frame_data = frames[frame_index]
        found = []
        for present_obj in frame_data["present"]:
            if present_obj == phrase:
                found.append(
                    Evidence(
                        step_id=step_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        object=phrase,
                        count=1,
                        mask_area_px=1500,
                        bbox=[100, 100, 50, 50],
                        confidence=0.99,
                    )
                )
        return found, segmenter.get_provenance()

    monkeypatch.setattr(segmenter, "find", mock_find)

    from stepproof.ingest.sampler import FrameInfo
    sampled_frames = [
        FrameInfo(frame_index=i, original_frame_number=i * 5, timestamp_s=f["t"], file_path=f"dummy_{i}.jpg")
        for i, f in enumerate(frames)
    ]

    checker = StepChecker(base_sop, segmenter=segmenter, consecutive_frames_required=3)
    verdicts = checker.verify_run(sampled_frames)

    # Step 2 completed out of order (after Step 3 completed)
    assert verdicts[0].step_id == 1 and verdicts[0].state == "DONE" and verdicts[0].flag_type == "none"
    assert verdicts[1].step_id == 2 and verdicts[1].state == "FLAGGED" and verdicts[1].flag_type == "out_of_order"
    assert verdicts[2].step_id == 3 and verdicts[2].state == "DONE" and verdicts[2].flag_type == "none"


def test_checker_needs_review_merge_rule(base_sop, monkeypatch):
    """Scenario 5: Segmentation rules pass, but LLM judge says wrong with high confidence."""
    segmenter = SegMockProvider()
    judge = LLMMockProvider()

    frames = [
        {"t": 0.5, "present": ["red block"]},
        {"t": 1.0, "present": ["red block"]},
        {"t": 1.5, "present": ["red block"]},
    ]

    def mock_find(image, phrase, step_id=0, frame_index=0, timestamp_s=0.0):
        return [
            Evidence(
                step_id=step_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                object=phrase,
                count=1,
                mask_area_px=1500,
                bbox=[100, 100, 50, 50],
                confidence=0.99,
            )
        ], segmenter.get_provenance()

    # Mock judge returns high-confidence failure
    def mock_judge_step(frame, step, evidence, learner_action_desc=""):
        return {
            "passed": False,
            "reason": "Block is misaligned with edge of workbench.",
            "confidence": 0.92,
            "failed_rules": ["Misalignment detected"],
        }, judge.get_provenance()

    monkeypatch.setattr(segmenter, "find", mock_find)
    monkeypatch.setattr(judge, "judge_step", mock_judge_step)

    from stepproof.ingest.sampler import FrameInfo
    sampled_frames = [
        FrameInfo(frame_index=i, original_frame_number=i * 5, timestamp_s=f["t"], file_path=f"dummy_{i}.jpg")
        for i, f in enumerate(frames)
    ]

    checker = StepChecker(
        base_sop,
        segmenter=segmenter,
        llm_judge=judge,
        consecutive_frames_required=3,
    )
    verdicts = checker.verify_run(sampled_frames)

    v1 = verdicts[0]
    assert v1.state == "FLAGGED"
    assert v1.flag_type == "needs_review"
    assert "judge flagged" in v1.reason
