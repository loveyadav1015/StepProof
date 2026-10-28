"""Tests for schemas and validation."""

import pytest
from pydantic import ValidationError

from stepproof.provenance import (
    assert_meta_compliant,
    create_provenance,
)
from stepproof.schemas import (
    SOP,
    Check,
    Evidence,
    RunReport,
    Step,
    StepVerdict,
    validate_with_retry,
)


def test_provenance_meta_compliance_rules():
    # Mock should never be meta compliant
    p_mock = create_provenance(provider="mock", model_name="mock-model")
    assert not p_mock.meta_compliant
    assert p_mock.badge == "Dev stand-in"

    # OpenAI-compat with non-meta model
    p_qwen = create_provenance(
        provider="openai_compat",
        model_name="qwen-2.5-72b",
        base_url="https://openrouter.ai/api/v1",
    )
    assert not p_qwen.meta_compliant
    assert p_qwen.badge == "Dev stand-in"

    # OpenAI-compat with Muse Glimmer
    p_glimmer = create_provenance(
        provider="openai_compat",
        model_name="meta/muse-glimmer-30b",
        base_url="https://openrouter.ai/api/v1",
    )
    assert p_glimmer.meta_compliant
    assert p_glimmer.badge == "Meta model"

    # Meta API host
    p_meta = create_provenance(
        provider="openai_compat",
        model_name="muse-wonder",
        base_url="https://api.meta.ai/v1",
    )
    assert p_meta.meta_compliant

    # Local SAM 3.1
    p_sam = create_provenance(
        provider="local_sam",
        model_name="facebookresearch/sam3.1",
    )
    assert p_sam.meta_compliant

    # Assertion raises on non-compliant
    with pytest.raises(RuntimeError):
        assert_meta_compliant(p_mock)


def test_sop_and_step_schemas():
    prov = create_provenance(provider="mock", model_name="mock-model")
    check1 = Check(type="present", object="red block", value=1)
    step1 = Step(
        id=1,
        instruction="Place red block on the base",
        objects=["red block"],
        expected_end_state="Red block resting on table base",
        checks=[check1],
        common_errors=["Skipping red block", "Using blue block"],
        start_s=0.5,
        end_s=2.5,
    )
    sop = SOP(
        task="Assemble Block Tower",
        steps=[step1],
        source_video_hash="dummyhash123",
        created_with=prov,
    )
    assert len(sop.steps) == 1
    assert sop.steps[0].checks[0].object == "red block"


def test_evidence_and_verdict_schemas():
    evidence = Evidence(
        step_id=1,
        frame_index=10,
        timestamp_s=2.0,
        object="red block",
        count=1,
        mask_area_px=1200,
        bbox=[100, 150, 40, 40],
        confidence=0.98,
    )
    assert evidence.mask_area_px == 1200

    verdict = StepVerdict(
        step_id=1,
        state="DONE",
        reason="Object placed and verified in 3 frames",
        failed_rules=[],
        evidence_frame_path="runs/evidence_1.png",
        confidence=0.95,
        flag_type="none",
    )
    assert verdict.state == "DONE"

    report = RunReport(
        sop_id="Assemble Block Tower",
        verdicts=[verdict],
        timings={"ingest": 0.4, "check": 0.8},
        provenance=[create_provenance("mock", "mock-model")],
        cost_estimate={"usd": 0.0},
    )
    assert len(report.verdicts) == 1


def test_validate_with_retry_success_on_first_try():
    data = {"type": "present", "object": "green block", "value": 1}
    res = validate_with_retry(Check, lambda: data)
    assert res.object == "green block"


def test_validate_with_retry_recovers_after_retry():
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            # Missing required field 'object'
            return {"type": "present"}
        return {"type": "present", "object": "blue block"}

    res = validate_with_retry(Check, provider, max_retries=1)
    assert calls == 2
    assert res.object == "blue block"


def test_validate_with_retry_fails_after_retries():
    calls = 0

    def bad_provider():
        nonlocal calls
        calls += 1
        return {"invalid": "missing fields"}

    with pytest.raises(ValidationError):
        validate_with_retry(Check, bad_provider, max_retries=1)
    assert calls == 2
