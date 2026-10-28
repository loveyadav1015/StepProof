"""Tests for LLM and Segmentation providers, stubs, and factory."""

import cv2
import numpy as np
import pytest

from stepproof.providers.factory import get_llm_provider, get_seg_provider
from stepproof.providers.llm_mock import LLMMockProvider
from stepproof.providers.seg_local_sam import SegLocalSAMProvider
from stepproof.providers.seg_meta_api import SegMetaAPIProvider
from stepproof.providers.seg_mock import SegMockProvider
from stepproof.schemas import Evidence


def test_mock_llm_generate_and_judge():
    provider = LLMMockProvider()
    assert not provider.get_provenance().meta_compliant

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sop, prov = provider.generate_sop([dummy_frame], prompt="Generate SOP")
    assert len(sop.steps) == 4
    assert sop.steps[0].instruction.startswith("Place the red block")
    assert prov.provider == "mock"

    # Test judge_step when evidence matches
    step1 = sop.steps[0]
    ev_present = [
        Evidence(
            step_id=1,
            frame_index=1,
            timestamp_s=1.0,
            object="red block",
            count=1,
            mask_area_px=500,
            bbox=[10, 10, 50, 50],
            confidence=0.99,
        )
    ]
    verdict, _ = provider.judge_step(dummy_frame, step1, ev_present)
    assert verdict["passed"] is True

    # Test judge_step when evidence is missing
    verdict_fail, _ = provider.judge_step(dummy_frame, step1, [])
    assert verdict_fail["passed"] is False
    assert len(verdict_fail["failed_rules"]) > 0


def test_mock_seg_color_contour_detection():
    seg = SegMockProvider()
    assert not seg.get_provenance().meta_compliant

    # Create synthetic frame with a 40x40 green block
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Green in BGR is (0, 255, 0)
    cv2.rectangle(img, (50, 50), (90, 90), (0, 255, 0), -1)

    ev_green, prov = seg.find(img, "green block", step_id=1, frame_index=0)
    assert len(ev_green) == 1
    assert ev_green[0].mask_area_px >= 1000
    assert ev_green[0].bbox[2] >= 35  # width
    assert ev_green[0].bbox[3] >= 35  # height

    # Red block should not be found in pure green image
    ev_red, _ = seg.find(img, "red block")
    assert len(ev_red) == 0


def test_seg_local_sam_stub_raises():
    sam = SegLocalSAMProvider()
    assert sam.get_provenance().meta_compliant is True
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError) as exc_info:
        sam.find(dummy_frame, "resistor")
    assert "SegLocalSAMProvider is a stub" in str(exc_info.value)


def test_seg_meta_api_with_fixture(monkeypatch):
    from pathlib import Path
    import json
    from stepproof.providers.seg_meta_api import parse_sam_output_text

    fixture_path = Path("tests/fixtures/sam_stream_fixture.txt")
    assert fixture_path.exists()

    content = fixture_path.read_text()
    deltas = []
    for line in content.splitlines():
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            try:
                ev = json.loads(line[6:])
                if ev.get("type") == "response.output_text.delta":
                    deltas.append(ev.get("delta", ""))
            except Exception:
                pass

    full_output = "".join(deltas)
    assert len(full_output) > 0

    meta = SegMetaAPIProvider(api_key="mock_key")
    assert meta.get_provenance().meta_compliant is True

    # Test parse_sam_output_text
    evs = parse_sam_output_text(full_output, phrase="resistor", step_id=1)
    assert len(evs) == 1
    assert evs[0].object == "resistor"
    assert evs[0].bbox == [211, 228, 59, 26]
    assert evs[0].mask_area_px == 1539
    assert evs[0].confidence == 0.97

    # Test find() without network call using monkeypatch
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    monkeypatch.setattr(meta, "_execute_sam_request", lambda phrase, b64_img: full_output)
    found_evs, prov = meta.find(dummy_frame, "resistor")
    assert len(found_evs) == 1
    assert prov.meta_compliant is True


def test_factory_requires_meta_guard():
    # In dev mode, mock is fine
    llm = get_llm_provider("mock", require_meta=False)
    assert isinstance(llm, LLMMockProvider)

    # When require_meta=True, mock must be refused!
    with pytest.raises(RuntimeError) as exc_info:
        get_llm_provider("mock", require_meta=True)
    assert "refuses non-Meta models" in str(exc_info.value)

    # Segmentation mock also refused when require_meta=True
    with pytest.raises(RuntimeError):
        get_seg_provider("mock", require_meta=True)

    # Local SAM and Meta API pass the require_meta guard because they are Meta compliant
    seg_sam = get_seg_provider("local_sam", require_meta=True)
    assert isinstance(seg_sam, SegLocalSAMProvider)

    seg_api = get_seg_provider("meta_api", require_meta=True)
    assert isinstance(seg_api, SegMetaAPIProvider)


def test_llm_openai_compat_chat_style(monkeypatch):
    from unittest.mock import MagicMock
    from stepproof.providers.llm_openai_compat import LLMOpenAICompatProvider

    provider = LLMOpenAICompatProvider(
        base_url="https://api.meta.ai/v1",
        model="muse-wonder-1.0",
        api_key="test-key",
        api_style="chat",
    )

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"task": "Assembly", "steps": []}'
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 42
    mock_resp.usage.completion_tokens = 15
    mock_resp.usage.total_tokens = 57

    monkeypatch.setattr(provider.client.chat.completions, "create", lambda **kwargs: mock_resp)

    text, usage, prov = provider.complete("Test prompt", json_mode=True)
    assert text == '{"task": "Assembly", "steps": []}'
    assert usage["input_tokens"] == 42
    assert usage["output_tokens"] == 15
    assert prov.input_tokens == 42
    assert prov.output_tokens == 15
    assert prov.meta_compliant is True


def test_llm_openai_compat_responses_style(monkeypatch):
    from unittest.mock import MagicMock
    from stepproof.providers.llm_openai_compat import LLMOpenAICompatProvider

    provider = LLMOpenAICompatProvider(
        base_url="https://api.meta.ai/v1",
        model="muse-wonder-1.0",
        api_key="test-key",
        api_style="responses",
    )

    mock_resp = MagicMock()
    mock_resp.output_text = "Here are the safety instructions."
    mock_usage = MagicMock()
    mock_usage.input_tokens = 30
    mock_usage.output_tokens = 12
    mock_usage.total_tokens = 42
    mock_resp.usage = mock_usage

    provider.client.responses = MagicMock()
    monkeypatch.setattr(provider.client.responses, "create", lambda **kwargs: mock_resp)

    text, usage, prov = provider.complete("Safety check", json_mode=False)
    assert text == "Here are the safety instructions."
    assert usage["input_tokens"] == 30
    assert usage["output_tokens"] == 12
    assert prov.total_tokens == 42
    assert prov.meta_compliant is True


def test_llm_openai_compat_budget_guard(monkeypatch):
    from stepproof.config import settings
    from stepproof.providers.llm_openai_compat import LLMOpenAICompatProvider

    provider = LLMOpenAICompatProvider(
        base_url="https://api.meta.ai/v1",
        model="muse-wonder-1.0",
        api_key="test-key",
    )
    provider.request_count = settings.max_requests_per_run

    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("Should fail budget")
    assert "Request budget exceeded" in str(exc_info.value)

