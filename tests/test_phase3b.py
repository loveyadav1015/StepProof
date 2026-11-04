"""Tests for Phase 3b tooling: doctor, inspect, windows, eval NOT-RUN, secrets check, meta-eval guard, prices wording."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from stepproof.doctor import check_video_quality, run_doctor
from stepproof.evaluation import run_evaluation
from stepproof.inspector import inspect_video_for_tuning
from stepproof.providers.seg_mock import SegMockProvider
from stepproof.schemas import SOP
from stepproof.windows import load_step_windows, propose_step_windows


@pytest.fixture
def sample_sop() -> SOP:
    sop_path = Path("data/sop/breadboard_led.json")
    if not sop_path.exists():
        sop_path = Path("data/outputs/demo_sop.json")
    return SOP.model_validate_json(sop_path.read_text(encoding="utf-8"))


def test_prices_yaml_provisional_wording():
    """Verify prices.yaml contains 'provisional - verify in Meta dashboard before live use'."""
    prices_path = Path("prices.yaml")
    assert prices_path.exists()
    content = prices_path.read_text(encoding="utf-8")
    assert "provisional - verify in Meta dashboard before live use" in content
    # The agent cannot verify dashboard prices, so ensure old wording is absent
    assert "dashboard-verified" not in content


def test_secrets_check_passes():
    """Verify scripts/check_secrets.py runs cleanly and finds no exposed live keys."""
    res = subprocess.run(
        [sys.executable, "scripts/check_secrets.py"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Secrets check passed" in res.stdout


def test_doctor_tool(tmp_path: Path):
    """Test stepproof doctor on missing video and synthetic video."""
    synthetic_video = Path("data/synthetic/expert.mp4")
    if not synthetic_video.exists():
        pytest.skip("Synthetic expert video not generated yet")

    report = check_video_quality(synthetic_video)
    assert report["exists"] is True
    assert report["duration_s"] > 0
    assert report["fps"] > 0
    assert "width" in report
    assert "height" in report
    assert "exposure_std" in report

    # Test doctor on dummy manifest
    dummy_manifest = tmp_path / "doctor_manifest.yaml"
    manifest_data = {
        "name": "Doctor Test Manifest",
        "sop": "data/sop/breadboard_led.json",
        "runs": [
            {
                "scenario": "Existing synthetic",
                "video": str(synthetic_video),
                "status": "ready",
            },
            {
                "scenario": "Planned missing",
                "video": "data/learner/missing_video.mp4",
                "status": "planned",
            },
        ],
    }
    dummy_manifest.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

    diag_results = run_doctor(dummy_manifest)
    reports = diag_results["reports"]
    assert len(reports) == 2
    assert reports[0]["exists"] is True
    assert reports[1]["exists"] is False


def test_inspect_tool_output(tmp_path: Path, sample_sop: SOP):
    """Test inspect tool generates evidence_log.csv, timeline.html, and contact sheet."""
    synthetic_video = Path("data/synthetic/expert.mp4")
    if not synthetic_video.exists():
        pytest.skip("Synthetic expert video not generated yet")

    out_dir = tmp_path / "inspect_out"
    seg = SegMockProvider()

    summary = inspect_video_for_tuning(
        video_path=synthetic_video,
        sop=sample_sop,
        segmenter=seg,
        out_dir=out_dir,
    )

    assert (out_dir / "evidence_log.csv").exists()
    assert (out_dir / "timeline.html").exists()
    assert (out_dir / "contact_sheet.jpg").exists()
    assert "unreliable_objects" in summary
    assert "csv_path" in summary
    assert "timeline_html" in summary


def test_windows_tool_proposal(tmp_path: Path, sample_sop: SOP):
    """Test window proposal extracts appearance and stability windows to YAML."""
    synthetic_video = Path("data/synthetic/expert.mp4")
    if not synthetic_video.exists():
        pytest.skip("Synthetic expert video not generated yet")

    out_windows = tmp_path / "windows.yaml"
    seg = SegMockProvider()

    windows_data = propose_step_windows(
        video_path=synthetic_video,
        sop=sample_sop,
        segmenter=seg,
        output_path=out_windows,
    )

    assert out_windows.exists()
    assert isinstance(windows_data, dict)
    assert len(windows_data) > 0
    loaded = load_step_windows(out_windows)
    assert isinstance(loaded, dict)
    assert len(loaded) > 0


def test_eval_not_run_status(tmp_path: Path):
    """Test stepproof eval marks planned/missing videos as NOT RUN with --expect-status ready."""
    dummy_manifest = tmp_path / "eval_manifest.yaml"
    manifest_data = {
        "name": "Eval Not-Run Test",
        "sop": "data/sop/breadboard_led.json",
        "runs": [
            {
                "scenario": "Missing planned video",
                "video": "data/learner/unfilmed_run.mp4",
                "source": "real",
                "status": "planned",
                "expected_flagged_step_ids": [1],
                "expected_primary_flag_types": {1: "skipped"},
            }
        ],
    }
    dummy_manifest.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

    out_report = tmp_path / "eval_report.md"
    results = run_evaluation(
        manifest_path=dummy_manifest,
        output_report_path=out_report,
        require_meta=False,
        expect_status="ready",
    )

    assert len(results["results"]) == 1
    run_res = results["results"][0]
    assert run_res["status"] == "NOT RUN"
    report_text = out_report.read_text(encoding="utf-8")
    assert "NOT RUN" in report_text
    assert "Dev stand-in, not a Meta result" in report_text


def test_meta_eval_refusal_without_yes(tmp_path: Path):
    """Test scripts/meta_eval.py refuses to run without --yes flag."""
    res = subprocess.run(
        [sys.executable, "scripts/meta_eval.py", "manifests/real_footage.yaml"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "REFUSAL" in res.stdout
    assert "--yes" in res.stdout
