"""Smoke tests for the StepProof CLI commands."""

import tempfile
from pathlib import Path

import pytest

from scripts.make_synthetic_video import generate_all_synthetic_videos
from stepproof.__main__ import main


@pytest.fixture(scope="module")
def synthetic_videos():
    return generate_all_synthetic_videos()


def test_cli_end_to_end_smoke(synthetic_videos, monkeypatch):
    expert_video = str(synthetic_videos["expert"])
    learner_skipped = str(synthetic_videos["learner_skipped"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sop_path = tmp_path / "sop.json"
        ground_dir = tmp_path / "grounding"
        run_dir = tmp_path / "runs" / "skipped_run"
        report_md_path = tmp_path / "custom_report.md"

        # 1. build-sop
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "build-sop", expert_video, "--out", str(sop_path)],
        )
        assert main() == 0
        assert sop_path.exists()

        # 2. ground
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "ground", expert_video, "--sop", str(sop_path), "--out-dir", str(ground_dir)],
        )
        assert main() == 0
        assert (ground_dir / "grounding_reference.json").exists()

        # 3. check
        monkeypatch.setattr(
            "sys.argv",
            [
                "stepproof",
                "check",
                learner_skipped,
                "--sop",
                str(sop_path),
                "--out-dir",
                str(run_dir),
            ],
        )
        assert main() == 0
        assert (run_dir / "report.json").exists()

        # 4. report
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "report", str(run_dir), "--out", str(report_md_path)],
        )
        assert main() == 0
        assert report_md_path.exists()
        assert "StepProof Verification Report" in report_md_path.read_text()

        # 5. replay
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "replay", str(run_dir)],
        )
        assert main() == 0

        # 6. replay with --require-meta fails because mock is not meta compliant
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "replay", str(run_dir), "--require-meta"],
        )
        assert main() == 1


def test_cli_calibrate_and_eval(synthetic_videos, monkeypatch):
    expert_video = str(synthetic_videos["expert"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sop_path = tmp_path / "sop.json"
        eval_report_path = tmp_path / "test_eval_report.md"

        # Build SOP first
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "build-sop", expert_video, "--out", str(sop_path)],
        )
        assert main() == 0

        # Calibrate
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "calibrate", expert_video, "--sop", str(sop_path)],
        )
        assert main() == 0

        # Eval
        manifest_path = "manifests/synthetic_manifest.yaml"
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "eval", manifest_path, "--out", str(eval_report_path)],
        )
        assert main() == 0
        assert eval_report_path.exists()
        assert "EVALUATION REPORT" in eval_report_path.read_text().upper()


def test_make_final_require_meta_refuses_cached_non_meta(synthetic_videos, monkeypatch, capsys):
    learner_skipped = str(synthetic_videos["learner_skipped"])
    expert_video = str(synthetic_videos["expert"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sop_path = tmp_path / "sop.json"
        run_dir = tmp_path / "runs" / "skipped"

        # 1. Build SOP
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "build-sop", expert_video, "--out", str(sop_path)],
        )
        assert main() == 0

        # 2. Check learner (generates run report with mock non-meta provenance)
        monkeypatch.setattr(
            "sys.argv",
            [
                "stepproof",
                "check",
                learner_skipped,
                "--sop",
                str(sop_path),
                "--out-dir",
                str(run_dir),
            ],
        )
        assert main() == 0

        # 3. Simulate make demo-cached: replay strictly from cache
        monkeypatch.setenv("STEPPROOF_MODE", "cache_only")
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "replay", str(run_dir)],
        )
        assert main() == 0

        # 4. Simulate make final: sets STEPPROOF_REQUIRE_META=1 and refuses non-meta cached results
        monkeypatch.setenv("STEPPROOF_REQUIRE_META", "1")
        monkeypatch.setattr(
            "sys.argv",
            ["stepproof", "replay", str(run_dir), "--require-meta"],
        )
        assert main() == 1

        captured = capsys.readouterr()
        err_msg = captured.err + captured.out
        assert "refuses non-Meta models" in err_msg or "STEPPROOF_REQUIRE_META REFUSAL" in err_msg

