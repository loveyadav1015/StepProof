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
