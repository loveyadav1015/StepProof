"""Tests for video ingest sampler and ffmpeg dependency checks."""

from pathlib import Path
import tempfile
import pytest

from stepproof.ingest.sampler import (
    check_ffmpeg_installed,
    compute_file_hash,
    sample_video,
)


def test_check_ffmpeg_installed_passes(monkeypatch):
    # System has ffmpeg installed
    check_ffmpeg_installed()

    # Simulate ffmpeg missing from system PATH
    monkeypatch.setattr("shutil.which", lambda prog: None)
    with pytest.raises(RuntimeError) as exc_info:
        check_ffmpeg_installed()
    assert "ffmpeg" in str(exc_info.value)


def test_sampler_on_synthetic_expert_video():
    expert_path = Path("data/expert/expert.mp4")
    if not expert_path.exists():
        from scripts.make_synthetic_video import generate_all_synthetic_videos
        generate_all_synthetic_videos()

    with tempfile.TemporaryDirectory() as tmpdir:
        sampled = sample_video(
            video_path=expert_path,
            output_dir=tmpdir,
            target_fps=2.0,
            scene_threshold=15.0,
            check_ffmpeg=False,
        )

        assert sampled.total_frames_extracted > 0
        assert len(sampled.frames) == sampled.total_frames_extracted
        assert sampled.duration_s >= 7.0
        assert len(sampled.video_hash) == 64  # SHA256 length

        # Verify that frame files are actually written and have valid timestamps
        first_frame = sampled.frames[0]
        assert Path(first_frame.file_path).exists()
        assert first_frame.timestamp_s == 0.0

        # Verify scene change detection captured events
        scene_changes = [f for f in sampled.frames if f.is_scene_change]
        assert len(scene_changes) > 0
