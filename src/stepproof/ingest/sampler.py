"""Video ingest, downsampling, and scene-change frame extraction using OpenCV."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import shutil
import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def check_ffmpeg_installed() -> None:
    """Verify that ffmpeg executable is available on system PATH.

    Raises:
        RuntimeError: If ffmpeg is not found.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "External dependency 'ffmpeg' was not found on your system PATH. "
            "StepProof requires ffmpeg for video ingest and decoding. "
            "Please install it: 'sudo apt install ffmpeg' (Linux) or 'brew install ffmpeg' (macOS)."
        )


def compute_file_hash(path: Path | str) -> str:
    """Compute SHA256 checksum of video file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class FrameInfo(BaseModel):
    """Metadata describing an extracted sample frame."""

    frame_index: int = Field(description="Sequential sample index")
    original_frame_number: int = Field(description="Frame number in raw video source")
    timestamp_s: float = Field(description="Exact timestamp in seconds")
    file_path: str = Field(description="Path to saved JPEG frame image")
    is_scene_change: bool = Field(default=False, description="True if captured due to visual change")


class SampledVideo(BaseModel):
    """Container for sampled frames and source video properties."""

    video_path: str = Field(description="Path to original source video")
    video_hash: str = Field(description="SHA256 checksum of source video")
    source_fps: float = Field(description="Native frame rate of source video")
    duration_s: float = Field(description="Total video duration in seconds")
    total_frames_extracted: int = Field(description="Count of sampled frames saved")
    frames: list[FrameInfo] = Field(default_factory=list, description="Ordered sampled frames")


def sample_video(
    video_path: Path | str,
    output_dir: Path | str,
    target_fps: float = 2.0,
    scene_threshold: float = 1.8,
    min_changed_pixels: int = 800,
    max_dimension: int = 1280,
    check_ffmpeg: bool = True,
) -> SampledVideo:
    """Sample video at target_fps plus significant scene-change frames.

    Args:
        video_path: Path to source video file (.mp4, .avi, etc.).
        output_dir: Directory to save extracted timestamped frames.
        target_fps: Frame sampling rate (default ~2.0 fps).
        scene_threshold: Mean grayscale difference threshold to trigger change detection.
        min_changed_pixels: Minimum pixel count with substantial change to flag scene event.
        max_dimension: Maximum width/height to resize frames for efficient storage.
        check_ffmpeg: If True, asserts ffmpeg is installed before processing.

    Returns:
        SampledVideo containing extracted frame paths and metadata.
    """
    if check_ffmpeg:
        check_ffmpeg_installed()

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    video_hash = compute_file_hash(src)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {src}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_raw_frames / source_fps if source_fps > 0 else 0.0

    step_interval = max(1, int(round(source_fps / target_fps)))

    frames: list[FrameInfo] = []
    prev_gray: np.ndarray | None = None
    raw_idx = 0
    sample_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = raw_idx / source_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_regular_sample = (raw_idx % step_interval == 0)
        is_scene_change = False

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_change = float(np.mean(diff))
            changed_pixels = int(np.count_nonzero(diff > 25))

            if mean_change >= scene_threshold or changed_pixels >= min_changed_pixels:
                is_scene_change = True

        prev_gray = gray

        if is_regular_sample or is_scene_change:
            # Resize if exceeding max_dimension
            h, w = frame.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / float(max(h, w))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            frame_filename = f"frame_{sample_idx:04d}_t{timestamp_s:05.2f}s.jpg"
            frame_filepath = dest / frame_filename
            cv2.imwrite(str(frame_filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            frames.append(
                FrameInfo(
                    frame_index=sample_idx,
                    original_frame_number=raw_idx,
                    timestamp_s=round(timestamp_s, 3),
                    file_path=str(frame_filepath),
                    is_scene_change=is_scene_change,
                )
            )
            sample_idx += 1

        raw_idx += 1

    cap.release()

    return SampledVideo(
        video_path=str(src),
        video_hash=video_hash,
        source_fps=round(source_fps, 2),
        duration_s=round(duration_s, 2),
        total_frames_extracted=len(frames),
        frames=frames,
    )
