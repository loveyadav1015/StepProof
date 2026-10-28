"""Generate synthetic expert and learner demonstration videos for offline testing and demos."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

# Coordinates and colors (BGR) for the demonstration blocks
BLOCK_W, BLOCK_H = 60, 60
SLOTS = {
    "left": (140, 180),
    "center": (260, 180),
    "right": (380, 180),
    "top": (260, 100),
}

COLORS_BGR = {
    "red": (30, 30, 220),
    "green": (40, 200, 40),
    "blue": (220, 70, 30),
    "yellow": (40, 220, 220),
    "purple": (180, 30, 180),  # Wrong part
}


def create_base_canvas(width: int = 640, height: int = 360) -> np.ndarray:
    """Create neutral workbench scene with assembly slot guides."""
    img = np.full((height, width, 3), 235, dtype=np.uint8)  # Light workbench grey

    # Draw table boundary line
    cv2.line(img, (0, 260), (width, 260), (190, 190, 190), 2)

    # Draw slot guide outlines (dashed / faint grey)
    for name, (x, y) in SLOTS.items():
        cv2.rectangle(img, (x, y), (x + BLOCK_W, y + BLOCK_H), (200, 200, 200), 2)
        cv2.putText(
            img,
            name,
            (x + 10, y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    return img


def draw_block(
    canvas: np.ndarray,
    slot_name: str,
    color_name: str,
    label: str = "",
) -> None:
    """Draw a colored block into an assembly slot with shading and text."""
    x, y = SLOTS[slot_name]
    color = COLORS_BGR[color_name]
    cv2.rectangle(canvas, (x, y), (x + BLOCK_W, y + BLOCK_H), color, -1)
    cv2.rectangle(canvas, (x, y), (x + BLOCK_W, y + BLOCK_H), (20, 20, 20), 2)  # border

    display_text = label or color_name[0].upper()
    cv2.putText(
        canvas,
        display_text,
        (x + 22, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_video(
    output_path: Path,
    timeline: list[tuple[float, float, list[tuple[str, str, str]]]],
    fps: int = 10,
    total_seconds: float = 8.0,
    title: str = "",
) -> Path:
    """Render a video with blocks appearing according to timeline intervals.

    timeline: list of (start_s, end_s, [(slot_name, color_name, label), ...])
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    total_frames = int(total_seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    base_frame = create_base_canvas(width, height)

    for f in range(total_frames):
        t = f / float(fps)
        frame = base_frame.copy()

        # Find active blocks at time t
        for start_s, end_s, blocks in timeline:
            if t >= start_s:
                for slot, color, label in blocks:
                    draw_block(frame, slot, color, label)

        # Header with title and timer
        cv2.putText(
            frame,
            f"{title} [t={t:04.1f}s]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"Generated synthetic video: {output_path} ({total_frames} frames, {total_seconds}s)")
    return output_path


def generate_all_synthetic_videos(base_dir: Path | None = None) -> dict[str, Path]:
    """Generate the full suite of expert and learner synthetic videos."""
    root = base_dir or Path("data")
    expert_dir = root / "expert"
    learner_dir = root / "learner"

    # 1. Expert video: Red (0.5s), Green (2.0s), Blue (4.0s), Yellow (6.0s)
    expert_timeline = [
        (0.5, 8.0, [("left", "red", "R")]),
        (2.0, 8.0, [("center", "green", "G")]),
        (4.0, 8.0, [("right", "blue", "B")]),
        (6.0, 8.0, [("top", "yellow", "Y")]),
    ]
    p_expert = render_video(
        expert_dir / "expert.mp4",
        expert_timeline,
        title="EXPERT DEMO: Block Assembly",
    )

    # 2. Learner: Correct execution
    p_correct = render_video(
        learner_dir / "learner_correct.mp4",
        expert_timeline,
        title="LEARNER: Correct Sequence",
    )

    # 3. Learner: Skipped step (Green block skipped, Blue placed at 2.0s)
    skipped_timeline = [
        (0.5, 8.0, [("left", "red", "R")]),
        # Green is skipped completely!
        (2.5, 8.0, [("right", "blue", "B")]),
        (5.0, 8.0, [("top", "yellow", "Y")]),
    ]
    p_skipped = render_video(
        learner_dir / "learner_skipped.mp4",
        skipped_timeline,
        title="LEARNER: Step 2 Skipped",
    )

    # 4. Learner: Wrong part (Purple block placed instead of Green)
    wrong_part_timeline = [
        (0.5, 8.0, [("left", "red", "R")]),
        (2.0, 8.0, [("center", "purple", "P")]),  # Purple block!
        (4.0, 8.0, [("right", "blue", "B")]),
        (6.0, 8.0, [("top", "yellow", "Y")]),
    ]
    p_wrong = render_video(
        learner_dir / "learner_wrong_part.mp4",
        wrong_part_timeline,
        title="LEARNER: Wrong Part Placed",
    )

    # 5. Learner: Out of order (Blue placed before Green)
    out_of_order_timeline = [
        (0.5, 8.0, [("left", "red", "R")]),
        (2.0, 8.0, [("right", "blue", "B")]),   # Step 3 placed early!
        (4.0, 8.0, [("center", "green", "G")]), # Step 2 placed late!
        (6.0, 8.0, [("top", "yellow", "Y")]),
    ]
    p_ooo = render_video(
        learner_dir / "learner_out_of_order.mp4",
        out_of_order_timeline,
        title="LEARNER: Out of Order",
    )

    return {
        "expert": p_expert,
        "learner_correct": p_correct,
        "learner_skipped": p_skipped,
        "learner_wrong_part": p_wrong,
        "learner_out_of_order": p_ooo,
    }


if __name__ == "__main__":
    generate_all_synthetic_videos()
