"""StepProof CLI: build-sop, ground, check, report, and replay."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from stepproof.cache import SQLiteCache
from stepproof.config import settings
from stepproof.ingest.sampler import sample_video
from stepproof.pipeline.checker import StepChecker
from stepproof.pipeline.grounding import ground_step_objects
from stepproof.pipeline.report import export_report, generate_markdown_report
from stepproof.pipeline.sop_builder import build_sop
from stepproof.provenance import assert_meta_compliant
from stepproof.providers.factory import get_llm_provider, get_seg_provider
from stepproof.schemas import SOP, RunReport

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("stepproof")


def cmd_build_sop(args: argparse.Namespace) -> int:
    """Extract frames and generate structured SOP."""
    video_path = Path(args.video)
    out_path = Path(args.out)
    frames_dir = out_path.parent / "expert_frames"

    logger.info("Sampling expert video: %s", video_path)
    sampled = sample_video(video_path, output_dir=frames_dir)

    llm = get_llm_provider(require_meta=args.require_meta)
    cache = SQLiteCache()

    t0 = time.time()
    sop, prov = build_sop(
        sampled_video=sampled,
        llm_provider=llm,
        output_path=out_path,
        cache=cache,
        mode=args.mode,
    )
    elapsed = time.time() - t0

    print(f"\n✅ SOP generated successfully in {elapsed:.2f}s")
    print(f"Task: {sop.task}")
    print(f"Steps extracted: {len(sop.steps)}")
    print(f"Model Provenance: {prov.provider} / {prov.model_name} [{prov.badge}]")
    print(f"Saved to: {out_path}\n")
    return 0


def cmd_ground(args: argparse.Namespace) -> int:
    """Ground SOP physical objects on keyframes."""
    video_path = Path(args.video)
    sop_path = Path(args.sop)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "expert_frames"

    with open(sop_path, "r", encoding="utf-8") as f:
        sop = SOP.model_validate_json(f.read())

    logger.info("Sampling expert video for grounding: %s", video_path)
    sampled = sample_video(video_path, output_dir=frames_dir)

    segmenter = get_seg_provider(require_meta=args.require_meta)

    logger.info("Grounding objects across %d steps...", len(sop.steps))
    grounded, prov = ground_step_objects(sop, sampled, segmenter, output_dir=out_dir)

    print(f"\n✅ Grounding complete ({prov.badge})")
    for step_id, evs in grounded.items():
        objs = [f"{ev.object} ({ev.mask_area_px}px)" for ev in evs]
        print(f"Step {step_id}: {', '.join(objs) if objs else 'No objects segmented'}")
    print(f"Overlays saved in: {out_dir}\n")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Verify learner demonstration video against SOP."""
    video_path = Path(args.video)
    sop_path = Path(args.sop)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "learner_frames"

    with open(sop_path, "r", encoding="utf-8") as f:
        sop = SOP.model_validate_json(f.read())

    t_ingest_start = time.time()
    logger.info("Sampling learner video: %s", video_path)
    sampled = sample_video(video_path, output_dir=frames_dir)
    t_ingest = time.time() - t_ingest_start

    segmenter = get_seg_provider(require_meta=args.require_meta)
    llm_judge = get_llm_provider(require_meta=args.require_meta)

    checker = StepChecker(
        sop=sop,
        segmenter=segmenter,
        llm_judge=llm_judge,
        consecutive_frames_required=3,
        out_dir=out_dir,
    )

    t_check_start = time.time()
    logger.info("Evaluating learner video against %d steps...", len(sop.steps))
    verdicts = checker.verify_run(sampled.frames)
    t_check = time.time() - t_check_start

    provenance_list = [segmenter.get_provenance(), llm_judge.get_provenance()]
    timings = {"ingest": round(t_ingest, 2), "check": round(t_check, 2)}

    export_report(
        sop_id=sop.task,
        verdicts=verdicts,
        provenance=provenance_list,
        timings=timings,
        cost_estimate={"usd": 0.0},
        output_dir=out_dir,
    )

    print("\n" + "=" * 60)
    print(f"STEPPROOF VERIFICATION RESULTS: {sop.task}")
    print("=" * 60)
    for v in verdicts:
        status_symbol = "✅ DONE" if v.state == "DONE" else f"❌ FLAGGED ({v.flag_type.upper()})"
        print(f"Step {v.step_id:2d}: {status_symbol} - {v.reason}")
    print("=" * 60)
    print(f"Report exported to: {out_dir / 'report.md'}\n")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print or export Markdown report from previous run."""
    run_dir = Path(args.run_dir)
    json_path = run_dir / "report.json"
    if not json_path.exists():
        logger.error("report.json not found in %s", run_dir)
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        report = RunReport.model_validate_json(f.read())

    md_text = generate_markdown_report(report)
    if args.out:
        out_file = Path(args.out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Report written to: {out_file}")
    else:
        print(md_text)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay cached run results with Meta compliance verification."""
    run_dir = Path(args.run_dir)
    json_path = run_dir / "report.json"
    if not json_path.exists():
        logger.error("Cannot replay: %s not found", json_path)
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        report = RunReport.model_validate_json(f.read())

    # Check Meta compliance guard if active
    if settings.stepproof_require_meta or args.require_meta:
        for prov in report.provenance:
            assert_meta_compliant(prov)

    print("\n" + "=" * 60)
    print(f"REPLAYING RUN: {report.sop_id}")
    print("=" * 60)
    for v in report.verdicts:
        print(
            f"Step {v.step_id} | State: {v.state:7s} | Flag: {v.flag_type:12s} | "
            f"Confidence: {v.confidence:.2f}"
        )
        print(f"  Reason: {v.reason}")
        if v.evidence_frame_path:
            print(f"  Evidence: {v.evidence_frame_path}")
    print("=" * 60 + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="stepproof",
        description="StepProof: Video-to-SOP generation and step verification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-sop
    p_build = subparsers.add_parser("build-sop", help="Generate SOP from expert video")
    p_build.add_argument("video", help="Path to expert demonstration video")
    p_build.add_argument("--out", default="data/outputs/sop.json", help="Output path for sop.json")
    p_build.add_argument("--mode", default=None, choices=["live", "cache_only", "refresh"])
    p_build.add_argument("--require-meta", action="store_true", help="Enforce Meta compliance guard")

    # ground
    p_ground = subparsers.add_parser("ground", help="Ground SOP physical objects on keyframes")
    p_ground.add_argument("video", help="Path to expert demonstration video")
    p_ground.add_argument("--sop", required=True, help="Path to golden sop.json")
    p_ground.add_argument("--out-dir", default="data/outputs/grounding", help="Directory for overlays")
    p_ground.add_argument("--require-meta", action="store_true", help="Enforce Meta compliance guard")

    # check
    p_check = subparsers.add_parser("check", help="Verify learner video against SOP")
    p_check.add_argument("video", help="Path to learner demonstration video")
    p_check.add_argument("--sop", required=True, help="Path to golden sop.json")
    p_check.add_argument("--out-dir", default="data/outputs/runs/latest", help="Output run directory")
    p_check.add_argument("--mode", default=None, choices=["live", "cache_only", "refresh"])
    p_check.add_argument("--require-meta", action="store_true", help="Enforce Meta compliance guard")

    # report
    p_report = subparsers.add_parser("report", help="Export or display Markdown report")
    p_report.add_argument("run_dir", help="Path to run output directory containing report.json")
    p_report.add_argument("--out", default=None, help="Optional output path for report.md")

    # replay
    p_replay = subparsers.add_parser("replay", help="Replay cached run results")
    p_replay.add_argument("run_dir", help="Path to run output directory containing report.json")
    p_replay.add_argument("--require-meta", action="store_true", help="Enforce Meta compliance guard")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "build-sop": cmd_build_sop,
        "ground": cmd_ground,
        "check": cmd_check,
        "report": cmd_report,
        "replay": cmd_replay,
    }

    cmd_fn = commands.get(args.command)
    if not cmd_fn:
        parser.print_help()
        return 1

    try:
        return cmd_fn(args)
    except Exception as e:
        logger.error("Execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
