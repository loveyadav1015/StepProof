#!/usr/bin/env python3
"""Run evaluation under Meta compliance guards with explicit cost preview and budget controls.

Usage:
    python scripts/meta_eval.py manifests/real_footage.yaml [--yes] [--out eval_report.md]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from stepproof.config import settings  # noqa: E402
from stepproof.evaluation import load_manifest, run_evaluation  # noqa: E402
from stepproof.pricing import calculate_cost, calculate_sam_cost  # noqa: E402
from stepproof.schemas import SOP  # noqa: E402


def estimate_manifest_cost(manifest_path: Path) -> dict[str, float]:
    """Calculate estimated cost for running all ready scenarios in manifest."""
    manifest = load_manifest(manifest_path)
    sop_path = Path(manifest.get("sop", "data/outputs/demo_sop.json"))
    num_steps = 6
    if sop_path.exists():
        try:
            sop = SOP.model_validate_json(sop_path.read_text(encoding="utf-8"))
            num_steps = len(sop.steps)
        except Exception:
            pass
    _ = num_steps

    runs = manifest.get("runs", [])
    # Estimate based on: ~30 frames sampled per video, active/next step checked
    # Typical SAM calls per run with change detection: ~10 to 20 calls
    # LLM judge calls: ~1 to 2 calls per run if ambiguous, ~500 in tokens, ~200 out tokens
    num_runs = len(runs)
    est_sam_images = num_runs * 15
    est_judge_calls = num_runs * 2
    est_in_tokens = est_judge_calls * 500
    est_out_tokens = est_judge_calls * 200

    sam_cost = calculate_sam_cost("sam-3.1", num_images=est_sam_images)
    llm_cost = calculate_cost("muse-wonder-1.0", input_tokens=est_in_tokens, output_tokens=est_out_tokens)
    total_est = sam_cost + llm_cost

    return {
        "num_runs": num_runs,
        "est_sam_images": est_sam_images,
        "est_judge_calls": est_judge_calls,
        "sam_cost": sam_cost,
        "llm_cost": llm_cost,
        "total_est": round(total_est, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Meta-compliant batch evaluation with cost confirmation and budget caps."
    )
    parser.add_argument("manifest", help="Path to manifest YAML file")
    parser.add_argument("--out", default="eval_report.md", help="Output path for evaluation report")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Explicit confirmation required to execute live Meta evaluation",
    )
    parser.add_argument(
        "--expect-status",
        default="ready",
        help="Expected status for runnable videos (default: ready)",
    )

    args = parser.parse_args()
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return 1

    estimate = estimate_manifest_cost(manifest_path)
    budget_limit = settings.max_cost_usd
    abort_threshold = budget_limit * 0.80

    print("=" * 70)
    print("STEPPROOF META EVALUATION LAUNCHER (STEPPROOF_REQUIRE_META=1)")
    print("=" * 70)
    print(f"Manifest:          {manifest_path}")
    print(f"Scenarios Count:   {estimate['num_runs']}")
    print(f"Est. SAM Images:   ~{estimate['est_sam_images']}")
    print(f"Est. Judge Calls:  ~{estimate['est_judge_calls']}")
    print(f"Est. Total Cost:   ${estimate['total_est']:.4f} USD")
    print(f"Configured Budget: ${budget_limit:.2f} USD (80% abort threshold: ${abort_threshold:.2f} USD)")
    print("=" * 70)

    if not args.yes:
        print("\n❌ REFUSAL: 'make meta-eval' requires explicit confirmation via '--yes'.")
        print("Live calls against Meta APIs will incur real costs per prices.yaml.")
        print(f"Estimated cost: ~${estimate['total_est']:.4f} USD. Configured MAX_COST_USD: ${budget_limit:.2f} USD.")
        print("To proceed, rerun with:")
        print(f"  python scripts/meta_eval.py {manifest_path} --yes")
        print("Or via make:")
        print("  make meta-eval CONFIRM=--yes")
        print("=" * 70)
        return 1

    # Enforce STEPPROOF_REQUIRE_META=1
    os.environ["STEPPROOF_REQUIRE_META"] = "1"

    print("\n🚀 Confirmation verified (--yes). Executing evaluation under STEPPROOF_REQUIRE_META=1...")
    try:
        run_evaluation(
            manifest_path=manifest_path,
            output_report_path=args.out,
            require_meta=True,
            expect_status=args.expect_status,
        )
        return 0
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
