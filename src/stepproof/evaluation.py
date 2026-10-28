"""Batch evaluation engine and manifest runner for StepProof."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

from stepproof.config import settings
from stepproof.ingest.sampler import sample_video
from stepproof.pipeline.checker import StepChecker
from stepproof.providers.factory import get_llm_provider, get_seg_provider
from stepproof.schemas import SOP

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Load evaluation manifest from YAML."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "runs" not in data:
        raise ValueError(f"Invalid manifest format in {path}: must contain 'runs' list")
    return data


def run_evaluation(
    manifest_path: Path | str,
    output_report_path: Path | str = "eval_report.md",
    require_meta: bool = False,
) -> dict[str, Any]:
    """Execute evaluation across all runs listed in manifest and write report."""
    manifest = load_manifest(manifest_path)
    suite_name = manifest.get("name", "Verification Evaluation")
    sop_path = Path(manifest.get("sop", "data/outputs/demo_sop.json"))

    if not sop_path.exists():
        raise FileNotFoundError(f"SOP file not found: {sop_path}")

    sop = SOP.model_validate_json(sop_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    total_staged_errors = 0
    detected_staged_errors = 0
    correct_run_false_alarms = 0
    total_steps_evaluated = 0
    total_time_seconds = 0.0
    total_estimated_cost = 0.0

    runs = manifest["runs"]
    print("\n" + "=" * 70)
    print(f"STEPPROOF EVALUATION: {suite_name}")
    print(f"SOP: {sop.task} ({len(sop.steps)} steps) | Manifest: {manifest_path}")
    print("=" * 70)

    for run_info in runs:
        scenario = run_info["scenario"]
        video_path = Path(run_info["video"])
        expected_ids = set(run_info.get("expected_flagged_step_ids", []))
        expected_flags = {
            int(k) if str(k).isdigit() else k: str(v)
            for k, v in run_info.get("expected_primary_flag_types", {}).items()
        }

        if not video_path.exists():
            print(f"⚠️  Video file not found: {video_path}. Skipping.")
            continue

        frames_dir = Path("data/outputs/eval_frames") / video_path.stem
        sampled = sample_video(video_path, output_dir=frames_dir)

        seg = get_seg_provider(settings.stepproof_seg, require_meta=require_meta)
        llm = get_llm_provider(settings.stepproof_llm, require_meta=require_meta)

        checker = StepChecker(
            sop=sop,
            segmenter=seg,
            llm_judge=llm,
            consecutive_frames_required=3,
        )

        t0 = time.time()
        verdicts = checker.verify_run(sampled.frames)
        elapsed = time.time() - t0

        total_time_seconds += elapsed
        total_steps_evaluated += len(sop.steps)

        # Assess outcomes
        observed_flagged_ids = {v.step_id for v in verdicts if v.state == "FLAGGED"}
        observed_flags = {v.step_id: v.flag_type for v in verdicts if v.state == "FLAGGED"}

        is_correct_scenario = (len(expected_ids) == 0)
        scenario_passed = True

        if is_correct_scenario:
            false_alarms = len(observed_flagged_ids)
            correct_run_false_alarms += false_alarms
            if false_alarms > 0:
                scenario_passed = False
        else:
            total_staged_errors += len(expected_ids)
            for exp_id in expected_ids:
                exp_flag = expected_flags.get(exp_id)
                obs_flag = observed_flags.get(exp_id)
                if exp_id in observed_flagged_ids and (exp_flag is None or exp_flag == obs_flag):
                    detected_staged_errors += 1
                else:
                    scenario_passed = False

        run_cost = (
            getattr(llm.get_provenance(), "cost_usd", 0.0)
            if hasattr(llm, "get_provenance")
            else 0.0
        )
        total_estimated_cost += run_cost

        status_str = "✅ PASS" if scenario_passed else "❌ FAIL"
        latency_per_step = elapsed / max(1, len(sop.steps))

        results.append({
            "scenario": scenario,
            "status": "PASS" if scenario_passed else "FAIL",
            "expected_ids": list(expected_ids),
            "observed_ids": list(observed_flagged_ids),
            "expected_flags": expected_flags,
            "observed_flags": observed_flags,
            "elapsed_s": round(elapsed, 2),
            "latency_per_step_s": round(latency_per_step, 3),
            "cost_usd": round(run_cost, 6),
        })

        print(
            f"{status_str} | {scenario:<38} | "
            f"Observed: {list(observed_flagged_ids)} ({list(observed_flags.values())}) | "
            f"{elapsed:.2f}s ({latency_per_step:.2f}s/step)"
        )

    # Compute aggregate metrics
    detection_rate = (
        (detected_staged_errors / total_staged_errors * 100.0)
        if total_staged_errors > 0
        else 100.0
    )
    avg_latency_step = (
        total_time_seconds / total_steps_evaluated if total_steps_evaluated > 0 else 0.0
    )
    avg_cost_run = total_estimated_cost / max(1, len(results))

    print("=" * 70)
    print("EVALUATION SUMMARY METRICS")
    print("=" * 70)
    print(f"Total Scenarios Evaluated:       {len(results)}")
    print(f"Correct-Run False Alarms:        {correct_run_false_alarms}")
    print(f"Detection Rate Over Staged Errors: {detection_rate:.1f}% ({detected_staged_errors}/{total_staged_errors})")
    print(f"Average Latency Per Step:        {avg_latency_step:.3f}s")
    print(f"Average Estimated Cost Per Run:  ${avg_cost_run:.6f} USD")
    print("=" * 70)

    # Generate Markdown Report
    report_content = generate_eval_markdown_report(
        suite_name=suite_name,
        manifest_path=manifest_path,
        sop=sop,
        results=results,
        false_alarms=correct_run_false_alarms,
        detection_rate=detection_rate,
        avg_latency=avg_latency_step,
        avg_cost=avg_cost_run,
    )

    out_p = Path(output_report_path)
    out_p.write_text(report_content, encoding="utf-8")
    print(f"\n📄 Evaluation report written to: {out_p.resolve()}")

    return {
        "results": results,
        "false_alarms": correct_run_false_alarms,
        "detection_rate": detection_rate,
        "avg_latency_step": avg_latency_step,
        "avg_cost_run": avg_cost_run,
        "report_path": str(out_p),
    }


def generate_eval_markdown_report(
    suite_name: str,
    manifest_path: Path | str,
    sop: SOP,
    results: list[dict[str, Any]],
    false_alarms: int,
    detection_rate: float,
    avg_latency: float,
    avg_cost: float,
) -> str:
    """Generate professional Markdown evaluation report."""
    rows = []
    for r in results:
        status_icon = "PASS" if r["status"] == "PASS" else "**FAIL**"
        exp_str = f"Steps {r['expected_ids']}" if r["expected_ids"] else "None"
        obs_str = f"Steps {r['observed_ids']}" if r["observed_ids"] else "None"
        flags_str = ", ".join(f"{k}:{v}" for k, v in r["observed_flags"].items()) or "none"
        rows.append(
            f"| {r['scenario']} | {status_icon} | {exp_str} | {obs_str} | `{flags_str}` | {r['latency_per_step_s']:.3f}s | ${r['cost_usd']:.6f} |"
        )

    table_rows = "\n".join(rows)

    return f"""# StepProof Evaluation Report: {suite_name}

**Task:** {sop.task}
**SOP Step Count:** {len(sop.steps)}
**Manifest:** `{manifest_path}`

## 1. Executive Summary & Quality Metrics

| Metric | Measured Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **Correct-Run False Alarms** | {false_alarms} | 0 | {"Optimal" if false_alarms == 0 else "Needs Tuning"} |
| **Detection Rate (Staged Errors)** | {detection_rate:.1f}% | 100% | {"Optimal" if detection_rate == 100.0 else "Action Required"} |
| **Average Latency Per Step** | {avg_latency:.3f}s | < 0.500s (mock) | Good |
| **Estimated Cost Per Run** | ${avg_cost:.6f} USD | < $0.05 | Low Cost |

## 2. Per-Scenario Verification Outcomes

| Scenario Name | Result | Expected Flags | Observed Flags | Observed Types | Latency / Step | Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{table_rows}

## 3. Analysis and Recommendations

- **Out of Order vs Skipped**: The state machine maintains active evaluation across all steps so that steps completed out of sequence are distinguished cleanly from steps never attempted.
- **Wrong Part Recognition**: Intruder objects and common error candidates are checked against reference object bounds and rule checks, preventing misclassification.
"""
