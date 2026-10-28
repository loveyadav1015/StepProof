"""Generate structured Markdown and JSON reports for verification runs."""

from __future__ import annotations

import logging
from pathlib import Path

from stepproof.provenance import ProvenanceRecord
from stepproof.schemas import RunReport, StepVerdict

logger = logging.getLogger(__name__)


def generate_markdown_report(report: RunReport) -> str:
    """Format a RunReport into a clean, human-readable Markdown summary."""
    total_steps = len(report.verdicts)
    done_steps = sum(1 for v in report.verdicts if v.state == "DONE")
    flagged_steps = sum(1 for v in report.verdicts if v.state == "FLAGGED")

    status_icon = "✅ PASSED" if flagged_steps == 0 else "⚠️ ISSUES DETECTED"

    lines = [
        f"# StepProof Verification Report: {report.sop_id}",
        "",
        f"**Run Status:** {status_icon} ({done_steps}/{total_steps} steps passed, {flagged_steps} flagged)",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Total Steps | {total_steps} |",
        f"| Steps Verified (DONE) | {done_steps} |",
        f"| Steps Flagged | {flagged_steps} |",
    ]

    # Timings
    if report.timings:
        total_time = sum(report.timings.values())
        lines.append(f"| Total Pipeline Time | {total_time:.2f}s |")
        for k, v in report.timings.items():
            lines.append(f"| Stage Timing: {k} | {v:.2f}s |")

    # Cost
    if isinstance(report.cost_estimate, dict):
        usd_val = report.cost_estimate.get("usd", 0.0)
        lines.append(f"| Estimated Run Cost | ${usd_val:.4f} USD |")
    else:
        lines.append(f"| Estimated Run Cost | ${float(report.cost_estimate):.4f} USD |")

    lines.extend([
        "",
        "## 2. Model Provenance & Compliance",
        "",
        "| Role / Provider | Model Name | Host Endpoint | Meta Badge | Timestamp (UTC) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for p in report.provenance:
        badge = "🟢 **Meta model**" if p.meta_compliant else "⚪ *Dev stand-in*"
        lines.append(
            f"| `{p.provider}` | `{p.model_name}` | `{p.base_url_host}` | {badge} | {p.timestamp[:19]} |"
        )

    lines.extend([
        "",
        "## 3. Step Verdicts & Visual Evidence",
        "",
    ])

    for v in report.verdicts:
        badge_state = (
            "✅ **DONE**"
            if v.state == "DONE"
            else f"❌ **FLAGGED ({v.flag_type.upper()})**"
        )
        lines.append(f"### Step {v.step_id}: {badge_state}")
        lines.append(f"- **Confidence:** {v.confidence:.2f}")
        lines.append(f"- **Reason:** {v.reason}")

        if v.failed_rules:
            lines.append("- **Failed Rules:**")
            for r in v.failed_rules:
                lines.append(f"  - `{r}`")

        if v.evidence_frame_path:
            lines.append(f"- **Evidence Keyframe:** `![Evidence]({v.evidence_frame_path})`")
        lines.append("")

    return "\n".join(lines)


def export_report(
    sop_id: str,
    verdicts: list[StepVerdict],
    provenance: list[ProvenanceRecord],
    timings: dict[str, float] | None = None,
    cost_estimate: dict[str, float] | float | None = None,
    output_dir: Path | str | None = None,
    output_md_path: Path | str | None = None,
) -> RunReport:
    """Build, save, and return a complete RunReport (Markdown and JSON).

    Args:
        sop_id: Identifier of the evaluated task.
        verdicts: Output verdicts for each step.
        provenance: Model provenance records from the run.
        timings: Execution timing breakdown.
        cost_estimate: Cost dictionary or float.
        output_dir: Optional directory to store report.md and report.json.
        output_md_path: Optional explicit file path for report.md.

    Returns:
        Validated RunReport instance.
    """
    report = RunReport(
        sop_id=sop_id,
        verdicts=verdicts,
        timings=timings or {},
        provenance=provenance,
        cost_estimate=cost_estimate or {"usd": 0.0},
    )

    md_content = generate_markdown_report(report)

    if output_md_path:
        md_file = Path(output_md_path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Wrote report to %s", md_file)

    if output_dir:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        if not output_md_path:
            md_file = dest / "report.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_content)

        json_file = dest / "report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    return report
