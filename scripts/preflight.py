#!/usr/bin/env python3
"""Run preflight verification for submission readiness.

Checks:
1. Pytest suite passes
2. Ruff linter passes
3. No secrets in repo or git history
4. LICENSE file exists
5. .env, weights, or raw private data are not tracked in git
6. demo_bundle exists and is under 25 MB
7. Static landing page builds without errors
8. docs/SUBMISSION.md has no unresolved placeholders (other than video / X post) and checks demo video URL
9. Git repository object size is under 500 MB (via git count-objects -vH)
10. Real footage manifest has >= 6 ready real scenarios with existing files (FAIL by default until filmed)
11. Real evaluation results exist with source: real rows (FAIL by default until executed)
12. Final Meta manifest exists, meta_compliant=true, freeze check passes (FAIL by default until executed)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent


def run_check(title: str, func) -> tuple[bool | str, str]:
    print(f"Checking: {title:<50}", end="", flush=True)
    try:
        status, msg = func()
        if status is True:
            print(" [ PASS ]")
            if msg:
                print(f"   ℹ️  {msg}")
            return True, msg
        elif status == "WARN":
            print(" [ WARN ]")
            if msg:
                print(f"   ⚠️  {msg}")
            return "WARN", msg
        else:
            print(" [ FAIL ]")
            print(f"   ❌ {msg}")
            return False, msg
    except Exception as e:
        print(" [ FAIL ]")
        print(f"   ❌ Exception: {e}")
        return False, str(e)


def check_tests() -> tuple[bool, str]:
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return True, "All tests passed"
    return False, res.stderr or res.stdout[:200]


def check_lint() -> tuple[bool, str]:
    res = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return True, "Zero linter warnings"
    return False, res.stdout[:200]


def check_secrets() -> tuple[bool, str]:
    res = subprocess.run(
        [sys.executable, "scripts/check_secrets.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return True, "No secrets detected in working tree or git history"
    return False, res.stdout


def check_license() -> tuple[bool, str]:
    lic_path = repo_root / "LICENSE"
    if lic_path.exists() and len(lic_path.read_text()) > 50:
        return True, "LICENSE file present"
    return False, "LICENSE file missing or empty"


def check_git_tracked_assets() -> tuple[bool, str]:
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    files = res.stdout.splitlines()
    forbidden = [f for f in files if f == ".env" or f.endswith(".pt") or f.endswith(".pth") or "weights/" in f]
    if forbidden:
        return False, f"Forbidden files tracked in git: {forbidden}"
    return True, "No .env or weights files tracked in git"


def check_demo_bundle() -> tuple[bool, str]:
    bundle_path = repo_root / "demo_bundle"
    if not bundle_path.exists():
        return False, "demo_bundle directory missing"

    # Calculate size
    total_size = sum(f.stat().st_size for f in bundle_path.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    if size_mb > 25.0:
        return False, f"demo_bundle exceeds 25 MB: {size_mb:.2f} MB"
    return True, f"demo_bundle present and compact ({size_mb:.2f} MB < 25 MB)"


def check_git_size() -> tuple[bool, str]:
    """Replace working tree size with .git size from git count-objects -vH, warning above 500 MB."""
    res = subprocess.run(
        ["git", "count-objects", "-vH"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return False, "Failed to execute git count-objects"

    info: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()

    du_res = subprocess.run(
        ["du", "-sm", ".git"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    git_mb = int(du_res.stdout.split()[0]) if du_res.returncode == 0 else 0
    size_pack = info.get("size-pack", "0 KiB")
    size_loose = info.get("size", "0 KiB")

    msg = f".git size: ~{git_mb} MB (pack: {size_pack}, loose: {size_loose})"
    if git_mb > 500:
        return False, f"Git repository exceeds 500 MB threshold: {msg}"
    return True, f"{msg} (< 500 MB threshold)"


def check_site_builder() -> tuple[bool, str]:
    res = subprocess.run(
        [sys.executable, "scripts/build_site.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return True, "Static site generated without errors"
    return False, res.stderr or res.stdout


def check_submission_doc() -> tuple[bool | str, str]:
    sub_path = repo_root / "docs" / "SUBMISSION.md"
    if not sub_path.exists():
        return False, "docs/SUBMISSION.md missing"
    content = sub_path.read_text(encoding="utf-8")

    unresolved = []
    for line in content.splitlines():
        if "TODO" in line:
            unresolved.append(line)
    if unresolved:
        return False, f"Unresolved TODO lines in docs/SUBMISSION.md: {unresolved}"

    # Check demonstration video URL
    video_lines = [
        line for line in content.splitlines()
        if "Demonstration Video" in line or "**Demonstration Video:**" in line
    ]
    if video_lines:
        v_line = video_lines[0]
        if "INSERT_DEMO_VIDEO_URL_HERE" in v_line or not any(p in v_line for p in ["http://", "https://"]):
            return "WARN", "docs/SUBMISSION.md demo video URL is still a placeholder (`[INSERT_DEMO_VIDEO_URL_HERE]`). Must be updated before final submission."

    return True, "docs/SUBMISSION.md ready and free of unresolved placeholders"


def check_real_footage() -> tuple[bool, str]:
    """Verify manifests/real_footage.yaml has at least 6 scenarios with status: ready and source: real, and files exist."""
    manifest_path = repo_root / "manifests" / "real_footage.yaml"
    if not manifest_path.exists():
        return False, "manifests/real_footage.yaml is missing"

    try:
        import yaml
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        return False, f"Failed to parse manifests/real_footage.yaml: {e}"

    runs = data.get("runs", [])
    if not isinstance(runs, list):
        return False, "manifests/real_footage.yaml has no valid 'runs' list"

    ready_real = [
        r for r in runs
        if isinstance(r, dict) and r.get("status") == "ready" and r.get("source") == "real"
    ]

    if len(ready_real) < 6:
        planned_count = sum(1 for r in runs if isinstance(r, dict) and r.get("status") == "planned")
        return False, (
            f"manifests/real_footage.yaml has {len(ready_real)}/6 scenarios marked status: ready with source: real "
            f"({planned_count} currently marked planned). Real footage must be recorded and marked ready."
        )

    missing_files = []
    for r in ready_real:
        v_path = repo_root / r.get("video", "")
        if not v_path.exists():
            missing_files.append(f"{r.get('scenario', 'unnamed')}: {r.get('video')}")

    if missing_files:
        return False, f"Video files missing for ready scenarios: {', '.join(missing_files)}"

    return True, f"{len(ready_real)} real scenarios marked ready and all video files exist"


def check_real_eval() -> tuple[bool, str]:
    """Verify real eval exists (eval_results.json with source: real rows, not all synthetic)."""
    eval_path = repo_root / "eval_results.json"
    if not eval_path.exists():
        return False, "eval_results.json missing. Must run stepproof eval on real_footage.yaml with real footage."

    try:
        data: Any = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to parse eval_results.json: {e}"

    rows = data.get("rows") or data.get("results") or data
    if isinstance(rows, dict):
        rows = rows.get("scenarios") or rows.get("runs") or []
    if not isinstance(rows, list) or len(rows) == 0:
        return False, "eval_results.json contains no evaluation scenario rows"

    real_rows = [r for r in rows if isinstance(r, dict) and r.get("source") == "real"]
    if len(real_rows) == 0:
        return False, "eval_results.json contains only synthetic/dev stand-in rows (0 source: real rows found)."

    return True, f"Real evaluation verified ({len(real_rows)} real scenario rows)"


def check_final_manifest() -> tuple[bool, str]:
    """Verify data/outputs/final/MANIFEST.json exists, every entry meta_compliant=true, and freeze check passes."""
    manifest_path = repo_root / "data" / "outputs" / "final" / "MANIFEST.json"
    if not manifest_path.exists():
        return False, "data/outputs/final/MANIFEST.json missing. Final Meta-only run (make final-run) not executed."

    try:
        data: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to parse data/outputs/final/MANIFEST.json: {e}"

    entries = data.get("entries") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if not entries:
        return False, "data/outputs/final/MANIFEST.json contains no result entries"

    required_keys = ["model", "provider", "endpoint_host", "meta_compliant", "tokens", "cost", "timestamp", "input_hash"]
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return False, f"Entry {i} in MANIFEST.json is not a valid dictionary"
        if not entry.get("meta_compliant"):
            return False, f"Entry {i} ({entry.get('model', 'unknown')}) has meta_compliant={entry.get('meta_compliant')}. Must be True."

        for k in required_keys:
            if k not in entry:
                return False, f"Entry {i} missing required freeze key '{k}'"

        # Freeze check: verify input file hash matches input_hash if input_file is specified
        input_file = entry.get("input_file")
        if input_file:
            in_path = repo_root / input_file
            if not in_path.exists():
                return False, f"Freeze check failed: recorded input file not found: {input_file}"
            current_hash = hashlib.sha256(in_path.read_bytes()).hexdigest()
            if current_hash != entry.get("input_hash"):
                return False, f"Freeze check failed: input hash mismatch for {input_file} (inputs changed after run)"

    return True, f"Final Meta manifest verified ({len(entries)} entries, meta_compliant=True, freeze passed)"


def main() -> int:
    print("=" * 70)
    print("STEPPROOF SUBMISSION PREFLIGHT QUALITY GATE")
    print("=" * 70)

    checks = [
        ("Pytest Test Suite", check_tests),
        ("Ruff Linter", check_lint),
        ("Credentials & Secrets Scan", check_secrets),
        ("Open-Source License", check_license),
        ("Git Tracked Asset Cleanliness", check_git_tracked_assets),
        ("Offline Demo Bundle (< 25 MB)", check_demo_bundle),
        ("Static Site Generation", check_site_builder),
        ("Submission Documentation", check_submission_doc),
        ("Git Repository Object Footprint", check_git_size),
        ("Real Footage Manifest & Files", check_real_footage),
        ("Real Footage Evaluation (eval_results.json)", check_real_eval),
        ("Final Meta Manifest & Freeze Check", check_final_manifest),
    ]

    all_passed = True
    missing_prerequisites: list[tuple[str, str]] = []

    for title, fn in checks:
        status, msg = run_check(title, fn)
        if status is False:
            all_passed = False
            missing_prerequisites.append((title, msg))

    print("=" * 70)
    if all_passed:
        print("🏆 PREFLIGHT STATUS: ALL CHECKS PASSED — SUBMISSION READY")
        print("StepProof is verified and submission-ready. You may safely create tag v1.0-submission.")
        print("=" * 70)
        return 0
    else:
        print("❌ PREFLIGHT STATUS: NOT SUBMISSION-READY (PREREQUISITES MISSING)")
        print("\nThe following prerequisite(s) must be fulfilled before final submission:")
        for title, msg in missing_prerequisites:
            print(f"  • {title}:")
            print(f"    {msg}")
        print("\nPlease fulfill the prerequisites above before tagging v1.0-submission or submitting.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
