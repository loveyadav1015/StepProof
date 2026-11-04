"""Tests for scripts/preflight.py submission readiness checks and failure modes."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.preflight import (
    check_final_manifest,
    check_git_size,
    check_real_eval,
    check_real_footage,
    check_submission_doc,
)


def test_real_footage_fails_when_scenarios_planned():
    """Verify check_real_footage fails by default when scenarios are status: planned."""
    ok, msg = check_real_footage()
    assert ok is False
    assert "0/6 scenarios marked status: ready" in msg
    assert "planned" in msg


def test_real_eval_fails_when_missing_or_synthetic(tmp_path: Path, monkeypatch):
    """Verify check_real_eval fails when eval_results.json is absent or has no real footage rows."""
    from scripts import preflight

    # 1. Missing file
    monkeypatch.setattr(preflight, "repo_root", tmp_path)
    ok, msg = check_real_eval()
    assert ok is False
    assert "eval_results.json missing" in msg

    # 2. Only synthetic rows
    eval_file = tmp_path / "eval_results.json"
    eval_file.write_text(json.dumps({
        "rows": [
            {"scenario": "Test", "source": "synthetic", "status": "PASS"},
            {"scenario": "Dev", "source": "dev_stand_in", "status": "PASS"},
        ]
    }))
    ok, msg = check_real_eval()
    assert ok is False
    assert "0 source: real rows found" in msg

    # 3. Real rows present
    eval_file.write_text(json.dumps({
        "rows": [
            {"scenario": "Real LED", "source": "real", "status": "PASS"},
        ]
    }))
    ok, msg = check_real_eval()
    assert ok is True
    assert "Real evaluation verified (1 real scenario rows)" in msg


def test_final_manifest_fails_when_missing_or_non_compliant(tmp_path: Path, monkeypatch):
    """Verify check_final_manifest fails if file missing, entries empty, or meta_compliant=false."""
    from scripts import preflight

    monkeypatch.setattr(preflight, "repo_root", tmp_path)
    ok, msg = check_final_manifest()
    assert ok is False
    assert "MANIFEST.json missing" in msg

    final_dir = tmp_path / "data" / "outputs" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = final_dir / "MANIFEST.json"

    # Non-compliant entry
    manifest_file.write_text(json.dumps({
        "entries": [
            {"model": "mock", "meta_compliant": False}
        ]
    }))
    ok, msg = check_final_manifest()
    assert ok is False
    assert "meta_compliant=False" in msg


def test_submission_doc_warns_on_placeholder():
    """Verify check_submission_doc outputs WARN when video URL is still placeholder."""
    status, msg = check_submission_doc()
    assert status == "WARN"
    assert "INSERT_DEMO_VIDEO_URL_HERE" in msg


def test_git_size_check_reports_pack_and_loose():
    """Verify check_git_size returns True under 500 MB and parses pack/loose size."""
    ok, msg = check_git_size()
    assert ok is True
    assert ".git size:" in msg
    assert "< 500 MB threshold" in msg


def test_preflight_cli_fails_by_default_and_prints_prerequisites(monkeypatch, capsys):
    """Verify scripts/preflight.main() exits with 1 and prints missing prerequisites banner."""
    from scripts import preflight

    # Mock check_tests so it doesn't recursively run pytest
    monkeypatch.setattr(preflight, "check_tests", lambda: (True, "Mock tests passed"))

    ret = preflight.main()
    out, _ = capsys.readouterr()

    assert ret == 1
    assert "NOT SUBMISSION-READY (PREREQUISITES MISSING)" in out
    assert "Real Footage Manifest & Files" in out
    assert "Real Footage Evaluation" in out
    assert "Final Meta Manifest & Freeze Check" in out
