#!/usr/bin/env python3
"""Scan working tree and git history for potential secret key leaks."""

import re
import subprocess
import sys
from pathlib import Path

# Patterns indicating potential live secrets (ignoring placeholders like 'gsk_your_groq_key')
SECRET_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"sk-ant-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?:api_key|apikey|secret|token)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{25,}['\"]", re.IGNORECASE),
]

IGNORED_PATHS = {".env", ".venv", ".git", ".pytest_cache", ".ruff_cache"}


def check_working_tree() -> list[str]:
    violations = []
    repo_root = Path(__file__).resolve().parent.parent

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(repo_root).parts)
        if rel_parts & IGNORED_PATHS:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            # Skip documentation examples or explicit placeholders
            if "your_key" in line.lower() or "example" in line.lower() or "placeholder" in line.lower():
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{path.relative_to(repo_root)}:{line_no} -> Potential secret match")
    return violations


def check_git_history() -> list[str]:
    violations = []
    # Check commit diffs across history
    try:
        res = subprocess.run(
            ["git", "log", "-p", "--all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        for line in res.stdout.splitlines():
            if not line.startswith("+"):
                continue
            added_line = line[1:]
            if "your_key" in added_line.lower() or "example" in added_line.lower() or "placeholder" in added_line.lower():
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(added_line):
                    violations.append(f"Git history: {added_line.strip()[:60]}...")
    except Exception as e:
        violations.append(f"Could not inspect git history: {e}")
    return violations


def main() -> int:
    print("Running StepProof Secrets & Credentials Security Check...")
    tree_violations = check_working_tree()
    history_violations = check_git_history()

    all_violations = tree_violations + history_violations
    if all_violations:
        print("❌ SECURITY CHECK FAILED: Potential live secrets detected!")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("✅ Secrets check passed: No live API keys detected in repository or git history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
