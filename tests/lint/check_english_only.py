# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that example source and documentation files contain English text only."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_INCLUDED_PREFIXES = ["examples/"]
DEFAULT_EXCLUDED_PATTERNS = []
SOURCE_EXTENSIONS = {".py", ".pyi", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".md"}


def get_git_tracked_files(root_dir: Path) -> list[Path]:
    """Return files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Error: failed to get git tracked files: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)

    return [
        root_dir / line
        for line in result.stdout.strip().split("\n")
        if line and (root_dir / line).is_file()
    ]


def contains_non_english(text: str) -> tuple[bool, list[tuple[int, str]]]:
    """Return whether ``text`` contains common non-English script characters."""
    non_english_pattern = re.compile(
        r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af"
        r"\u0400-\u04ff\u0600-\u06ff\u0590-\u05ff\u0e00-\u0e7f"
        r"\u3000-\u303f\uff01-\uff5e]+"
    )

    violations = []
    for line_number, line in enumerate(text.split("\n"), 1):
        matches = non_english_pattern.findall(line)
        if matches:
            violations.append((line_number, ", ".join(matches)))
    return bool(violations), violations


def check_file_english_only(file_path: Path) -> tuple[bool, list[tuple[int, str]]]:
    """Check one file for common non-English script characters."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, [(0, f"Could not read file: {exc}")]

    has_non_english, violations = contains_non_english(content)
    return not has_non_english, violations


def main() -> int:
    """Run the English-only check."""
    parser = argparse.ArgumentParser(
        description="Check that example source files and documentation are in English only"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to git repository (default: current directory)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional directory patterns to exclude (can be specified multiple times)",
    )
    args = parser.parse_args()

    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"Error: path '{root_path}' does not exist", file=sys.stderr)
        return 1
    if not (root_path / ".git").exists():
        print(f"Error: '{root_path}' is not a git repository", file=sys.stderr)
        return 1

    excluded_patterns = DEFAULT_EXCLUDED_PATTERNS.copy()
    if args.exclude:
        excluded_patterns.extend(args.exclude)

    files_to_check = []
    for file_path in get_git_tracked_files(root_path):
        relative_path = str(file_path.relative_to(root_path))
        if not any(relative_path.startswith(prefix) for prefix in DEFAULT_INCLUDED_PREFIXES):
            continue
        if any(relative_path.startswith(pattern) for pattern in excluded_patterns):
            continue
        if "/_build/" in relative_path or "_build/" in relative_path:
            continue
        if "/_dump/" in relative_path or "_dump/" in relative_path:
            continue
        if file_path.suffix in SOURCE_EXTENSIONS:
            files_to_check.append(file_path)

    if not files_to_check:
        print("No source files found to check.")
        return 0

    print(f"Checking {len(files_to_check)} file(s) for English-only content...")
    failed_files = []
    passed_files = []

    for file_path in files_to_check:
        is_english_only, violations = check_file_english_only(file_path)
        if is_english_only:
            passed_files.append(file_path)
            if args.verbose:
                print(f"OK {file_path.relative_to(root_path)}")
            continue

        failed_files.append((file_path, violations))
        for line_number, non_english_text in violations[:5]:
            print(f"FAIL {file_path.relative_to(root_path)}:{line_number} {non_english_text}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more line(s) in {file_path.relative_to(root_path)}")

    print()
    print(f"Results: {len(passed_files)} passed, {len(failed_files)} failed")
    if failed_files:
        print("\nFiles with non-English content:")
        for file_path, _ in failed_files:
            print(f"  - {file_path.relative_to(root_path)}")
        print("\nPlease ensure checked source files and documentation are written in English.")
        return 1

    print("\nAll checked source files and documentation are in English.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
