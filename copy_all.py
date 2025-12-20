#!/usr/bin/env python3
"""
copy_code_to_clipboard.py

Read code files from a source directory and copy their concatenated contents
to the system clipboard.

Default behavior:
  - Non-recursive: reads only files directly under source_dir.

Optional behavior:
  - Recurse ONLY into subfolders explicitly listed via --subdir (repeatable).
  - File types included: .py, .html, .css, .js (configurable via CLI).

Usage:
    # Copy from current directory (non-recursive):
    ./copy_code_to_clipboard.py

    # Copy from /some/dir (non-recursive):
    ./copy_code_to_clipboard.py /some/dir

    # Also include specific subfolders recursively:
    ./copy_code_to_clipboard.py . --subdir app --subdir web

    # Customize extensions:
    ./copy_code_to_clipboard.py . --ext py --ext js

Notes:
  - Only the specified subfolders are recursed; no other directories are scanned.
  - Output order is deterministic (sorted by relative path).
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path
from shutil import which
from typing import Iterable, List, Sequence, Set, Tuple


DEFAULT_EXTS = ("py", "html", "css", "js")

# Common directories that are almost never useful to include.
# Applied only within the explicitly-recursed subdirs.
DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__",
    ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache",
    ".tox",
    "node_modules",
    "dist", "build",
    ".idea", ".vscode",
}


def get_clipboard_command() -> List[str] | None:
    """
    Detect a clipboard utility on the system and return the appropriate command
    as a list. Supports:
      - Linux: xclip or xsel
      - macOS: pbcopy
      - Windows: clip (invoked via cmd /c clip)
    If none is found, returns None.
    """
    if which("pbcopy"):
        return ["pbcopy"]
    if which("clip"):
        return ["cmd", "/c", "clip"]
    if which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    return None


def _normalize_exts(exts: Sequence[str]) -> Set[str]:
    """
    Normalize extensions:
      - accept: "py" or ".py"
      - store: "py" (no dot), lowercased
    """
    out: Set[str] = set()
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if e.startswith("."):
            e = e[1:]
        out.add(e)
    return out


def _iter_direct_files(source_dir: Path, exts: Set[str]) -> Iterable[Path]:
    for p in source_dir.iterdir():
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            yield p


def _iter_recursive_files(root: Path, exts: Set[str], ignore_dirs: Set[str]) -> Iterable[Path]:
    # Use rglob, but prune by filtering parts (simple & portable).
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue

        # Prune ignored directories anywhere in the path relative to the recursive root.
        rel_parts = p.relative_to(root).parts
        if any(part in ignore_dirs for part in rel_parts[:-1]):
            continue

        yield p


def collect_files(
    source_dir: Path,
    exts: Set[str],
    recurse_subdirs: Sequence[str],
    ignore_dirs: Set[str],
) -> List[Path]:
    """
    Collect files to copy:
      - Always include direct files under source_dir (non-recursive).
      - Additionally include recursive files under each explicitly listed subdir.
    Deduplicates and sorts deterministically by relative path.
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"'{source_dir}' is not a directory.")

    found: Set[Path] = set()

    # Non-recursive (always)
    for p in _iter_direct_files(source_dir, exts):
        found.add(p.resolve())

    # Recursive (only specified subdirs)
    for sd in recurse_subdirs:
        subdir_path = (source_dir / sd).resolve()
        # Ensure the subdir stays within source_dir (avoid weird path tricks)
        try:
            subdir_path.relative_to(source_dir)
        except ValueError:
            raise ValueError(f"Subdir '{sd}' resolves outside source dir; refusing.") from None

        if not subdir_path.exists():
            raise FileNotFoundError(f"Subdir '{sd}' does not exist under '{source_dir}'.")
        if not subdir_path.is_dir():
            raise NotADirectoryError(f"Subdir '{sd}' is not a directory under '{source_dir}'.")

        for p in _iter_recursive_files(subdir_path, exts, ignore_dirs):
            found.add(p.resolve())

    # Sort by path relative to source_dir for stable output.
    def sort_key(p: Path) -> Tuple[str, str]:
        rel = p.relative_to(source_dir).as_posix()
        return (rel.lower(), rel)

    return sorted(found, key=sort_key)


def build_combined_text(source_dir: Path, paths: Sequence[Path]) -> str:
    """
    Concatenate file contents with clear boundaries and normalized newlines.
    """
    chunks: List[str] = []
    for p in paths:
        rel = p.resolve().relative_to(source_dir.resolve()).as_posix()
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fail fast: copying mangled text is worse than being explicit.
            raise UnicodeDecodeError(
                "utf-8", b"", 0, 1, f"File '{rel}' is not valid UTF-8."
            )
        except Exception as e:
            raise RuntimeError(f"Could not read '{rel}': {e}") from e

        chunks.append(f"# ── Begin: {rel} ──\n")
        chunks.append(text.rstrip("\n") + "\n")
        chunks.append(f"# ── End: {rel} ──\n\n")

    return "".join(chunks).rstrip() + "\n"


def copy_text_to_clipboard(full_text: str) -> None:
    clip_cmd = get_clipboard_command()
    if clip_cmd is None:
        raise RuntimeError(
            "Could not find any clipboard utility on your system.\n"
            "On macOS: pbcopy is required.\n"
            "On Linux: xclip or xsel must be installed.\n"
            "On Windows: clip must be available in PATH.\n"
        )

    proc = subprocess.Popen(clip_cmd, stdin=subprocess.PIPE)
    try:
        proc.communicate(full_text.encode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to write to clipboard: {e}") from e

    if proc.returncode != 0:
        raise RuntimeError(f"Clipboard command returned non-zero exit status {proc.returncode}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate code files from a directory and copy to clipboard."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=".",
        help="Root folder (default: current directory).",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        default=[],
        help="Subfolder (relative to source_dir) to include recursively. Repeatable.",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=list(DEFAULT_EXTS),
        help=f"File extension to include (default: {', '.join(DEFAULT_EXTS)}). Repeatable.",
    )
    parser.add_argument(
        "--no-ignore-defaults",
        action="store_true",
        help="Do not ignore common directories like node_modules/.git when recursing.",
    )
    parser.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="Directory name to ignore during recursion (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    exts = _normalize_exts(args.ext)

    ignore_dirs = set()
    if not args.no_ignore_defaults:
        ignore_dirs |= set(DEFAULT_IGNORE_DIRS)
    ignore_dirs |= set(args.ignore_dir)

    try:
        paths = collect_files(
            source_dir=source_dir,
            exts=exts,
            recurse_subdirs=args.subdir,
            ignore_dirs=ignore_dirs,
        )
        if not paths:
            print(
                f"No matching files found under '{source_dir.resolve()}' "
                f"(direct) plus subdirs={args.subdir}.",
                file=sys.stderr,
            )
            sys.exit(1)

        full_text = build_combined_text(source_dir.resolve(), paths)
        copy_text_to_clipboard(full_text)

        print(
            f"✅ Copied {len(paths)} file(s) to clipboard from '{source_dir.resolve()}'.\n"
            f"   Extensions: {', '.join(sorted(exts))}\n"
            f"   Recursed subdirs: {', '.join(args.subdir) if args.subdir else '(none)'}"
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
