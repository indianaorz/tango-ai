#!/usr/bin/env python3
"""
copy_to_clipboard.py

Read all source files from a directory (optionally recursive, with configurable
extensions), concatenate their contents, and copy to the system clipboard.

Usage:
    # Copy all .py files in current directory:
    ./copy_to_clipboard.py

    # Copy all .py and .txt files from /some/dir (recursive):
    ./copy_to_clipboard.py /some/dir -e .py .txt -r
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from shutil import which

def get_clipboard_command():
    """Detect available clipboard utility and return command list."""
    if which("pbcopy"):
        return ["pbcopy"]                     # macOS
    if which("clip"):
        return ["cmd", "/c", "clip"]          # Windows
    if which("xclip"):
        return ["xclip", "-selection", "clipboard"]  # Linux
    if which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    return None

def collect_files(source_dir: Path, extensions, recursive=False):
    """Return sorted list of files under source_dir with given extensions."""
    if recursive:
        files = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix in extensions]
    else:
        files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix in extensions]
    return sorted(files)

def copy_files_to_clipboard(source_dir: Path, extensions, recursive=False):
    """Concatenate selected files and copy to system clipboard."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    files = collect_files(source_dir, extensions, recursive)
    if not files:
        print(f"No matching files found in '{source_dir}'.", file=sys.stderr)
        sys.exit(1)

    combined = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read '{path.name}': {e}", file=sys.stderr)
            continue
        combined.append(text)

    if not combined:
        print("No readable files to copy.", file=sys.stderr)
        sys.exit(1)

    full_text = "\n".join(combined)

    clip_cmd = get_clipboard_command()
    if clip_cmd is None:
        print("Error: No clipboard utility found (pbcopy, clip, xclip, xsel).", file=sys.stderr)
        sys.exit(1)

    proc = subprocess.Popen(clip_cmd, stdin=subprocess.PIPE)
    try:
        proc.communicate(full_text.encode("utf-8"))
    except Exception as e:
        print(f"Error: Failed to write to clipboard: {e}", file=sys.stderr)
        sys.exit(1)

    if proc.returncode != 0:
        print(f"Error: Clipboard command returned {proc.returncode}.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Copied contents of {len(files)} file(s) from '{source_dir}' into the clipboard.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate files with given extensions and copy to clipboard."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=".",
        help="Directory containing files (default: current directory)."
    )
    parser.add_argument(
        "-e", "--ext",
        nargs="+",
        default=[".py"],
        help="File extensions to include (e.g. -e .py .txt). Default: .py"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search subdirectories."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    copy_files_to_clipboard(Path(args.source_dir), set(args.ext), args.recursive)

if __name__ == "__main__":
    main()
