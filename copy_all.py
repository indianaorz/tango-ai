#!/usr/bin/env python3
"""
copy_py_to_clipboard.py

Read all .py files (non‑recursively) from a source directory, concatenate their contents,
and copy that combined content to the system clipboard.

Usage:
    # Copy from current directory into clipboard:
    ./copy_py_to_clipboard.py

    # Copy from /some/dir into clipboard:
    ./copy_py_to_clipboard.py /some/dir
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from shutil import which

def get_clipboard_command():
    """
    Detect a clipboard utility on the system and return the appropriate command
    as a list. Supports:
      - Linux: xclip or xsel
      - macOS: pbcopy
      - Windows: clip
    If none is found, returns None.
    """
    # macOS has 'pbcopy' by default
    if which("pbcopy"):
        return ["pbcopy"]
    # Windows has 'clip' in cmd.exe
    if which("clip"):
        # On Windows, 'clip' is a builtin—you may need to invoke it via 'cmd /c clip'
        return ["cmd", "/c", "clip"]
    # Linux: prefer xclip, fallback to xsel
    if which("xclip"):
        # -selection clipboard → affect the X clipboard (not PRIMARY)
        return ["xclip", "-selection", "clipboard"]
    if which("xsel"):
        # --clipboard → affect the X clipboard
        return ["xsel", "--clipboard", "--input"]
    return None

def copy_py_contents_to_clipboard(source_dir: Path):
    """
    Read all .py files directly under source_dir (non‑recursive),
    concatenate their contents (in alphabetical order), and copy that
    to the system clipboard.
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Find all .py files (non‑recursive), sorted by filename
    py_files = sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix == ".py"])
    if not py_files:
        print(f"No .py files found in '{source_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Read & concatenate their contents
    combined = []
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read '{path.name}': {e}", file=sys.stderr)
            continue
        header = ""#f"# ── Begin: {path.name} ──\n"
        footer = ""#f"\n# ── End: {path.name} ──\n\n"
        combined.append(header)
        combined.append(text)
        combined.append(footer)

    if not combined:
        print("No readable .py files to copy.", file=sys.stderr)
        sys.exit(1)

    full_text = "".join(combined)

    # Determine clipboard command
    clip_cmd = get_clipboard_command()
    if clip_cmd is None:
        print(
            "Error: Could not find any clipboard utility on your system.\n"
            "On macOS: pbcopy is required.\n"
            "On Linux: xclip or xsel must be installed.\n"
            "On Windows: clip must be available in PATH.\n",
            file=sys.stderr
        )
        sys.exit(1)

    # Invoke the clipboard command
    proc = subprocess.Popen(clip_cmd, stdin=subprocess.PIPE)
    try:
        proc.communicate(full_text.encode("utf-8"))
    except Exception as e:
        print(f"Error: Failed to write to clipboard: {e}", file=sys.stderr)
        sys.exit(1)

    if proc.returncode != 0:
        print(f"Error: Clipboard command returned non-zero exit status {proc.returncode}.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Successfully copied contents of {len(py_files)} .py file(s) from '{source_dir}' into the clipboard.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate all .py files (non‑recursive) from a directory and copy to clipboard."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=".",
        help="Folder containing .py files (default: current directory)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    copy_py_contents_to_clipboard(Path(args.source_dir))

if __name__ == "__main__":
    main()
