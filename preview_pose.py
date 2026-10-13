#!/usr/bin/env python3
"""Compatibility entry point; use preview/scripts/preview_pose.py for new commands."""
from pathlib import Path
import runpy


def main():
    runpy.run_path(str(Path(__file__).resolve().parent / "preview/scripts/preview_pose.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
