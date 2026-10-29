#!/usr/bin/env python3
"""Adjust the verified training model's displayed pose, without physics steps.

Uses the same frozen model and memory-limited launcher as launch_piano_preview.sh.
Model/configuration changes require a refreshed preview package; this entry never
compiles a scene or falls back to the old interactive dynamics preview.
"""
import argparse
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--show-hands', action=argparse.BooleanOptionalAction, default=True,
                        help='Show both hands initially (default: on); H toggles them')
    parser.add_argument('--pose-controls', action=argparse.BooleanOptionalAction, default=True,
                        help='Show joint pose sliders (default: on); R resets both hands')
    parser.add_argument('--view', type=int, choices=(1, 2, 3),
                        help='Initial camera view; 1/2/3 also switch views in the window')
    parser.add_argument('--duration', '--check-seconds', dest='duration', type=float, default=0,
                        help='Close after this many seconds; 0 keeps the window open')
    parser.add_argument('--screenshot', type=Path, help='Save an image from the desktop preview')
    # Harmless former flags remain accepted; the model is always static/no-arm.
    parser.add_argument('--no-arms', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--paused', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration must be finite and nonnegative')
    launcher = Path(__file__).resolve().parent / 'launch_piano_preview.sh'
    command = ['/bin/bash', str(launcher)]
    if args.pose_controls:
        command.append('--pose-controls')
    if args.show_hands:
        command.append('--show-hands')
    if args.view is not None:
        command.extend(['--view', str(args.view)])
    if args.duration:
        command.extend(['--check-seconds', str(args.duration)])
    if args.screenshot is not None:
        command.extend(['--screenshot', str(args.screenshot.resolve())])
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
