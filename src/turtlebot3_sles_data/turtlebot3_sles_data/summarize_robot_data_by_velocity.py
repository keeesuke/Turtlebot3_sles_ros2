#!/usr/bin/env python3
"""
Summarize robot_data sessions by velocity category (v=0.14 vs v=0.2).

Scans all folders inside robot_data, reads analysis_report.txt for each folder
that has both NPZ files and an analysis report. Sessions are categorized by
folder name: v0p14 or v0.14 -> v=0.14; v0p2 or v0.2 -> v=0.2.

For each category prints:
  1. Number of folders with NPZ + analysis report
  2. Average total distance traveled (m)
  3. Average control input duration (s)
  4. Average linear velocity (m/s)
  5. Average angular velocity (rad/s)
"""

import os
import re
from pathlib import Path
from collections import defaultdict


# Default robot_data location
DEFAULT_ROBOT_DATA_DIR = os.path.join(os.path.expanduser("~"), "robot_data")

# Patterns to categorize sessions (folder name)
# v_limit_haa is encoded as v0p14 / v0p2 in session folder names by robot_data_recorder_MPC_v2
CATEGORY_V014_PATTERNS = ("v0p14", "v0.14", "_0p14", "_0.14")
CATEGORY_V02_PATTERNS = ("v0p2", "v0.2", "_0p2", "_0.2")

# Report line patterns (from analyze_robot_data.py generate_report)
REPORT_PATTERNS = {
    "total_distance": re.compile(r"Total Distance Traveled:\s*([\d.]+)\s*m"),
    "control_duration": re.compile(r"Control Input Duration:\s*([\d.]+)\s*s"),
    "avg_linear_speed": re.compile(r"Average Linear Speed:\s*([\d.]+)\s*m/s"),
    "avg_angular_speed": re.compile(r"Average Angular Speed:\s*([\d.]+)\s*rad/s"),
}


def get_velocity_category(folder_name):
    """
    Return 'v0.14', 'v0.2', or None.
    v0.14 must be checked before v0.2 so that 'v0p2' does not match 'v0p14'.
    """
    name_lower = folder_name.lower()
    for p in CATEGORY_V014_PATTERNS:
        if p in name_lower:
            return "v0.14"
    for p in CATEGORY_V02_PATTERNS:
        if p in name_lower:
            return "v0.2"
    return None


def folder_has_npz_and_report(folder_path):
    """True if folder contains at least one robot_data_*.npz and analysis_report.txt."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return False
    npz_files = list(folder.glob("robot_data_*.npz"))
    report_file = folder / "analysis_report.txt"
    return len(npz_files) > 0 and report_file.is_file()


def parse_analysis_report(report_path):
    """
    Parse analysis_report.txt and return dict with total_distance, control_duration,
    avg_linear_speed, avg_angular_speed. Missing or invalid values are None.
    """
    result = {
        "total_distance": None,
        "control_duration": None,
        "avg_linear_speed": None,
        "avg_angular_speed": None,
    }
    try:
        text = Path(report_path).read_text()
    except Exception:
        return result

    for key, pattern in REPORT_PATTERNS.items():
        m = pattern.search(text)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass
    return result


def collect_session_data(robot_data_dir):
    """
    Scan robot_data_dir for session folders. For each folder that has NPZ + report
    and matches v=0.14 or v=0.2, parse the report and add to the right category.
    Returns: dict category -> list of parsed report dicts (with same keys as parse_analysis_report).
    """
    robot_data_path = Path(robot_data_dir)
    if not robot_data_path.is_dir():
        return defaultdict(list)

    by_category = defaultdict(list)
    for item in robot_data_path.iterdir():
        if not item.is_dir():
            continue
        folder_name = item.name
        cat = get_velocity_category(folder_name)
        if cat is None:
            continue
        if not folder_has_npz_and_report(item):
            continue
        report_path = item / "analysis_report.txt"
        parsed = parse_analysis_report(report_path)
        # Only include if we got at least total_distance and control_duration
        if parsed["total_distance"] is not None and parsed["control_duration"] is not None:
            by_category[cat].append(parsed)
    return by_category


def compute_averages(records):
    """From a list of parsed report dicts, compute averages. None if no records."""
    if not records:
        return None
    n = len(records)
    totals = {}
    for key in ("total_distance", "control_duration", "avg_linear_speed", "avg_angular_speed"):
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            totals[key] = None
        else:
            totals[key] = sum(values) / len(values)
    return totals


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Summarize robot_data sessions by velocity (v=0.14 vs v=0.2)."
    )
    parser.add_argument(
        "robot_data_dir",
        nargs="?",
        default=DEFAULT_ROBOT_DATA_DIR,
        help=f"Path to robot_data root (default: {DEFAULT_ROBOT_DATA_DIR})",
    )
    args = parser.parse_args()

    by_category = collect_session_data(args.robot_data_dir)

    # Order: v0.14, then v0.2
    categories = ["v0.14", "v0.2"]
    for cat in categories:
        records = by_category.get(cat, [])
        count = len(records)
        avgs = compute_averages(records)

        print("=" * 60)
        print(f"Category: v = {cat[1:]} ({cat})")
        print("=" * 60)
        print(f"  1. Number of folders (with NPZ + analysis report): {count}")
        if count == 0:
            print("  2. Average total distance traveled (m): N/A")
            print("  3. Average control input duration (s): N/A")
            print("  4. Average linear velocity (m/s): N/A")
            print("  5. Average angular velocity (rad/s): N/A")
        else:
            print(f"  2. Average total distance traveled (m): {avgs['total_distance']:.3f}")
            print(f"  3. Average control input duration (s): {avgs['control_duration']:.2f}")
            lin = avgs["avg_linear_speed"]
            ang = avgs["avg_angular_speed"]
            print(f"  4. Average linear velocity (m/s): {lin:.3f}" if lin is not None else "  4. Average linear velocity (m/s): N/A")
            print(f"  5. Average angular velocity (rad/s): {ang:.3f}" if ang is not None else "  5. Average angular velocity (rad/s): N/A")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
