#!/usr/bin/env python3
"""
Aggregate experiment runs into a paper-ready performance matrix.

Reads every  <data_dir>/<TS>_<LOGIC>/experiment_summary.json  and groups by
control logic (mpc / nn / switch).  Produces:

    runs.csv               One row per run.
    by_logic.csv           Aggregated stats per logic (mean / std / min / max).
    by_logic.md            Markdown table for the paper / report.
    metrics_comparison.png Bar chart of the headline metrics with error bars.
    trajectories.png       Overlay of every run's xy trajectory, coloured by logic.

Metrics aggregated per logic:
    success_rate            fraction of runs with goal_reached == true
    duration_s              window duration (first /cmd_vel → goal-reach)
    travel_distance_m       sum of pose diffs inside the window (note:
                            inflated by SLAM jitter — kept for reference but
                            do not use as the primary speed source)
    avg_speed_m_s           mean of /cmd_vel.linear.x inside the window
                            (= cmd_linear_velocity.mean in the recorder
                            summary). Paper-friendly and bounded by the
                            planner's cap. The recorder's own dist/duration
                            value is preserved as `avg_speed_path_m_s`
                            in runs.csv for debugging.
    cmd_v_max_m_s           peak  /cmd_vel linear command
    cmd_w_max_rad_s         peak |/cmd_vel| angular command
    odom_v_mean_m_s         mean odom-measured linear velocity
    odom_v_max_m_s          peak odom-measured linear velocity
    final_dist_to_goal_m    distance to goal at end of window

Usage
-----
    # Aggregate every run under ~/robot_data/experiments/*  (default)
    python3 analyze_experiments.py

    # Limit to a subfolder (e.g. paper-ready set)
    python3 analyze_experiments.py --data-dir ~/robot_data/experiments/for_paper

    # Only include successful runs in the time/distance averages
    python3 analyze_experiments.py --successful-only-for-time

    # Custom output location
    python3 analyze_experiments.py --out ~/Desktop/exp_analysis
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LOGIC_ORDER  = ['mpc', 'nn', 'switch']
LOGIC_LABEL  = {'mpc': 'MPC only', 'nn': 'NN only', 'switch': 'MPC+NN switching'}
LOGIC_COLOUR = {'mpc': '#E53935', 'nn': '#2196F3', 'switch': '#7B1FA2'}

# Per-run metric extractors. Returns a flat dict from one summary JSON.
def extract_run_metrics(summary: dict, run_folder: str) -> dict:
    odom_v = summary.get('odom_linear_velocity', {})
    odom_w = summary.get('odom_angular_velocity', {})
    cmd_v  = summary.get('cmd_linear_velocity',  {})
    cmd_w  = summary.get('cmd_angular_velocity', {})
    samples = summary.get('samples', {})
    goal_pos = summary.get('goal_position') or [None, None]
    # NOTE: `avg_speed_m_s` here is the mean commanded linear velocity
    # (= cmd_linear_velocity.mean), NOT path length / duration.
    # The path-length/duration variant in the recorder summary is inflated by
    # SLAM jitter — every TF-pose tick adds a few mm of noise, which `np.diff`
    # accumulates as fake motion. The commanded-velocity mean is bounded by
    # the planner's velocity cap and is the right number to report.
    return {
        'run':                  os.path.basename(run_folder.rstrip('/')),
        'logic':                summary.get('logic'),
        'timestamp':            summary.get('timestamp'),
        'goal_reached':         bool(summary.get('goal_reached', False)),
        'duration_s':           float(summary.get('duration_sec', 0.0)),
        'travel_distance_m':    float(summary.get('travel_distance_m', 0.0)),
        'avg_speed_m_s':        float(cmd_v.get('mean', 0.0)),
        'avg_speed_path_m_s':   float(summary.get('avg_speed_m_s', 0.0)),  # legacy / debug
        'goal_x':               goal_pos[0],
        'goal_y':               goal_pos[1],
        'final_dist_to_goal_m': summary.get('final_distance_to_goal_m'),
        'cmd_v_max_m_s':        float(cmd_v.get('max',  0.0)),
        'cmd_v_mean_m_s':       float(cmd_v.get('mean', 0.0)),
        'cmd_w_max_abs_rad_s':  max(abs(float(cmd_w.get('min', 0.0))),
                                    abs(float(cmd_w.get('max', 0.0)))),
        'odom_v_max_m_s':       float(odom_v.get('max',  0.0)),
        'odom_v_mean_m_s':      float(odom_v.get('mean', 0.0)),
        'odom_w_max_abs_rad_s': max(abs(float(odom_w.get('min', 0.0))),
                                    abs(float(odom_w.get('max', 0.0)))),
        'samples_states':       int(samples.get('states', 0)),
        'samples_cmds':         int(samples.get('control_cmds', 0)),
        'samples_lidar':        int(samples.get('lidar_scans', 0)),
    }


def aggregate_by_logic(runs, successful_only_for_time):
    """Group runs by logic and compute mean/std/min/max for each numeric metric.

    `successful_only_for_time` — if True, restrict the time/distance/speed
    metrics to runs with goal_reached == True (a failed run skews them).
    Velocity peaks and goal-reach rate are always computed over all runs.
    """
    by_logic = defaultdict(list)
    for r in runs:
        by_logic[r['logic']].append(r)

    rows = []
    for logic in LOGIC_ORDER:
        if logic not in by_logic:
            continue
        runs_l = by_logic[logic]
        succ = [r for r in runs_l if r['goal_reached']]
        time_basis = succ if successful_only_for_time else runs_l

        def _agg(values):
            arr = np.array([v for v in values if v is not None], dtype=float)
            if len(arr) == 0:
                return {'mean': float('nan'), 'std': float('nan'),
                        'min': float('nan'), 'max': float('nan'), 'n': 0}
            return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=0)),
                    'min': float(arr.min()),  'max': float(arr.max()),  'n': int(len(arr))}

        row = {
            'logic':                  logic,
            'n_runs':                 len(runs_l),
            'n_successful':           len(succ),
            'success_rate':           len(succ) / len(runs_l) if runs_l else 0.0,
            'duration_s':             _agg([r['duration_s']           for r in time_basis]),
            'travel_distance_m':      _agg([r['travel_distance_m']    for r in time_basis]),
            'avg_speed_m_s':          _agg([r['avg_speed_m_s']        for r in time_basis]),
            'cmd_v_max_m_s':          _agg([r['cmd_v_max_m_s']        for r in runs_l]),
            'cmd_w_max_abs_rad_s':    _agg([r['cmd_w_max_abs_rad_s']  for r in runs_l]),
            'odom_v_max_m_s':         _agg([r['odom_v_max_m_s']       for r in runs_l]),
            'odom_v_mean_m_s':        _agg([r['odom_v_mean_m_s']      for r in runs_l]),
            'odom_w_max_abs_rad_s':   _agg([r['odom_w_max_abs_rad_s'] for r in runs_l]),
            'final_dist_to_goal_m':   _agg([r['final_dist_to_goal_m'] for r in runs_l]),
        }
        rows.append(row)
    return rows


def write_runs_csv(runs, path):
    if not runs:
        return
    fields = list(runs[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(runs)


def write_by_logic_csv(rows, path):
    if not rows:
        return
    headers = ['logic', 'n_runs', 'n_successful', 'success_rate']
    metric_keys = [
        'duration_s', 'travel_distance_m', 'avg_speed_m_s',
        'cmd_v_max_m_s', 'cmd_w_max_abs_rad_s',
        'odom_v_max_m_s', 'odom_v_mean_m_s', 'odom_w_max_abs_rad_s',
        'final_dist_to_goal_m',
    ]
    flat_headers = list(headers)
    for k in metric_keys:
        for stat in ('mean', 'std', 'min', 'max', 'n'):
            flat_headers.append(f'{k}_{stat}')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(flat_headers)
        for row in rows:
            out = [row['logic'], row['n_runs'], row['n_successful'], f"{row['success_rate']:.3f}"]
            for k in metric_keys:
                d = row[k]
                out.extend([f"{d['mean']:.4f}", f"{d['std']:.4f}",
                            f"{d['min']:.4f}",  f"{d['max']:.4f}", d['n']])
            w.writerow(out)


def write_markdown(rows, path, successful_only_for_time, n_total):
    """Paper-ready markdown table."""
    def fmt(d, decimals=3):
        if d['n'] == 0 or d['mean'] != d['mean']:  # NaN check
            return '–'
        return f"{d['mean']:.{decimals}f} ± {d['std']:.{decimals}f}"

    lines = [
        '# Experiment Performance Matrix',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        f'Total runs analysed: {n_total}',
        f'Time/distance metrics computed from: '
        f'{"successful runs only" if successful_only_for_time else "all runs"}',
        f'Velocity peaks and success rate computed from: all runs',
        '',
        '## Headline metrics (mean ± std across runs)',
        '',
        '| Metric | ' + ' | '.join(LOGIC_LABEL[r['logic']] for r in rows) + ' |',
        '|---' + '|---' * len(rows) + '|',
    ]

    metric_table = [
        ('Runs (N)',                  lambda r: f"{r['n_runs']}"),
        ('Successful',                lambda r: f"{r['n_successful']} / {r['n_runs']}"),
        ('Success rate',              lambda r: f"{r['success_rate']*100:.1f} %"),
        ('Travel time (s)',           lambda r: fmt(r['duration_s'], 2)),
        ('Travel distance (m)',       lambda r: fmt(r['travel_distance_m'], 2)),
        ('Avg commanded v (m/s)',     lambda r: fmt(r['avg_speed_m_s'], 3)),
        ('Peak commanded v (m/s)',    lambda r: fmt(r['cmd_v_max_m_s'], 3)),
        ('Peak |cmd ω| (rad/s)',      lambda r: fmt(r['cmd_w_max_abs_rad_s'], 3)),
        ('Mean odom v (m/s)',         lambda r: fmt(r['odom_v_mean_m_s'], 3)),
        ('Peak odom v (m/s)',         lambda r: fmt(r['odom_v_max_m_s'], 3)),
        ('Final dist. to goal (m)',   lambda r: fmt(r['final_dist_to_goal_m'], 3)),
    ]
    for label, getter in metric_table:
        lines.append('| ' + label + ' | ' +
                     ' | '.join(getter(r) for r in rows) + ' |')

    lines.append('')
    lines.append('## Per-run details')
    lines.append('')
    lines.append('See `runs.csv` (one row per run) and `by_logic.csv` (full mean/std/min/max table).')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(rows, path):
    """Grouped bar chart with error bars for a few headline metrics."""
    if not rows:
        return
    metrics = [
        ('Success rate (×100)',    lambda r: r['success_rate'] * 100, None),
        ('Travel time (s)',        lambda r: r['duration_s']['mean'],      lambda r: r['duration_s']['std']),
        ('Distance (m)',           lambda r: r['travel_distance_m']['mean'], lambda r: r['travel_distance_m']['std']),
        ('Avg commanded v (m/s)',  lambda r: r['avg_speed_m_s']['mean'],   lambda r: r['avg_speed_m_s']['std']),
        ('Peak commanded v (m/s)', lambda r: r['cmd_v_max_m_s']['mean'],   lambda r: r['cmd_v_max_m_s']['std']),
        ('Peak odom v (m/s)',      lambda r: r['odom_v_max_m_s']['mean'],  lambda r: r['odom_v_max_m_s']['std']),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    logics = [r['logic'] for r in rows]
    colours = [LOGIC_COLOUR[l] for l in logics]
    labels  = [LOGIC_LABEL[l]  for l in logics]
    x = np.arange(len(rows))
    for ax, (title, mean_fn, std_fn) in zip(axes, metrics):
        means = [mean_fn(r) for r in rows]
        if std_fn is not None:
            stds = [std_fn(r) for r in rows]
            ax.bar(x, means, yerr=stds, capsize=8, color=colours, alpha=0.85,
                   edgecolor='black', linewidth=0.8)
        else:
            ax.bar(x, means, color=colours, alpha=0.85,
                   edgecolor='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for xi, m in zip(x, means):
            ax.text(xi, m, f'{m:.2f}', ha='center', va='bottom', fontsize=9)
    fig.suptitle('Performance comparison across control logics',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_trajectories(run_dirs, path):
    """Overlay every run's xy trajectory, coloured by logic."""
    if not run_dirs:
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    seen = set()
    for run_dir, logic in run_dirs:
        ts = os.path.basename(run_dir).split('_')
        # ts is YYYYMMDD_HHMMSS_logic, pre-window timestamp folder
        npz_glob = sorted(glob.glob(os.path.join(run_dir, 'robot_data_*.npz')))
        if not npz_glob:
            continue
        try:
            d = np.load(npz_glob[0])
            pos = d['positions']
            if len(pos) < 2:
                continue
            label = LOGIC_LABEL[logic] if logic not in seen else None
            ax.plot(pos[:, 0], pos[:, 1], color=LOGIC_COLOUR[logic],
                    linewidth=1.0, alpha=0.55, label=label)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=LOGIC_COLOUR[logic],
                    markersize=5, alpha=0.6)
            seen.add(logic)
        except Exception:
            continue
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory overlay — every recorded run', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Aggregate experiment runs into a paper-ready summary.')
    parser.add_argument(
        '--data-dir', default=os.path.expanduser('~/robot_data/experiments'),
        help='Root folder containing <TS>_<LOGIC> subfolders.')
    parser.add_argument(
        '--out', default=None,
        help='Output folder. Default: <data-dir>/analysis_<NOW>/')
    parser.add_argument(
        '--successful-only-for-time', action='store_true',
        help='Compute time/distance/speed averages over successful runs only '
             '(failed runs hit Ctrl+C and skew duration). Default: all runs.')
    parser.add_argument(
        '--logic', default=None,
        choices=['mpc', 'nn', 'switch'],
        help='Optional: include only one logic.')
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_dir = (os.path.expanduser(args.out)
               if args.out is not None
               else os.path.join(data_dir,
                                 f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    os.makedirs(out_dir, exist_ok=True)

    # Discover runs (top-level only; <data_dir>/<ts>_<logic>/experiment_summary.json)
    candidates = sorted(glob.glob(os.path.join(data_dir, '*', 'experiment_summary.json')))
    runs = []
    run_dirs = []
    for path in candidates:
        try:
            summary = json.load(open(path))
        except json.JSONDecodeError:
            print(f'  ! malformed JSON skipped: {path}')
            continue
        run_dir = os.path.dirname(path)
        m = extract_run_metrics(summary, run_dir)
        if args.logic and m['logic'] != args.logic:
            continue
        runs.append(m)
        run_dirs.append((run_dir, m['logic']))

    if not runs:
        print(f'No runs found under {data_dir}.')
        return

    n_by_logic = defaultdict(int)
    for r in runs:
        n_by_logic[r['logic']] += 1
    print(f'Discovered {len(runs)} runs:')
    for logic in LOGIC_ORDER:
        if n_by_logic.get(logic):
            print(f'  {logic}: {n_by_logic[logic]}')

    rows = aggregate_by_logic(runs, args.successful_only_for_time)

    # Write CSVs + Markdown + plots
    write_runs_csv(runs, os.path.join(out_dir, 'runs.csv'))
    write_by_logic_csv(rows, os.path.join(out_dir, 'by_logic.csv'))
    write_markdown(rows, os.path.join(out_dir, 'by_logic.md'),
                   args.successful_only_for_time, len(runs))
    plot_metric_comparison(rows, os.path.join(out_dir, 'metrics_comparison.png'))
    plot_trajectories(run_dirs, os.path.join(out_dir, 'trajectories.png'))

    print(f'\n✅ Analysis written to: {out_dir}')
    for fn in ('runs.csv', 'by_logic.csv', 'by_logic.md',
               'metrics_comparison.png', 'trajectories.png'):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            print(f'   - {fn:30s} ({os.path.getsize(p)} bytes)')

    # Print the markdown table to stdout for instant viewing
    print('\n' + '=' * 70)
    with open(os.path.join(out_dir, 'by_logic.md')) as f:
        print(f.read())


if __name__ == '__main__':
    main()
