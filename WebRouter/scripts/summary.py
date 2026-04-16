"""Scan all study folders in results/ and generate a metrics.txt summary.

Fast: reads only summary_info.json per experiment (no pickle loading).
Uses pre-existing metadata CSV for site info (no benchmark instantiation).
"""

import json
import glob
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "metrics.txt"

# Pre-built metadata CSV (already exists after first benchmark run)
try:
    import browsergym.experiments as _bge
    METADATA_CSV = Path(_bge.__file__).parent / "benchmark" / "metadata" / "webarena_verified.csv"
except ImportError:
    METADATA_CSV = Path("nonexistent")  # site breakdown will be skipped


def _load_site_map() -> dict[str, str]:
    """Load task_name -> sites mapping from the metadata CSV."""
    if METADATA_CSV.exists():
        md = pd.read_csv(METADATA_CSV)
        return dict(zip(md["task_name"], md["sites"]))
    return {}


def summarize_folder(study_dir: Path, site_map: dict[str, str]) -> dict | None:
    """Scan experiment subdirs and read summary_info.json for each."""
    exp_dirs = [
        d for d in study_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "summary_info.json").exists()
    ]

    # Also count dirs without summary (pending/in-progress)
    all_exp_dirs = [
        d for d in study_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]

    total = len(all_exp_dirs)
    if total == 0:
        return None

    # --- Progress / liveness tracking ---
    now = time.time()
    summary_mtimes = []  # mtime of each summary_info.json (completion times)
    latest_any_mtime = 0.0  # most recent file modification across all exp dirs

    for d in all_exp_dirs:
        sfile = d / "summary_info.json"
        if sfile.exists():
            mt = sfile.stat().st_mtime
            summary_mtimes.append(mt)
            if mt > latest_any_mtime:
                latest_any_mtime = mt
        else:
            # For in-progress dirs, check the dir mtime itself
            try:
                mt = d.stat().st_mtime
                if mt > latest_any_mtime:
                    latest_any_mtime = mt
            except OSError:
                pass

    n_in_progress = sum(
        1
        for d in all_exp_dirs
        if not (d / "summary_info.json").exists()
    )

    # Throughput: completions in the last hour
    one_hour_ago = now - 3600
    completions_last_hour = sum(1 for mt in summary_mtimes if mt > one_hour_ago)

    results = []
    for exp_dir in exp_dirs:
        summary_file = exp_dir / "summary_info.json"
        try:
            with open(summary_file) as f:
                info = json.load(f)
            # Extract task name from dir name: ..._on_<task_name>_<seed>
            dirname = exp_dir.name
            # Pattern: timestamp_AgentName_on_taskname_seed
            match = re.search(r"_on_(.+)_\d+$", dirname)
            task_name = match.group(1) if match else dirname
            info["task_name"] = task_name
            info["exp_dir"] = dirname
            results.append(info)
        except Exception:
            continue

    completed = len(results)
    if completed == 0:
        return None

    rewards = [r.get("cum_reward", 0) or 0 for r in results]
    n_success = sum(1 for r in results if (r.get("cum_reward") or 0) > 0)
    n_err = sum(1 for r in results if r.get("err_msg") is not None)
    steps = [r["n_steps"] for r in results if r.get("n_steps") is not None]
    costs = [r.get("stats.cum_cost", 0) or 0 for r in results]
    step_times = [r.get("stats.cum_step_elapsed", 0) or 0 for r in results]
    agent_times = [r.get("stats.cum_agent_elapsed", 0) or 0 for r in results]
    input_toks = [r.get("stats.cum_input_tokens", 0) or 0 for r in results]
    output_toks = [r.get("stats.cum_output_tokens", 0) or 0 for r in results]

    avg_reward = np.mean(rewards)
    # Std error over all tasks (pending tasks count as 0)
    all_rewards = rewards + [0] * (total - completed)
    mean_all = np.mean(all_rewards)
    std_err = np.sqrt(mean_all * (1 - mean_all) / total) if total > 0 else 0.0

    # Per-site breakdown
    site_stats = {}
    if site_map:
        site_results = {}
        site_totals = {}
        # Count totals from all exp dirs
        for d in all_exp_dirs:
            match = re.search(r"_on_(.+)_\d+$", d.name)
            if match:
                tn = match.group(1)
                site = site_map.get(tn, "unknown")
                site_totals[site] = site_totals.get(site, 0) + 1

        for r in results:
            site = site_map.get(r["task_name"], "unknown")
            if site not in site_results:
                site_results[site] = []
            site_results[site].append(r)

        for site, rlist in site_results.items():
            done = len(rlist)
            successes = sum(1 for r in rlist if (r.get("cum_reward") or 0) > 0)
            s_steps = sum(r.get("n_steps", 0) or 0 for r in rlist)
            s_cost = sum(r.get("stats.cum_cost", 0) or 0 for r in rlist)
            s_env = sum(r.get("stats.cum_step_elapsed", 0) or 0 for r in rlist)
            s_agent = sum(r.get("stats.cum_agent_elapsed", 0) or 0 for r in rlist)
            s_in_tok = sum(r.get("stats.cum_input_tokens", 0) or 0 for r in rlist)
            s_out_tok = sum(r.get("stats.cum_output_tokens", 0) or 0 for r in rlist)
            site_stats[site] = {
                "success_rate": successes / done if done > 0 else 0.0,
                "completed": done,
                "total": site_totals.get(site, done),
                "total_steps": s_steps,
                "avg_steps": s_steps / done if done > 0 else 0.0,
                "cost": s_cost,
                "task_cost": s_cost / done if done > 0 else 0.0,
                "avg_step_cost": s_cost / s_steps if s_steps > 0 else 0.0,
                "avg_step_time": (s_env + s_agent) / s_steps if s_steps > 0 else 0.0,
                "avg_e2e_time": (s_env + s_agent) / done if done > 0 else 0.0,
                "task_in_tok": s_in_tok / done if done > 0 else 0.0,
                "task_out_tok": s_out_tok / done if done > 0 else 0.0,
                "step_input_tok": s_in_tok / s_steps if s_steps > 0 else 0.0,
                "step_output_tok": s_out_tok / s_steps if s_steps > 0 else 0.0,
            }

    return {
        "total": total,
        "completed": completed,
        "pending": total - completed,
        "in_progress": n_in_progress,
        "last_activity_ts": latest_any_mtime if latest_any_mtime > 0 else None,
        "last_completion_ts": max(summary_mtimes) if summary_mtimes else None,
        "staleness_sec": (now - max(summary_mtimes)) if summary_mtimes else None,
        "completions_last_hour": completions_last_hour,
        "success_count": n_success,
        "success_rate": n_success / total,
        "success_rate_completed": n_success / completed if completed > 0 else 0.0,
        "avg_reward": avg_reward,
        "std_err": std_err,
        "n_errors": n_err,
        "cum_cost": sum(costs),
        "total_steps": sum(steps),
        "avg_steps": np.mean(steps) if steps else 0.0,
        "cum_step_time": sum(step_times),
        "cum_agent_time": sum(agent_times),
        "avg_step_time": sum(step_times) / sum(steps) if sum(steps) > 0 else 0.0,
        "avg_agent_time": sum(agent_times) / sum(steps) if sum(steps) > 0 else 0.0,
        "avg_task_time": np.mean(step_times) if step_times else 0.0,
        "total_input_tok": int(sum(input_toks)),
        "total_output_tok": int(sum(output_toks)),
        "site_stats": site_stats,
    }


def main():
    study_dirs = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ) if RESULTS_DIR.exists() else []

    if not study_dirs:
        print(f"No study folders found in {RESULTS_DIR}")
        return

    site_map = _load_site_map()

    lines = []
    lines.append("AgentLab Metrics Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    for study_dir in study_dirs:
        folder_name = study_dir.name
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Study: {folder_name}")
        lines.append("─" * 80)

        stats = summarize_folder(study_dir, site_map)

        if stats is None:
            lines.append("  No completed tasks found.")
            continue
        if "error" in stats:
            lines.append(f"  Error loading: {stats['error']}")
            continue

        # --- Progress / liveness ---
        last_act = stats.get("last_activity_ts")
        last_comp = stats.get("last_completion_ts")
        staleness = stats.get("staleness_sec")
        throughput = stats.get("completions_last_hour", 0)
        in_prog = stats.get("in_progress", 0)

        if last_act:
            lines.append(f"  Last activity:   {datetime.fromtimestamp(last_act).strftime('%Y-%m-%d %H:%M:%S')}")
        if last_comp:
            lines.append(f"  Last completion: {datetime.fromtimestamp(last_comp).strftime('%Y-%m-%d %H:%M:%S')}")
        if staleness is not None:
            if staleness < 60:
                stale_str = f"{staleness:.0f}s ago"
            elif staleness < 3600:
                stale_str = f"{staleness / 60:.1f}m ago"
            else:
                stale_str = f"{staleness / 3600:.1f}h ago"
            status_flag = ""
            if in_prog > 0 and staleness > 1800:
                status_flag = "  ** POSSIBLY STUCK **"
            elif in_prog == 0 and stats["pending"] == 0:
                status_flag = "  (finished)"
            lines.append(f"  Staleness:       {stale_str}{status_flag}")
        lines.append(f"  Throughput:      {throughput} completions in last hour")
        lines.append(f"  In-progress:     {in_prog} tasks currently running")
        lines.append(f"")
        lines.append(f"  Tasks:        {stats['completed']}/{stats['total']} completed, {stats['pending']} pending, {stats['n_errors']} errors")
        lines.append(f"  Success rate: {stats['success_rate']:.1%} (over all {stats['total']} tasks)")
        lines.append(f"  Success rate: {stats['success_rate_completed']:.1%} (over {stats['completed']} completed only)")
        lines.append(f"  Avg reward:   {stats['avg_reward']:.3f} +/- {stats['std_err']:.3f}")
        lines.append(f"  Successes:    {stats['success_count']}/{stats['total']}")
        lines.append(f"  Total cost:   ${stats['cum_cost']:.4f}")
        lines.append(f"  Total tokens: {stats['total_input_tok']:,d} in, {stats['total_output_tok']:,d} out")
        lines.append(f"")
        lines.append(f"  Steps:        {stats['total_steps']} total, {stats['avg_steps']:.1f} avg/task")
        cum_h, cum_m = divmod(stats['cum_step_time'], 3600)
        cum_m = cum_m / 60
        agent_h, agent_m = divmod(stats['cum_agent_time'], 3600)
        agent_m = agent_m / 60
        lines.append(f"  Env time:     {cum_h:.0f}h{cum_m:.0f}m (action exec + observation)")
        lines.append(f"  Agent time:   {agent_h:.0f}h{agent_m:.0f}m (LLM calls)")
        lines.append(f"  Per step:     {stats['avg_step_time']:.1f}s env, {stats['avg_agent_time']:.1f}s agent")
        lines.append(f"  Per task:     {stats['avg_task_time']:.0f}s avg")

        if stats["site_stats"]:
            lines.append(f"\n  Per-site breakdown:")
            hdr = (f"    {'site':20s} {'success':>8s}  {'tasks':>7s}  {'steps':>7s}  {'cost':>8s}"
                   f"  {'task_steps':>10s}  {'task_cost':>10s}  {'task_time':>9s}  {'task_in_tok':>12s}  {'task_out_tok':>13s}"
                   f"  {'step_cost':>10s}  {'step_time':>10s}  {'step_in_tok':>12s}  {'step_out_tok':>13s}")
            sep = (f"    {'─'*20} {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}"
                   f"  {'─'*10}  {'─'*10}  {'─'*9}  {'─'*12}  {'─'*13}"
                   f"  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*13}")
            lines.append(hdr)
            lines.append(sep)
            for site, ss in sorted(stats["site_stats"].items()):
                lines.append(
                    f"    {site:20s} {ss['success_rate']:>7.1%}  "
                    f"{ss['completed']:>3d}/{ss['total']:<3d}  "
                    f"{ss['total_steps']:>7d}  "
                    f"${ss['cost']:>7.2f}  "
                    f"{ss['avg_steps']:>9.1f}  "
                    f"${ss['task_cost']:>9.4f}  "
                    f"{ss['avg_e2e_time']:>7.0f}s  "
                    f"{ss['task_in_tok']:>11,.0f}  "
                    f"{ss['task_out_tok']:>12,.0f}  "
                    f"${ss['avg_step_cost']:>9.4f}  "
                    f"{ss['avg_step_time']:>9.1f}s  "
                    f"{ss['step_input_tok']:>11.0f}  "
                    f"{ss['step_output_tok']:>12.0f}"
                )
            # Total row
            t_done = stats["completed"]
            t_total = stats["total"]
            t_steps = stats["total_steps"]
            t_cost = stats["cum_cost"]
            t_succ = stats["success_count"] / t_done if t_done > 0 else 0.0
            t_in = stats["total_input_tok"]
            t_out = stats["total_output_tok"]
            t_time = (stats["cum_step_time"] + stats["cum_agent_time"])
            lines.append(sep)
            lines.append(
                f"    {'TOTAL':20s} {t_succ:>7.1%}  "
                f"{t_done:>3d}/{t_total:<3d}  "
                f"{t_steps:>7d}  "
                f"${t_cost:>7.2f}  "
                f"{t_steps / t_done if t_done else 0:>9.1f}  "
                f"${t_cost / t_done if t_done else 0:>9.4f}  "
                f"{t_time / t_done if t_done else 0:>7.0f}s  "
                f"{t_in / t_done if t_done else 0:>11,.0f}  "
                f"{t_out / t_done if t_done else 0:>12,.0f}  "
                f"${t_cost / t_steps if t_steps else 0:>9.4f}  "
                f"{t_time / t_steps if t_steps else 0:>9.1f}s  "
                f"{t_in / t_steps if t_steps else 0:>11.0f}  "
                f"{t_out / t_steps if t_steps else 0:>12.0f}"
            )

    lines.append(f"\n{'=' * 80}")

    output = "\n".join(lines)
    print(output)

    with open(OUTPUT_FILE, "w") as f:
        f.write(output + "\n")


if __name__ == "__main__":
    main()
