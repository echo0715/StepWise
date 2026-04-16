"""
Quantitative analysis of GUI agent trajectory heterogeneity.

Analyzes both evocua_8b (EvoCUA/Claude-3.5) and qwen3_8b_thinking_results (Qwen3-VL-8B)
to characterize:
  1. Step-count distributions: successful vs. failed trajectories
  2. Action repetition (progress-stall detection)
  3. Action-type composition (click, scroll, type, key, wait, ...)
  4. Stall-onset position within failed trajectories
  5. Per-app success rates for both models

Outputs: trajectory_stats.json (used by plot_trajectory_analysis.py)
"""

import ast
import json
import re
import statistics
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/gpfs/radev/project/cohan/jw3278/GUI_router/BERT_Training")
EVOCUA_ROOT = BASE / "evocua_8b/pyautogui/screenshot/EvoCUA"
QWEN_ROOT   = BASE / "qwen3_8b_thinking_results/pyautogui/screenshot/Qwen/Qwen3-VL-8B-Thinking"

# ---------------------------------------------------------------------------
# Action-type classifier
# ---------------------------------------------------------------------------
def classify_action(action_str: str) -> str:
    """Map a pyautogui action string to a coarse action type."""
    a = action_str.strip().lower()
    if not a or a == "none" or a == "wait":
        return "wait"
    if "scroll" in a:
        return "scroll"
    if "typewrite" in a or "write(" in a:
        return "type"
    if "hotkey" in a or "keydown" in a or "keyup" in a or "press(" in a:
        return "key"
    if "click" in a or "double_click" in a or "right_click" in a:
        return "click"
    if "moveto" in a or "move(" in a or "drag" in a:
        return "move"
    if "screenshot" in a:
        return "screenshot"
    return "other"


def detect_stall(actions: list[str], threshold: int = 3) -> dict:
    """
    Detect progress stalls: runs of ≥ `threshold` consecutive identical actions.
    Returns:
      stall_count          - number of distinct stall runs
      stall_fraction       - fraction of steps inside a stall run
      n_stall_steps        - integer count of steps inside stall runs
      n_active_steps       - steps NOT inside any stall run
      first_stall_pos      - normalised position (0-1) of first stall onset; 1.0 if none
    """
    n = len(actions)
    if n < threshold:
        return {
            "stall_count": 0, "stall_fraction": 0.0,
            "n_stall_steps": 0, "n_active_steps": n,
            "first_stall_pos": 1.0,
        }

    in_stall = [False] * n
    stall_count = 0
    first_stall_pos = 1.0
    i = 0
    while i < n:
        j = i
        while j < n and actions[j] == actions[i]:
            j += 1
        run_len = j - i
        if run_len >= threshold:
            stall_count += 1
            for k in range(i, j):
                in_stall[k] = True
            if first_stall_pos == 1.0:
                first_stall_pos = i / n
        i = j

    n_stall = sum(in_stall)
    return {
        "stall_count":     stall_count,
        "stall_fraction":  n_stall / n,
        "n_stall_steps":   n_stall,
        "n_active_steps":  n - n_stall,
        "first_stall_pos": first_stall_pos,
    }


def action_repetition_rate(actions: list[str]) -> float:
    """Fraction of consecutive pairs that are identical."""
    if len(actions) < 2:
        return 0.0
    reps = sum(1 for a, b in zip(actions, actions[1:]) if a == b)
    return reps / (len(actions) - 1)


# ---------------------------------------------------------------------------
# Terminal-action classifier
# ---------------------------------------------------------------------------
def terminal_reason(actions: list[str]) -> str:
    """
    Classify how a trajectory ended based on the last action token.
      'drift'       - agent called DONE but task score < 1 (silent semantic drift)
      'explicit_fail' - agent explicitly called FAIL (gave up)
      'step_limit'  - hit step cap with a normal pyautogui action
    """
    if not actions:
        return "step_limit"
    last = actions[-1].strip().upper()
    if last == "DONE":
        return "drift"
    if last == "FAIL":
        return "explicit_fail"
    return "step_limit"


# ---------------------------------------------------------------------------
# Evocua-8b loader (uses traj.jsonl — same format as qwen3)
# ---------------------------------------------------------------------------
def load_evocua_dataset():
    """Load all evocua_8b trajectories from traj.jsonl files."""
    results_file = EVOCUA_ROOT / "all_result.json"
    results = ast.literal_eval(results_file.read_text())

    trajs = []
    for app, tasks in results.items():
        app_dir = EVOCUA_ROOT / app
        for task_id, score in tasks.items():
            traj_file = app_dir / task_id / "traj.jsonl"
            if not traj_file.exists():
                continue
            steps = [json.loads(l) for l in traj_file.read_text().strip().splitlines()]
            actions = [s.get("action", "").strip() for s in steps]
            if not actions:
                continue
            trajs.append({
                "model": "evocua_8b",
                "app": app,
                "task_id": task_id,
                "score": score,
                "success": score >= 1.0,
                "partial": 0.0 < score < 1.0,
                "actions": actions,
                "n_steps": len(actions),
            })
    return trajs


# ---------------------------------------------------------------------------
# Qwen3-8B-Thinking parser (parses traj.jsonl)
# ---------------------------------------------------------------------------
def load_qwen3_dataset():
    """Load all qwen3_8b_thinking trajectories."""
    results_file = QWEN_ROOT / "all_result.json"
    results = ast.literal_eval(results_file.read_text())

    trajs = []
    for app, tasks in results.items():
        app_dir = QWEN_ROOT / app
        for task_id, score in tasks.items():
            traj_file = app_dir / task_id / "traj.jsonl"
            if not traj_file.exists():
                continue
            steps = [json.loads(l) for l in traj_file.read_text().strip().splitlines()]
            actions = [s.get("action", "").strip() for s in steps]
            if not actions:
                continue
            trajs.append({
                "model": "qwen3_8b_thinking",
                "app": app,
                "task_id": task_id,
                "score": score,
                "success": score >= 1.0,
                "partial": 0.0 < score < 1.0,
                "actions": actions,
                "n_steps": len(actions),
            })
    return trajs


# ---------------------------------------------------------------------------
# Compute derived statistics for each trajectory
# ---------------------------------------------------------------------------
def enrich(traj: dict) -> dict:
    actions = traj["actions"]
    action_types = [classify_action(a) for a in actions]
    type_counts  = Counter(action_types)
    total        = len(actions)

    stall = detect_stall(actions)

    # Steps that are active (not stalling) but the trajectory still fails.
    # For successful trajectories this is 0 by definition.
    n_active_failing = stall["n_active_steps"] if not traj["success"] else 0

    # How the trajectory terminated
    term = terminal_reason(actions) if not traj["success"] else "success"

    traj.update({
        "action_types":         action_types,
        "type_counts":          dict(type_counts),
        "type_fractions":       {t: c / total for t, c in type_counts.items()},
        "rep_rate":             action_repetition_rate(actions),
        "n_active_failing":     n_active_failing,
        "terminal_reason":      term,
        **stall,
    })
    return traj


# ---------------------------------------------------------------------------
# Per-app aggregated stats
# ---------------------------------------------------------------------------
def app_stats(trajs: list[dict]) -> dict:
    by_app = defaultdict(list)
    for t in trajs:
        by_app[t["app"]].append(t)

    out = {}
    for app, ts in sorted(by_app.items()):
        success = [t for t in ts if t["success"]]
        failed  = [t for t in ts if not t["success"]]

        def mean_steps(lst):
            return statistics.mean(t["n_steps"] for t in lst) if lst else None

        out[app] = {
            "n_total":            len(ts),
            "n_success":          len(success),
            "success_rate":       len(success) / len(ts),
            "mean_steps_success": mean_steps(success),
            "mean_steps_failed":  mean_steps(failed),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading evocua_8b trajectories...")
    evocua = load_evocua_dataset()
    print(f"  Loaded {len(evocua)} trajectories")

    print("Loading qwen3_8b_thinking trajectories...")
    qwen3 = load_qwen3_dataset()
    print(f"  Loaded {len(qwen3)} trajectories")

    print("Computing statistics...")
    all_trajs = []
    for t in evocua + qwen3:
        all_trajs.append(enrich(t))

    # ---------- summary statistics ----------
    def summarise(trajs, label):
        success = [t for t in trajs if t["success"]]
        failed  = [t for t in trajs if not t["success"]]

        def mean(lst, key):
            vals = [t[key] for t in lst if t[key] is not None]
            return statistics.mean(vals) if vals else None

        def median(lst, key):
            vals = sorted(t[key] for t in lst if t[key] is not None)
            return statistics.median(vals) if vals else None

        print(f"\n--- {label} ---")
        print(f"  Total: {len(trajs)}, Success: {len(success)}, Failed: {len(failed)}")
        print(f"  Mean steps (success): {mean(success, 'n_steps'):.1f}")
        print(f"  Mean steps (failed):  {mean(failed,  'n_steps'):.1f}")
        print(f"  Mean rep_rate (success): {mean(success, 'rep_rate'):.3f}")
        print(f"  Mean rep_rate (failed):  {mean(failed,  'rep_rate'):.3f}")
        print(f"  Mean stall_fraction (success): {mean(success, 'stall_fraction'):.3f}")
        print(f"  Mean stall_fraction (failed):  {mean(failed,  'stall_fraction'):.3f}")
        stalled_failed = [t for t in failed if t["stall_count"] > 0]
        drift_failed   = [t for t in failed if t["stall_count"] == 0]
        print(f"  Stalled trajectories in failed:       {len(stalled_failed)}/{len(failed)} "
              f"({100*len(stalled_failed)/max(len(failed),1):.1f}%)")
        print(f"  Active-but-failing (drift) in failed: {len(drift_failed)}/{len(failed)} "
              f"({100*len(drift_failed)/max(len(failed),1):.1f}%)")

        # Cross-tab: stall status × terminal reason
        from collections import Counter as _Counter
        xtab = _Counter((t["stall_count"] > 0, t["terminal_reason"]) for t in failed)
        print(f"  Failure mode cross-tab (stall × terminal):")
        for (stalled, term), cnt in sorted(xtab.items()):
            tag = "stall" if stalled else "no_stall"
            print(f"    {tag:8s} + {term:14s}: {cnt:3d}  ({100*cnt/max(len(failed),1):.1f}%)")

        # Three-way step budget
        total_steps        = sum(t["n_steps"] for t in trajs)
        success_steps      = sum(t["n_steps"] for t in success)
        stall_steps        = sum(t["n_stall_steps"] for t in failed)
        active_fail_steps  = sum(t["n_active_failing"] for t in failed)
        print(f"\n  --- Step budget breakdown (N={total_steps} total LLM calls) ---")
        print(f"  Success steps:                  {success_steps:5d}  ({100*success_steps/total_steps:.1f}%)")
        print(f"  Stall steps (failed):           {stall_steps:5d}  ({100*stall_steps/total_steps:.1f}%)")
        print(f"  Active-failing steps (drift):   {active_fail_steps:5d}  ({100*active_fail_steps/total_steps:.1f}%)")

    summarise([t for t in all_trajs if t["model"] == "evocua_8b"], "EvoCUA-8B (Claude)")
    summarise([t for t in all_trajs if t["model"] == "qwen3_8b_thinking"], "Qwen3-VL-8B-Thinking")
    summarise(all_trajs, "Combined")

    # ---------- type fraction averages ----------
    ALL_TYPES = ["click", "scroll", "type", "key", "wait", "move", "other"]
    for model_name, trajs in [
        ("evocua_8b", [t for t in all_trajs if t["model"] == "evocua_8b"]),
        ("qwen3_8b_thinking", [t for t in all_trajs if t["model"] == "qwen3_8b_thinking"]),
    ]:
        print(f"\nAction-type fractions ({model_name}):")
        for outcome, subset in [("success", [t for t in trajs if t["success"]]),
                                 ("failed",  [t for t in trajs if not t["success"]])]:
            # Include 0 for trajectories missing a type
            type_means = {typ: [t["type_fractions"].get(typ, 0.0) for t in subset]
                          for typ in ALL_TYPES}
            print(f"  {outcome} (n={len(subset)}):")
            for typ in ALL_TYPES:
                print(f"    {typ:12s}: {statistics.mean(type_means[typ]):.3f}")

    # ---------- write output ----------
    # Strip raw actions before saving (saves space)
    summary = []
    for t in all_trajs:
        row = {k: v for k, v in t.items() if k not in ("actions", "action_types")}
        summary.append(row)

    def step_budget(trajs):
        success = [t for t in trajs if t["success"]]
        failed  = [t for t in trajs if not t["success"]]
        total   = sum(t["n_steps"] for t in trajs)
        return {
            "total_steps":       total,
            "success_steps":     sum(t["n_steps"]          for t in success),
            "stall_steps":       sum(t["n_stall_steps"]    for t in failed),
            "active_fail_steps": sum(t["n_active_failing"] for t in failed),
        }

    out = {
        "trajectories": summary,
        "app_stats": {
            "evocua_8b":         app_stats([t for t in all_trajs if t["model"] == "evocua_8b"]),
            "qwen3_8b_thinking": app_stats([t for t in all_trajs if t["model"] == "qwen3_8b_thinking"]),
        },
        "step_budget": {
            "evocua_8b":         step_budget([t for t in all_trajs if t["model"] == "evocua_8b"]),
            "qwen3_8b_thinking": step_budget([t for t in all_trajs if t["model"] == "qwen3_8b_thinking"]),
        },
    }

    out_path = BASE / "trajectory_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved statistics to {out_path}")


if __name__ == "__main__":
    main()
