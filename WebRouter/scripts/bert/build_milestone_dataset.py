#!/usr/bin/env python3
"""Build a per-step BERT training dataset for milestone detection.

Creates training samples from GPT analysis results (milestone labels) and
WebArena trajectory step data. Unlike the stuck detector, the milestone BERT
*does* include the user query/goal in each sample's text.

Supports multi-run consensus: when multiple analysis runs exist, a step is
labeled as a milestone only if it appears in a sufficient number of runs
(configurable via --min-consensus). Ambiguous steps (present in some but
not enough runs) are excluded from the dataset.

Train/test split is performed BY TASK (not by step) to avoid data leakage.

Output is compatible with train_router.py (JSON list with "text", "label", "split").
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *  # noqa: E402, F403

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_ANALYSIS_DIR = get_milestone_analysis_dir(OSS20B_RESULTS_DIR)
DEFAULT_OUTPUT = get_milestone_dataset_json(OSS20B_RESULTS_DIR)
DEFAULT_OUTPUT_STATS = get_milestone_dataset_stats_json(OSS20B_RESULTS_DIR)
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_SPLIT_MODE = "task"


# ── Helpers ──────────────────────────────────────────────────────────────────


def extract_goal_short(goal: str) -> str:
    """Extract the core task instruction from the full goal text.

    The goal text often contains a long JSON schema after a separator.
    We only want the human-readable instruction before that.
    """
    if not goal:
        return ""
    for sep in ["\n---", "\n===", "\nYour response should"]:
        if sep in goal:
            return goal[: goal.index(sep)].strip()
    return goal[:500].strip()


def load_analysis_results(analysis_dir: Path) -> list[dict]:
    """Load all milestone analysis JSON files from a single directory.

    Returns a list of analysis dicts, each with at least:
      - task_name
      - exp_dir
      - milestone_steps
    """
    results = []
    if not analysis_dir.is_dir():
        return results
    for f in sorted(analysis_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if "task_name" in data and "milestone_steps" in data:
            results.append(data)
    return results


def load_multi_run_results(base_dir: Path, num_runs: int) -> dict[str, list[dict]]:
    """Load analysis results from multiple consensus runs.

    Looks for directories named ``{base_dir}_run1``, ``{base_dir}_run2``, etc.

    Returns a dict mapping task_name -> list of analysis dicts (one per run
    that contains that task).
    """
    task_runs: dict[str, list[dict]] = defaultdict(list)
    for run_idx in range(1, num_runs + 1):
        run_dir = base_dir.parent / f"{base_dir.name}_run{run_idx}"
        if not run_dir.is_dir():
            print(f"  WARNING: Run directory not found: {run_dir}")
            continue
        results = load_analysis_results(run_dir)
        print(f"  Run {run_idx}: {len(results)} tasks from {run_dir}")
        for r in results:
            task_runs[r["task_name"]].append(r)
    return task_runs


def compute_consensus_labels(
    task_runs: dict[str, list[dict]],
    num_runs: int,
    min_consensus: int,
) -> dict[str, dict]:
    """Compute consensus milestone labels from multiple analysis runs.

    For each task and step:
      - If the step appears as milestone in >= min_consensus runs -> milestone (label=1)
      - If the step appears in 0 runs -> non-milestone (label=0)
      - Otherwise -> excluded (ambiguous)

    Returns a dict mapping task_name -> {
        "milestone_steps": set of confirmed milestone step numbers,
        "excluded_steps": set of ambiguous step numbers,
        "exp_dir": str,
    }
    """
    consensus = {}
    for task_name, runs in task_runs.items():
        # Count how many runs each step appears as milestone
        step_counts: Counter = Counter()
        all_steps: set[int] = set()
        exp_dir = None
        for run in runs:
            if exp_dir is None:
                exp_dir = run.get("exp_dir", "")
            for s in run.get("milestone_steps", []):
                step_counts[int(s)] += 1
                all_steps.add(int(s))

        milestone_steps = set()
        excluded_steps = set()
        for s in all_steps:
            if step_counts[s] >= min_consensus:
                milestone_steps.add(s)
            elif step_counts[s] > 0:
                excluded_steps.add(s)

        consensus[task_name] = {
            "milestone_steps": milestone_steps,
            "excluded_steps": excluded_steps,
            "exp_dir": exp_dir or "",
        }
    return consensus


def load_steps(exp_dir: Path) -> list[dict]:
    """Load all step JSON files from an experiment directory.

    Returns a sorted list of step dicts (excluding terminal steps where
    action and think are both None).
    """
    steps = []
    for f in sorted(exp_dir.glob("step_*.json")):
        try:
            step_data = json.loads(f.read_text())
        except Exception:
            continue
        # Skip the terminal observation step (action=null, think=null)
        if step_data.get("action") is None and step_data.get("think") is None:
            continue
        steps.append(step_data)
    steps.sort(key=lambda s: s.get("step", 0))
    return steps


def build_step_text(
    goal_short: str,
    steps: list[dict],
    current_idx: int,
    context_window: int = 5,
) -> str:
    """Build BERT input text from goal + sliding window of trajectory context.

    Format::

        Task: Get the top-1 best-selling product name(s) in 2022
        Step 0: [click('227')] We need to view the order history...
        Step 1: [click('1530')] I need to navigate to the orders page...

    Parameters
    ----------
    goal_short : str
        Core task instruction (extracted from full goal).
    steps : list[dict]
        All step dicts for this task.
    current_idx : int
        Index of the current step (inclusive end of the window).
    context_window : int
        Maximum number of steps in the sliding window.

    Returns
    -------
    str
        BERT input text.
    """
    parts = [f"Task: {goal_short}"]

    # Context window: up to context_window previous steps + current step
    start_idx = max(0, current_idx - context_window + 1)
    for i in range(start_idx, current_idx + 1):
        step = steps[i]
        action = step.get("action", "")
        think = step.get("think", "")
        step_num = step.get("step", i)

        if think:
            parts.append(f"Step {step_num}: [{action}] {think}")
        else:
            parts.append(f"Step {step_num}: [{action}]")

    return "\n".join(parts)


def find_exp_dir_for_task(results_dir: Path, task_name: str) -> Path | None:
    """Find the experiment directory with the best reward for a given task.

    Scans results_dir for directories matching the task_name, picks the one
    with the highest cum_reward.
    """
    candidates = []
    for d in results_dir.iterdir():
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
            continue
        tn = get_task_name_from_dir(d.name)
        if tn != task_name:
            continue
        summary_file = d / "summary_info.json"
        if not summary_file.exists():
            continue
        try:
            info = json.loads(summary_file.read_text())
            candidates.append((d, info.get("cum_reward", 0.0)))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def assign_splits(task_names: list[str], test_fraction: float, seed: int) -> dict[str, str]:
    """Assign train/test split by task with deterministic shuffling.

    Parameters
    ----------
    task_names : list[str]
        Unique task names.
    test_fraction : float
        Fraction of tasks to assign to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, str]
        Mapping from task_name to "train" or "test".
    """
    rng = random.Random(seed)
    names = sorted(task_names)
    rng.shuffle(names)
    n_test = max(1, int(len(names) * test_fraction))
    test_set = set(names[:n_test])
    return {t: ("test" if t in test_set else "train") for t in names}


def compute_stats(dataset: list[dict], n_excluded: int) -> dict:
    """Compute summary statistics for the dataset."""
    total = len(dataset)
    if total == 0:
        return {"total_samples": 0}

    labels = [d["label"] for d in dataset]
    splits = [d["split"] for d in dataset]
    task_names = sorted(set(d["task_name"] for d in dataset))

    label_counts = Counter(labels)
    split_counts = Counter(splits)

    # Per-split label counts
    split_label_counts: dict[str, dict[str, int]] = {}
    for split_name in ["train", "test"]:
        split_items = [d for d in dataset if d["split"] == split_name]
        lc = Counter(d["label"] for d in split_items)
        split_label_counts[split_name] = {
            "total": len(split_items),
            "milestone": lc.get(1, 0),
            "non_milestone": lc.get(0, 0),
        }

    # Tasks per split
    tasks_per_split: dict[str, int] = {}
    for split_name in ["train", "test"]:
        split_tasks = set(d["task_name"] for d in dataset if d["split"] == split_name)
        tasks_per_split[split_name] = len(split_tasks)

    # Steps per task
    steps_per_task = Counter(d["task_name"] for d in dataset)
    step_counts = list(steps_per_task.values())

    # Milestones per task
    milestone_per_task = Counter(d["task_name"] for d in dataset if d["label"] == 1)

    return {
        "total_samples": total,
        "total_tasks": len(task_names),
        "milestone_samples": label_counts.get(1, 0),
        "non_milestone_samples": label_counts.get(0, 0),
        "excluded_ambiguous_samples": n_excluded,
        "milestone_fraction": label_counts.get(1, 0) / total if total else 0,
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_counts": split_label_counts,
        "tasks_per_split": tasks_per_split,
        "steps_per_task": {
            "mean": sum(step_counts) / len(step_counts),
            "min": min(step_counts),
            "max": max(step_counts),
        },
        "milestones_per_task": {
            "mean": (sum(milestone_per_task.values()) / len(task_names) if task_names else 0),
            "tasks_with_milestones": len(milestone_per_task),
            "tasks_without_milestones": len(task_names) - len(milestone_per_task),
        },
    }


def print_summary(stats: dict) -> None:
    """Print a human-readable summary."""
    print(f"\n{'=' * 70}")
    print("  Milestone Detection Dataset Summary")
    print(f"{'=' * 70}")

    print(f"\n  Total samples:         {stats['total_samples']}")
    print(f"  Total tasks:           {stats['total_tasks']}")
    print(
        f"  Milestone (label=1):   {stats['milestone_samples']}"
        f"  ({stats['milestone_fraction']:.1%})"
    )
    print(f"  Non-milestone (label=0): {stats['non_milestone_samples']}")
    if stats.get("excluded_ambiguous_samples", 0) > 0:
        print(f"  Excluded (ambiguous):  {stats['excluded_ambiguous_samples']}")

    if "split_label_counts" in stats:
        print("\n  Per-split breakdown:")
        for split_name in ["train", "test"]:
            sc = stats["split_label_counts"].get(split_name, {})
            n_tasks = stats["tasks_per_split"].get(split_name, 0)
            print(
                f"    {split_name:6s}:  {sc.get('total', 0):5d} samples"
                f"  ({n_tasks} tasks)"
                f"  [milestone={sc.get('milestone', 0)},"
                f" non-milestone={sc.get('non_milestone', 0)}]"
            )

    spt = stats.get("steps_per_task", {})
    if spt:
        print(f"\n  Steps/task:  mean={spt['mean']:.1f}" f"  min={spt['min']}  max={spt['max']}")

    mpt = stats.get("milestones_per_task", {})
    if mpt:
        print(
            f"  Milestones/task:  mean={mpt['mean']:.1f}"
            f"  with_milestones={mpt['tasks_with_milestones']}"
            f"  without={mpt['tasks_without_milestones']}"
        )

    print(f"\n{'=' * 70}\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build per-step BERT training dataset for milestone detection."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Path to milestone analysis directory "
            "(default: results/data/<run-name>/milestone_analysis)"
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help=(
            "Number of consensus runs. If >1, looks for {analysis-dir}_run1, "
            "{analysis-dir}_run2, etc. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=None,
        help=(
            "Minimum runs for a step to count as milestone. "
            "Default: 1 for single run, 3 for 4 runs."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=OSS20B_RESULTS_DIR,
        help="Path to gpt-oss-20b results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output dataset JSON "
            "(default: results/data/<run-name>/milestone_training_dataset.json)"
        ),
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=None,
        help=(
            "Output stats JSON "
            "(default: results/data/<run-name>/milestone_training_dataset_stats.json)"
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help="Number of steps in sliding context window (default: %(default)s)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of tasks for test split (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for train/test split (default: %(default)s)",
    )
    parser.add_argument(
        "--split-mode",
        choices=["task", "sample"],
        default=DEFAULT_SPLIT_MODE,
        help=(
            "Split dataset by task (original behavior) or by individual samples. "
            "Use 'sample' to allow the same task in both train and test "
            "(default: %(default)s)"
        ),
    )
    args = parser.parse_args()

    if args.analysis_dir is None:
        args.analysis_dir = get_milestone_analysis_dir(args.results_dir)
    if args.output is None:
        args.output = get_milestone_dataset_json(args.results_dir)
    if args.output_stats is None:
        args.output_stats = get_milestone_dataset_stats_json(args.results_dir)

    # Resolve min_consensus default
    if args.min_consensus is None:
        args.min_consensus = 1 if args.num_runs == 1 else 3

    print(f"Milestone dataset builder")
    print(f"  Analysis dir:    {args.analysis_dir}")
    print(f"  Num runs:        {args.num_runs}")
    print(f"  Min consensus:   {args.min_consensus}")
    print(f"  Results dir:     {args.results_dir}")
    print(f"  Context window:  {args.context_window}")
    print(f"  Test fraction:   {args.test_fraction}")
    print(f"  Seed:            {args.seed}")
    print(f"  Split mode:      {args.split_mode}")

    # ── 1. Load analysis results ─────────────────────────────────────────
    if args.num_runs == 1:
        # Single run: load directly from analysis_dir
        print(f"\nLoading single-run analysis from {args.analysis_dir}")
        analysis_results = load_analysis_results(args.analysis_dir)
        print(f"  Loaded {len(analysis_results)} task analyses")

        # Convert to the same format as consensus output
        task_labels: dict[str, dict] = {}
        for r in analysis_results:
            task_name = r["task_name"]
            milestone_steps = set(int(s) for s in r.get("milestone_steps", []))
            task_labels[task_name] = {
                "milestone_steps": milestone_steps,
                "excluded_steps": set(),
                "exp_dir": r.get("exp_dir", ""),
            }
    else:
        # Multi-run consensus
        print(f"\nLoading {args.num_runs}-run consensus analysis")
        task_runs = load_multi_run_results(args.analysis_dir, args.num_runs)
        print(f"  Found {len(task_runs)} unique tasks across runs")

        task_labels = compute_consensus_labels(task_runs, args.num_runs, args.min_consensus)
        total_milestones = sum(len(v["milestone_steps"]) for v in task_labels.values())
        total_excluded = sum(len(v["excluded_steps"]) for v in task_labels.values())
        print(f"  Consensus milestone steps: {total_milestones}")
        print(f"  Excluded ambiguous steps:  {total_excluded}")

    if not task_labels:
        print("ERROR: No analysis results loaded. Check --analysis-dir.")
        sys.exit(1)

    # ── 2. Build experiment directory index ───────────────────────────────
    print(f"\nIndexing experiment directories from {args.results_dir}")
    task_to_dir: dict[str, Path] = {}
    for d in args.results_dir.iterdir():
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
            continue
        tn = get_task_name_from_dir(d.name)
        if tn is None:
            continue
        summary_file = d / "summary_info.json"
        if not summary_file.exists():
            continue
        try:
            info = json.loads(summary_file.read_text())
            reward = info.get("cum_reward", 0.0)
        except Exception:
            reward = 0.0
        if tn not in task_to_dir:
            task_to_dir[tn] = d
        else:
            # Keep the one with highest reward
            existing_summary = task_to_dir[tn] / "summary_info.json"
            try:
                existing_reward = json.loads(existing_summary.read_text()).get("cum_reward", 0.0)
            except Exception:
                existing_reward = 0.0
            if reward > existing_reward:
                task_to_dir[tn] = d
    print(f"  Indexed {len(task_to_dir)} tasks with experiment data")

    # ── 3. Build per-step dataset ────────────────────────────────────────
    print(f"\nBuilding per-step dataset...")
    dataset = []
    tasks_processed = 0
    tasks_skipped_no_exp = 0
    tasks_skipped_no_steps = 0
    total_excluded = 0

    for task_name, label_info in sorted(task_labels.items()):
        milestone_steps = label_info["milestone_steps"]
        excluded_steps = label_info["excluded_steps"]

        # Try to find experiment directory: first from analysis, then from index
        exp_dir = None
        analysis_exp_dir = label_info.get("exp_dir", "")
        if analysis_exp_dir and Path(analysis_exp_dir).is_dir():
            exp_dir = Path(analysis_exp_dir)
        else:
            exp_dir = task_to_dir.get(task_name)

        if exp_dir is None:
            tasks_skipped_no_exp += 1
            continue

        # Load step data
        steps = load_steps(exp_dir)
        if not steps:
            tasks_skipped_no_steps += 1
            continue

        # Extract short goal from step 0
        goal_short = extract_goal_short(steps[0].get("goal", ""))

        # Create one sample per step
        for step_idx in range(len(steps)):
            step_num = steps[step_idx].get("step", step_idx)

            # Skip excluded (ambiguous) steps in consensus mode
            if step_num in excluded_steps:
                total_excluded += 1
                continue

            label = 1 if step_num in milestone_steps else 0
            text = build_step_text(goal_short, steps, step_idx, args.context_window)

            dataset.append(
                {
                    "text": text,
                    "label": label,
                    "split": "",  # assigned below
                    "task_name": task_name,
                    "step_num": step_num,
                }
            )

        tasks_processed += 1

    print(f"  Tasks processed:          {tasks_processed}")
    print(f"  Tasks skipped (no exp):   {tasks_skipped_no_exp}")
    print(f"  Tasks skipped (no steps): {tasks_skipped_no_steps}")
    print(f"  Total samples:            {len(dataset)}")
    if total_excluded > 0:
        print(f"  Excluded ambiguous steps: {total_excluded}")

    if not dataset:
        print("ERROR: No samples generated. Check input data.")
        sys.exit(1)

    # ── 4. Assign train/test split ───────────────────────────────────────
    if args.split_mode == "task":
        all_task_names = sorted(set(d["task_name"] for d in dataset))
        split_map = assign_splits(all_task_names, args.test_fraction, args.seed)
        for sample in dataset:
            sample["split"] = split_map[sample["task_name"]]

        train_count = sum(1 for d in dataset if d["split"] == "train")
        test_count = sum(1 for d in dataset if d["split"] == "test")
        print(f"\n  Train/test split (by task, seed={args.seed}):")
        print(f"    Train: {train_count} samples")
        print(f"    Test:  {test_count} samples")
    else:
        rng = random.Random(args.seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * args.test_fraction))
        test_indices = set(indices[:n_test])
        for idx, sample in enumerate(dataset):
            sample["split"] = "test" if idx in test_indices else "train"

        train_count = sum(1 for d in dataset if d["split"] == "train")
        test_count = sum(1 for d in dataset if d["split"] == "test")
        print(f"\n  Train/test split (by sample, seed={args.seed}):")
        print("    Train/test can share task names in sample split mode")
        print(f"    Train: {train_count} samples")
        print(f"    Test:  {test_count} samples")

    # ── 5. Save dataset ──────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nDataset saved to {args.output}")

    # ── 6. Compute and save stats ────────────────────────────────────────
    stats = compute_stats(dataset, total_excluded)
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {args.output_stats}")

    print_summary(stats)


if __name__ == "__main__":
    main()
