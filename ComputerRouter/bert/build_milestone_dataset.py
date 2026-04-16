import json
import os
import math
import argparse
from pathlib import Path
from collections import defaultdict


def load_run_data(run_dir):
    """Load all JSON files from a run directory. Returns task_id -> set of milestone_steps."""
    data = {}
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Warning: {run_dir} does not exist")
        return data
    for json_file in run_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
                task_id = content.get('task_id')
                milestone_steps = content.get('milestone_steps', [])
                total_steps = content.get('total_steps', 0)
                file_path = content.get('file_path', '')
                if task_id:
                    data[task_id] = {
                        'milestone_steps': set(milestone_steps),
                        'total_steps': total_steps,
                        'file_path': file_path,
                    }
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return data


def load_task_description(task_id, examples_base):
    """Load task instruction from evaluation_examples."""
    for domain_dir in Path(examples_base).iterdir():
        if not domain_dir.is_dir():
            continue
        task_file = domain_dir / f"{task_id}.json"
        if task_file.exists():
            try:
                with open(task_file, 'r') as f:
                    data = json.load(f)
                return data.get('instruction', '')
            except Exception:
                pass
    return None


def load_trajectory(traj_path):
    """Load trajectory from jsonl file. Returns dict of step_num -> response text."""
    steps = {}
    try:
        with open(traj_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    step_num = entry.get('step_num')
                    response = entry.get('response', '')
                    if step_num is not None and step_num not in steps:
                        steps[step_num] = response
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {traj_path}: {e}")
    return steps


def format_step(step_num, response):
    return f"Step {step_num}:\n{response}"


def build_text_for_step(task_description, trajectory, target_step, context_steps=5):
    """Build input text: task description + up to context_steps previous steps."""
    parts = [f"Task: {task_description}\n"]
    start_step = max(1, target_step - context_steps)
    for step_num in range(start_step, target_step + 1):
        if step_num in trajectory:
            parts.append(format_step(step_num, trajectory[step_num]))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Build milestone detection training dataset from multi-run milestone annotations."
    )
    parser.add_argument(
        "--milestone-dir", required=True,
        help=(
            "Base path for milestone annotation directories. "
            "With --num-runs 1, reads directly from this directory. "
            "With --num-runs N (N>1), reads from <dir>_run1 ... <dir>_runN."
        ),
    )
    parser.add_argument(
        "--num-runs", type=int, default=4,
        help="Number of milestone annotation runs to aggregate (default: 4)."
    )
    parser.add_argument(
        "--min-overlap", type=int, default=None,
        help="Minimum runs a step must appear in to count as milestone. Default: ceil(num_runs * 0.75)."
    )
    parser.add_argument(
        "--examples-dir", required=True,
        help="Path to evaluation_examples/examples directory for loading task descriptions."
    )
    parser.add_argument(
        "--output", default="milestone_training_dataset.json",
        help="Output path for the training dataset JSON (default: milestone_training_dataset.json)."
    )
    parser.add_argument(
        "--context-steps", type=int, default=5,
        help="Number of previous steps to include as context (default: 5)."
    )
    args = parser.parse_args()

    if args.min_overlap is None:
        args.min_overlap = max(1, math.ceil(args.num_runs * 0.75))

    # --- Step 1: Load milestone data from runs ---
    print(f"Loading data from {args.num_runs} milestone run(s)...")
    all_runs = {}
    if args.num_runs == 1:
        all_runs["run1"] = load_run_data(args.milestone_dir)
        print(f"  {args.milestone_dir}: {len(all_runs['run1'])} tasks loaded")
    else:
        for i in range(1, args.num_runs + 1):
            run_dir = f"{args.milestone_dir}_run{i}"
            run_name = f"run{i}"
            all_runs[run_name] = load_run_data(run_dir)
            print(f"  {run_dir}: {len(all_runs[run_name])} tasks loaded")

    # Collect all task IDs
    all_task_ids = set()
    for run_data in all_runs.values():
        all_task_ids.update(run_data.keys())
    print(f"\nTotal unique task IDs: {len(all_task_ids)}")
    print(f"Min overlap threshold: {args.min_overlap} out of {args.num_runs} runs")

    # --- Step 2: Compute step overlap across runs ---
    task_meta = {}
    step_overlap = defaultdict(lambda: defaultdict(int))

    for run_name, run_data in all_runs.items():
        for task_id, info in run_data.items():
            if task_id not in task_meta:
                task_meta[task_id] = {
                    'file_path': info['file_path'],
                    'total_steps': info['total_steps'],
                }
            for step in info['milestone_steps']:
                step_overlap[task_id][step] += 1

    # Classify steps
    milestone_steps = defaultdict(list)
    excluded_steps = defaultdict(set)

    for task_id in all_task_ids:
        for step, count in step_overlap[task_id].items():
            if count >= args.min_overlap:
                milestone_steps[task_id].append(step)
            else:
                excluded_steps[task_id].add(step)

    total_milestone = sum(len(v) for v in milestone_steps.values())
    total_excluded = sum(len(v) for v in excluded_steps.values())
    print(f"\nMilestone steps (in {args.min_overlap}+ runs): {total_milestone}")
    if args.min_overlap > 1:
        print(f"Excluded steps (in 1-{args.min_overlap - 1} runs): {total_excluded}")
    else:
        print(f"Excluded steps: {total_excluded}")

    # --- Step 3: Build training dataset ---
    training_data = []
    stats = {
        "total_milestone_samples": 0,
        "total_non_milestone_samples": 0,
        "tasks_processed": 0,
        "tasks_skipped_no_description": 0,
        "tasks_skipped_no_trajectory": 0,
    }

    print("\nBuilding training dataset...")
    for task_id in sorted(all_task_ids):
        meta = task_meta.get(task_id)
        if not meta:
            continue

        task_description = load_task_description(task_id, args.examples_dir)
        if not task_description:
            stats["tasks_skipped_no_description"] += 1
            continue

        traj_path = meta['file_path']
        trajectory = load_trajectory(traj_path)
        if not trajectory:
            stats["tasks_skipped_no_trajectory"] += 1
            continue

        stats["tasks_processed"] += 1
        task_milestone_set = set(milestone_steps.get(task_id, []))
        task_excluded_set = excluded_steps.get(task_id, set())

        for step_num in sorted(trajectory.keys()):
            if step_num in task_excluded_set:
                continue

            text = build_text_for_step(task_description, trajectory, step_num, context_steps=args.context_steps)
            label = 1 if step_num in task_milestone_set else 0

            training_data.append({
                "task_id": task_id,
                "step_num": step_num,
                "text": text,
                "label": label,
            })

            if label == 1:
                stats["total_milestone_samples"] += 1
            else:
                stats["total_non_milestone_samples"] += 1

        if stats["tasks_processed"] % 50 == 0:
            print(f"  Processed {stats['tasks_processed']} tasks...")

    training_data.sort(key=lambda x: (x["task_id"], x["step_num"]))

    # --- Step 4: Save ---
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(training_data, f, indent=2)

    if args.min_overlap > 1:
        excluded_desc = f"steps appearing in 1 to {args.min_overlap - 1} runs"
    else:
        excluded_desc = "none (all flagged steps are included)"

    summary = {
        "total_samples": len(training_data),
        "milestone_samples": stats["total_milestone_samples"],
        "non_milestone_samples": stats["total_non_milestone_samples"],
        "tasks_processed": stats["tasks_processed"],
        "tasks_skipped_no_description": stats["tasks_skipped_no_description"],
        "tasks_skipped_no_trajectory": stats["tasks_skipped_no_trajectory"],
        "num_runs": args.num_runs,
        "min_overlap": args.min_overlap,
        "milestone_criteria": f"steps appearing in {args.min_overlap}+ out of {args.num_runs} runs",
        "non_milestone_criteria": f"steps appearing in 0 out of {args.num_runs} runs (never flagged)",
        "excluded": excluded_desc,
        "context_steps": args.context_steps,
    }

    output_path = Path(args.output)
    summary_file = str(output_path.parent / f"{output_path.stem}_summary{output_path.suffix}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput file: {args.output}")
    print(f"Summary file: {summary_file}")
    print(f"\nStatistics:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Milestone samples (label=1): {stats['total_milestone_samples']}")
    print(f"  Non-milestone samples (label=0): {stats['total_non_milestone_samples']}")
    if stats['total_milestone_samples'] > 0:
        ratio = stats['total_non_milestone_samples'] / stats['total_milestone_samples']
        print(f"  Imbalance ratio (non-milestone:milestone): {ratio:.2f}:1")
    print(f"  Tasks processed: {stats['tasks_processed']}")
    print(f"  Tasks skipped (no description): {stats['tasks_skipped_no_description']}")
    print(f"  Tasks skipped (no trajectory): {stats['tasks_skipped_no_trajectory']}")
    print(f"\nData format:")
    print(f"  - Each sample includes task description + up to {args.context_steps} previous steps as context")
    print(f"  - Milestone steps (in {args.min_overlap}+ of {args.num_runs} runs) are labeled as 1")
    print(f"  - Non-milestone steps (in 0 runs) are labeled as 0")
    if args.min_overlap > 1:
        print(f"  - Steps in 1-{args.min_overlap - 1} runs are excluded")


if __name__ == "__main__":
    main()
