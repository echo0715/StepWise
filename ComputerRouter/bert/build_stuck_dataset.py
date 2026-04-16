import json
import os
import argparse
from pathlib import Path


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


def build_text_for_step(trajectory, target_step, context_steps=5):
    """Build input text: up to context_steps previous steps + current step."""
    parts = []
    start_step = max(1, target_step - context_steps)
    for step_num in range(start_step, target_step + 1):
        if step_num in trajectory:
            parts.append(format_step(step_num, trajectory[step_num]))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Build stuck detection training dataset from trajectory analysis results."
    )
    parser.add_argument(
        "--analysis-dir", required=True,
        help="Directory containing per-task stuck analysis JSONs (output of analyze_trajectories.py --analysis-type stuck)."
    )
    parser.add_argument(
        "--output", default="stuck_training_dataset.json",
        help="Output path for the training dataset JSON (default: stuck_training_dataset.json)."
    )
    parser.add_argument(
        "--context-steps", type=int, default=5,
        help="Number of previous steps to include as context (default: 5)."
    )
    args = parser.parse_args()

    # Load all stuck analysis results
    print(f"Loading stuck analysis results from {args.analysis_dir}...")
    all_tasks = {}
    for json_file in Path(args.analysis_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                task_id = data.get('task_id')
                if task_id:
                    all_tasks[task_id] = {
                        'stuck_steps': set(data.get('stuck_steps', [])),
                        'is_stuck': data.get('is_stuck', False),
                        'file_path': data.get('file_path', ''),
                        'total_steps': data.get('total_steps', 0),
                    }
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    stuck_tasks = sum(1 for t in all_tasks.values() if t['is_stuck'])
    print(f"Loaded {len(all_tasks)} tasks ({stuck_tasks} stuck, {len(all_tasks) - stuck_tasks} not stuck)")

    # Build training dataset
    training_data = []
    stats = {
        "total_stuck_samples": 0,
        "total_non_stuck_samples": 0,
        "tasks_processed": 0,
        "tasks_skipped": 0,
    }

    print("\nBuilding training dataset...")
    for task_id in sorted(all_tasks.keys()):
        info = all_tasks[task_id]
        traj_path = info['file_path']
        trajectory = load_trajectory(traj_path)

        if not trajectory:
            stats["tasks_skipped"] += 1
            continue

        stats["tasks_processed"] += 1
        stuck_set = info['stuck_steps']

        for step_num in sorted(trajectory.keys()):
            text = build_text_for_step(trajectory, step_num, context_steps=args.context_steps)
            label = 1 if step_num in stuck_set else 0

            training_data.append({
                "task_id": task_id,
                "step_num": step_num,
                "text": text,
                "label": label,
            })

            if label == 1:
                stats["total_stuck_samples"] += 1
            else:
                stats["total_non_stuck_samples"] += 1

        if stats["tasks_processed"] % 50 == 0:
            print(f"  Processed {stats['tasks_processed']} tasks...")

    training_data.sort(key=lambda x: (x["task_id"], x["step_num"]))

    # Save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(training_data, f, indent=2)

    summary = {
        "total_samples": len(training_data),
        "stuck_samples": stats["total_stuck_samples"],
        "non_stuck_samples": stats["total_non_stuck_samples"],
        "tasks_processed": stats["tasks_processed"],
        "tasks_skipped": stats["tasks_skipped"],
        "analysis_dir": args.analysis_dir,
        "context_steps": args.context_steps,
    }

    output_path = Path(args.output)
    summary_file = str(output_path.parent / f"{output_path.stem}_summary{output_path.suffix}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("STUCK TRAINING DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput file: {args.output}")
    print(f"Summary file: {summary_file}")
    print(f"\nStatistics:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Stuck samples (label=1): {stats['total_stuck_samples']}")
    print(f"  Non-stuck samples (label=0): {stats['total_non_stuck_samples']}")
    if stats['total_stuck_samples'] > 0:
        ratio = stats['total_non_stuck_samples'] / stats['total_stuck_samples']
        print(f"  Imbalance ratio (non-stuck:stuck): {ratio:.2f}:1")
    print(f"  Tasks processed: {stats['tasks_processed']}")
    print(f"  Tasks skipped: {stats['tasks_skipped']}")
    print(f"\nData format:")
    print(f"  - Each sample includes up to {args.context_steps} previous steps + current step as context")
    print(f"  - Stuck steps are labeled as 1")
    print(f"  - Non-stuck steps are labeled as 0")


if __name__ == "__main__":
    main()
