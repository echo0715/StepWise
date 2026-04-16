import json
import os
import re
from pathlib import Path
from collections import defaultdict

def extract_task_description(runtime_log_path):
    """Extract task description from runtime.log file."""
    try:
        with open(runtime_log_path, 'r') as f:
            content = f.read()
        
        # Find the instruction between "Instruction:" and "LLM Response:"
        match = re.search(r'Instruction:\s*\n(.*?)\nLLM Response:', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Error reading {runtime_log_path}: {e}")
    return None

def load_trajectory(traj_path):
    """Load trajectory from jsonl file and return dict of step_num -> response."""
    steps = {}
    try:
        with open(traj_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    step_num = entry.get('step_num')
                    response = entry.get('response', '')
                    if step_num is not None:
                        # Combine all entries for the same step (some steps have multiple actions)
                        if step_num not in steps:
                            steps[step_num] = response
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {traj_path}: {e}")
    return steps

def format_step(step_num, response):
    """Format a single step for the training data."""
    return f"Step {step_num}:\n{response}"

def build_text_for_step(task_description, trajectory, target_step, context_steps=5):
    """Build input text for a specific step with task description and previous steps context."""
    # Start with task description
    parts = [f"Task: {task_description}\n"]
    
    # Add previous steps (up to context_steps)
    start_step = max(1, target_step - context_steps)
    
    for step_num in range(start_step, target_step + 1):
        if step_num in trajectory:
            parts.append(format_step(step_num, trajectory[step_num]))
    
    return "\n".join(parts)

def main():
    base_dir = "/gpfs/radev/home/jw3278/project/GUI_router/BERT_Training"
    evocua_base = "/gpfs/radev/project/cohan/jw3278/GUI_router/BERT_Training/evocua_8b/pyautogui/screenshot/EvoCUA"
    
    # Load the step overlap analysis
    overlap_file = os.path.join(base_dir, "step_overlap_analysis.json")
    print(f"Loading step overlap analysis from {overlap_file}...")
    
    with open(overlap_file, 'r') as f:
        overlap_data = json.load(f)
    
    # Collect all milestone steps (in 3 runs) and non-milestone steps (in 0 runs)
    milestone_steps = {}  # task_id -> list of steps
    non_milestone_steps = {}  # task_id -> list of steps
    
    for task_id, data in overlap_data["steps_in_3_runs"].items():
        milestone_steps[task_id] = data["steps"]
    
    for task_id, data in overlap_data["steps_in_0_runs"].items():
        non_milestone_steps[task_id] = data["steps"]
    
    print(f"Tasks with milestone steps (in 3 runs): {len(milestone_steps)}")
    print(f"Tasks with non-milestone steps (in 0 runs): {len(non_milestone_steps)}")
    
    # Get all task_ids
    all_task_ids = set(milestone_steps.keys()) | set(non_milestone_steps.keys())
    
    # Load file paths from run1 data to get trajectory paths
    run1_dir = os.path.join(base_dir, "my_results_run1")
    task_file_paths = {}
    
    for json_file in Path(run1_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
                task_id = content.get('task_id')
                file_path = content.get('file_path')
                if task_id and file_path:
                    task_file_paths[task_id] = file_path
        except Exception as e:
            continue
    
    print(f"Loaded file paths for {len(task_file_paths)} tasks")
    
    # Build training dataset
    training_data = []
    
    # Track statistics
    stats = {
        "total_milestone_samples": 0,
        "total_non_milestone_samples": 0,
        "tasks_processed": 0,
        "tasks_skipped": 0
    }
    
    print("\nBuilding training dataset...")
    
    for task_id in sorted(all_task_ids):
        if task_id not in task_file_paths:
            stats["tasks_skipped"] += 1
            continue
        
        traj_path = task_file_paths[task_id]
        task_dir = os.path.dirname(traj_path)
        runtime_log_path = os.path.join(task_dir, "runtime.log")
        
        # Extract task description
        task_description = extract_task_description(runtime_log_path)
        if not task_description:
            stats["tasks_skipped"] += 1
            continue
        
        # Load trajectory
        trajectory = load_trajectory(traj_path)
        if not trajectory:
            stats["tasks_skipped"] += 1
            continue
        
        stats["tasks_processed"] += 1
        
        # Add milestone steps (label = 1)
        if task_id in milestone_steps:
            for step_num in milestone_steps[task_id]:
                if step_num in trajectory:
                    text = build_text_for_step(task_description, trajectory, step_num, context_steps=5)
                    training_data.append({
                        "task_id": task_id,
                        "step_num": step_num,
                        "text": text,
                        "label": 1
                    })
                    stats["total_milestone_samples"] += 1
        
        # Add non-milestone steps (label = 0)
        if task_id in non_milestone_steps:
            for step_num in non_milestone_steps[task_id]:
                if step_num in trajectory:
                    text = build_text_for_step(task_description, trajectory, step_num, context_steps=5)
                    training_data.append({
                        "task_id": task_id,
                        "step_num": step_num,
                        "text": text,
                        "label": 0
                    })
                    stats["total_non_milestone_samples"] += 1
        
        if stats["tasks_processed"] % 50 == 0:
            print(f"  Processed {stats['tasks_processed']} tasks...")
    
    # Sort by task_id and step_num
    training_data.sort(key=lambda x: (x["task_id"], x["step_num"]))
    
    # Save training dataset
    output_file = os.path.join(base_dir, "milestone_training_dataset.json")
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Save summary
    summary = {
        "total_samples": len(training_data),
        "milestone_samples": stats["total_milestone_samples"],
        "non_milestone_samples": stats["total_non_milestone_samples"],
        "tasks_processed": stats["tasks_processed"],
        "tasks_skipped": stats["tasks_skipped"],
        "runs_used": ["my_results_run1", "my_results_run2", "my_results_run6"],
        "milestone_criteria": "steps appearing in all 3 runs",
        "non_milestone_criteria": "steps appearing in 0 runs",
        "excluded": "steps appearing in 1 or 2 runs",
        "context_steps": 5
    }
    
    summary_file = os.path.join(base_dir, "milestone_training_dataset_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING DATASET BUILD COMPLETE")
    print("="*60)
    print(f"\nOutput file: {output_file}")
    print(f"Summary file: {summary_file}")
    print(f"\nStatistics:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Milestone samples (label=1): {stats['total_milestone_samples']}")
    print(f"  Non-milestone samples (label=0): {stats['total_non_milestone_samples']}")
    print(f"  Tasks processed: {stats['tasks_processed']}")
    print(f"  Tasks skipped: {stats['tasks_skipped']}")
    print(f"\nData format:")
    print(f"  - Each sample includes task description + up to 5 previous steps as context")
    print(f"  - Milestone steps (in all 3 runs) are labeled as 1")
    print(f"  - Non-milestone steps (in 0 runs) are labeled as 0")
    print(f"  - Steps in 1 or 2 runs are excluded")

if __name__ == "__main__":
    main()
