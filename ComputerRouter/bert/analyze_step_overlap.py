import json
import os
from collections import defaultdict, Counter
from pathlib import Path

def load_run_data(run_dir):
    """Load all JSON files from a run directory."""
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
                if task_id:
                    data[task_id] = set(milestone_steps)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return data

def main():
    base_dir = "/gpfs/radev/home/jw3278/project/GUI_router/BERT_Training"
    
    # Load data from qwen3 milestone runs 1-4
    print("Loading data from qwen3 milestone runs 1-4...")
    all_runs = {}
    for i in [1, 2, 3, 4]:
        run_name = f"qwen3_milestone_results_run{i}"
        run_dir = os.path.join(base_dir, run_name)
        all_runs[run_name] = load_run_data(run_dir)
        print(f"  {run_name}: {len(all_runs[run_name])} tasks loaded")
    
    # Get all task_ids that appear in at least one run
    all_task_ids = set()
    for run_data in all_runs.values():
        all_task_ids.update(run_data.keys())
    
    print(f"\nTotal unique task_ids across all runs: {len(all_task_ids)}")
    
    # For each step, count how many runs it appears in (across all tasks)
    step_run_counts = Counter()  # step -> count of runs it appears in
    
    # Collect all unique (task_id, step) pairs from each run
    for task_id in all_task_ids:
        # For this task, collect all unique steps from each run
        task_steps_per_run = {}
        
        for run_name, run_data in all_runs.items():
            if task_id in run_data:
                task_steps_per_run[run_name] = run_data[task_id]
        
        # Get all unique steps for this task across all runs
        all_steps_for_task = set()
        for steps in task_steps_per_run.values():
            all_steps_for_task.update(steps)
        
        # For each step in this task, count how many runs it appears in
        for step in all_steps_for_task:
            runs_with_this_step = sum(1 for run_steps in task_steps_per_run.values() 
                                     if step in run_steps)
            step_run_counts[runs_with_this_step] += 1
    
    # Display results
    num_runs_analyzed = len(all_runs)
    print("\n" + "="*60)
    print(f"STEP OVERLAP ANALYSIS ACROSS {num_runs_analyzed} RUNS (qwen3 milestone run1-4)")
    print("="*60)
    print("\nFor each milestone step across all tasks, counting how many")
    print("runs it appears in (i.e., step N in task X appears in how many runs):\n")
    
    for num_runs in sorted(range(1, num_runs_analyzed + 1), reverse=True):
        count = step_run_counts[num_runs]
        percentage = (count / sum(step_run_counts.values()) * 100) if sum(step_run_counts.values()) > 0 else 0
        print(f"Steps appearing in {num_runs} run(s):  {count:5d} ({percentage:.1f}%)")
    
    total_steps = sum(step_run_counts.values())
    print(f"\nTotal step instances analyzed: {total_steps}")
    
    # Additional statistics
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    
    # Count tasks by how many runs they appear in
    task_run_counts = Counter()
    for task_id in all_task_ids:
        runs_with_task = sum(1 for run_data in all_runs.values() if task_id in run_data)
        task_run_counts[runs_with_task] += 1
    
    print("\nTasks by number of runs they appear in:")
    for num_runs in sorted(range(1, num_runs_analyzed + 1), reverse=True):
        count = task_run_counts[num_runs]
        print(f"  Tasks in {num_runs} run(s): {count}")
    
    # Average number of milestone steps per task per run
    print("\nAverage milestone steps per task per run:")
    for run_name, run_data in sorted(all_runs.items()):
        if run_data:
            avg_steps = sum(len(steps) for steps in run_data.values()) / len(run_data)
            print(f"  {run_name}: {avg_steps:.2f} steps/task")
    
    # Detailed breakdown for a sample task (optional)
    print("\n" + "="*60)
    print("SAMPLE TASK BREAKDOWN")
    print("="*60)
    
    # Pick a task that appears in all runs
    sample_task = None
    for task_id in all_task_ids:
        if all(task_id in run_data for run_data in all_runs.values()):
            sample_task = task_id
            break
    
    if sample_task:
        print(f"\nSample task (appears in all runs): {sample_task}")
        print("\nMilestone steps per run:")
        for run_name in sorted(all_runs.keys()):
            steps = sorted(all_runs[run_name].get(sample_task, set()))
            print(f"  {run_name}: {steps}")

if __name__ == "__main__":
    main()
