import json
import os
from collections import defaultdict
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
                total_steps = content.get('total_steps', 0)
                if task_id:
                    data[task_id] = {
                        'milestone_steps': set(milestone_steps),
                        'total_steps': total_steps
                    }
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return data

def main():
    base_dir = "/gpfs/radev/home/jw3278/project/GUI_router/BERT_Training"
    
    # Load data from runs 1, 2, and 6 only
    print("Loading data from runs 1, 2, and 6...")
    all_runs = {}
    for i in [1, 2, 6]:
        run_name = f"my_results_run{i}"
        run_dir = os.path.join(base_dir, run_name)
        all_runs[run_name] = load_run_data(run_dir)
        print(f"  {run_name}: {len(all_runs[run_name])} tasks loaded")
    
    # Get all task_ids
    all_task_ids = set()
    for run_data in all_runs.values():
        all_task_ids.update(run_data.keys())
    
    print(f"\nTotal unique task_ids: {len(all_task_ids)}")
    
    # Prepare the output structure
    output = {
        "summary": {
            "total_tasks": len(all_task_ids),
            "runs_analyzed": ["my_results_run1", "my_results_run2", "my_results_run6"]
        },
        "steps_in_3_runs": {},
        "steps_in_2_runs": {},
        "steps_in_1_run": {},
        "steps_in_0_runs": {}
    }
    
    print("\nAnalyzing step overlap...")
    
    # For each task, categorize steps
    for task_id in sorted(all_task_ids):
        # Collect milestone steps from each run for this task
        task_data = {}
        max_total_steps = 0
        
        for run_name in ["my_results_run1", "my_results_run2", "my_results_run6"]:
            if task_id in all_runs[run_name]:
                task_data[run_name] = all_runs[run_name][task_id]['milestone_steps']
                max_total_steps = max(max_total_steps, all_runs[run_name][task_id]['total_steps'])
            else:
                task_data[run_name] = set()
        
        # Get all milestone steps for this task
        all_milestone_steps = set()
        for steps in task_data.values():
            all_milestone_steps.update(steps)
        
        # Categorize each milestone step by how many runs it appears in
        steps_3_runs = []
        steps_2_runs = []
        steps_1_run = []
        
        for step in sorted(all_milestone_steps):
            count = sum(1 for run_steps in task_data.values() if step in run_steps)
            if count == 3:
                steps_3_runs.append(step)
            elif count == 2:
                steps_2_runs.append(step)
            elif count == 1:
                steps_1_run.append(step)
        
        # Find steps not identified by any run (steps 1 to total_steps that are not milestones)
        all_steps = set(range(1, max_total_steps + 1)) if max_total_steps > 0 else set()
        steps_0_runs = sorted(all_steps - all_milestone_steps)
        
        # Add to output
        task_info = {
            "total_steps": max_total_steps,
            "milestone_steps_per_run": {
                run_name: sorted(list(task_data[run_name])) 
                for run_name in ["my_results_run1", "my_results_run2", "my_results_run6"]
            }
        }
        
        if steps_3_runs:
            output["steps_in_3_runs"][task_id] = {
                **task_info,
                "steps": steps_3_runs
            }
        
        if steps_2_runs:
            output["steps_in_2_runs"][task_id] = {
                **task_info,
                "steps": steps_2_runs
            }
        
        if steps_1_run:
            output["steps_in_1_run"][task_id] = {
                **task_info,
                "steps": steps_1_run
            }
        
        if steps_0_runs:
            output["steps_in_0_runs"][task_id] = {
                **task_info,
                "steps": steps_0_runs
            }
    
    # Add statistics
    total_steps_3 = sum(len(v["steps"]) for v in output["steps_in_3_runs"].values())
    total_steps_2 = sum(len(v["steps"]) for v in output["steps_in_2_runs"].values())
    total_steps_1 = sum(len(v["steps"]) for v in output["steps_in_1_run"].values())
    total_steps_0 = sum(len(v["steps"]) for v in output["steps_in_0_runs"].values())
    total_milestone_steps = total_steps_3 + total_steps_2 + total_steps_1
    
    output["summary"]["statistics"] = {
        "steps_in_3_runs": {
            "count": total_steps_3,
            "percentage": round(total_steps_3 / total_milestone_steps * 100, 1) if total_milestone_steps > 0 else 0,
            "num_tasks": len(output["steps_in_3_runs"])
        },
        "steps_in_2_runs": {
            "count": total_steps_2,
            "percentage": round(total_steps_2 / total_milestone_steps * 100, 1) if total_milestone_steps > 0 else 0,
            "num_tasks": len(output["steps_in_2_runs"])
        },
        "steps_in_1_run": {
            "count": total_steps_1,
            "percentage": round(total_steps_1 / total_milestone_steps * 100, 1) if total_milestone_steps > 0 else 0,
            "num_tasks": len(output["steps_in_1_run"])
        },
        "steps_in_0_runs": {
            "count": total_steps_0,
            "num_tasks": len(output["steps_in_0_runs"])
        },
        "total_milestone_steps": total_milestone_steps,
        "total_all_steps": total_milestone_steps + total_steps_0
    }
    
    # Write to JSON file
    output_file = os.path.join(base_dir, "step_overlap_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  Steps in 3 runs: {total_steps_3} ({output['summary']['statistics']['steps_in_3_runs']['percentage']}%)")
    print(f"  Steps in 2 runs: {total_steps_2} ({output['summary']['statistics']['steps_in_2_runs']['percentage']}%)")
    print(f"  Steps in 1 run:  {total_steps_1} ({output['summary']['statistics']['steps_in_1_run']['percentage']}%)")
    print(f"  Steps in 0 runs: {total_steps_0}")
    print(f"  Total milestone steps: {total_milestone_steps}")
    print(f"  Total all steps: {total_milestone_steps + total_steps_0}")

if __name__ == "__main__":
    main()
