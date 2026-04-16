"""
Script to analyze trajectory files and identify stuck and milestone steps using GPT.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

try:
    # Python 3.8+
    from typing import Literal
except Exception:  # pragma: no cover
    # Python 3.7 fallback
    try:
        from typing_extensions import Literal  # type: ignore
    except Exception:  # pragma: no cover
        Literal = object  # type: ignore

try:
    # Preferred: official SDK (gives typed helpers + retries, etc.)
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI_SDK = True
except Exception:
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI_SDK = False


def _load_dotenv_if_present() -> None:
    """
    Lightweight .env loader (no external dependencies).

    Looks for a `.env` in:
    - repo root (parent of BERT_Training/)
    - BERT_Training/
    - current working directory

    For each KEY=VALUE line, sets os.environ[KEY] if not already set.
    Supports quoted values and ignores blank lines / comments (#...).
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    candidates = [
        repo_root / ".env",
        here.parent / ".env",
        Path.cwd() / ".env",
    ]

    def _strip_quotes(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            return v[1:-1]
        return v

    for env_path in candidates:
        if not env_path.exists() or not env_path.is_file():
            continue
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                val = _strip_quotes(val)
                os.environ.setdefault(key, val)
        except Exception:
            # Never fail analysis just because .env couldn't be read/parsed.
            pass


_load_dotenv_if_present()

# Azure OpenAI Configuration
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://davidrohan16278-5774-resource.cognitiveservices.azure.com/")
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-5.1-chat")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1-chat")
SUBSCRIPTION_KEY = os.environ["AZURE_OPENAI_API_KEY"]
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Base directory for trajectory files
BASE_DIR = Path(__file__).parent / "evocua_8b" / "pyautogui" / "screenshot" / "EvoCUA"

AnalysisType = Literal["stuck", "milestones"]


def initialize_client() -> Optional["AzureOpenAI"]:
    """Initialize Azure OpenAI client."""
    if not _HAS_OPENAI_SDK:
        return None
    return AzureOpenAI(  # type: ignore[misc]
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
    )


def _chat_completions_create(
    client: Optional["AzureOpenAI"],
    *,
    messages: List[Dict[str, str]],
    response_format: Dict,
    max_completion_tokens: int,
) -> str:
    """
    Create a chat completion using either:
    - the OpenAI Python SDK (if installed), or
    - direct Azure OpenAI REST call (fallback).

    Returns: assistant message content (string).
    """
    force_rest = os.getenv("AZURE_OPENAI_FORCE_REST", "").strip().lower() in {"1", "true", "yes"}
    if (not force_rest) and _HAS_OPENAI_SDK and client is not None:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
        )
        return resp.choices[0].message.content

    # Fallback: REST call (no openai package needed).
    url = (
        f"{ENDPOINT.rstrip('/')}/openai/deployments/{DEPLOYMENT}/chat/completions"
        f"?api-version={API_VERSION}"
    )
    payload = {
        "messages": messages,
        "response_format": response_format,
        "max_completion_tokens": max_completion_tokens,
    }
    bearer = os.getenv("AZURE_OPENAI_BEARER_TOKEN", "").strip()
    api_key = (SUBSCRIPTION_KEY or "").strip()
    if bearer:
        headers = {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}
    elif api_key:
        headers = {"api-key": api_key, "Content-Type": "application/json"}
    else:
        raise RuntimeError(
            "No Azure OpenAI auth configured. Set either AZURE_OPENAI_API_KEY or AZURE_OPENAI_BEARER_TOKEN."
        )

    # Prefer requests if available; otherwise use urllib.
    try:
        import requests  # type: ignore

        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except ImportError:
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # nosec - internal endpoint
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def find_trajectory_file(task_id: str) -> Optional[Path]:
    """
    Find trajectory file by task ID.
    
    Args:
        task_id: The task ID to search for
        
    Returns:
        Path to the trajectory file if found, None otherwise
    """
    if not BASE_DIR.exists():
        print(f"Base directory does not exist: {BASE_DIR}")
        return None
    
    # Search through all subdirectories
    for app_dir in BASE_DIR.iterdir():
        if app_dir.is_dir():
            task_dir = app_dir / task_id
            traj_file = task_dir / "traj.jsonl"
            if traj_file.exists():
                return traj_file
    
    return None


def load_trajectory(traj_file: Path) -> List[Dict]:
    """
    Load trajectory data from JSONL file.
    
    Args:
        traj_file: Path to the trajectory file
        
    Returns:
        List of trajectory steps
    """
    trajectory = []
    with open(traj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trajectory.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
    return trajectory


def format_trajectory_for_analysis(trajectory: List[Dict]) -> str:
    """
    Format trajectory data for GPT analysis.
    
    Args:
        trajectory: List of trajectory steps
        
    Returns:
        Formatted string representation
    """
    formatted_steps = []
    
    for step in trajectory:
        step_info = {
            "step_num": step.get("step_num"),
            "action": step.get("action"),
            "response": step.get("response", "")[:500],  # Truncate long responses
            "done": step.get("done"),
            "info": step.get("info", {})
        }
        formatted_steps.append(json.dumps(step_info, indent=2))
    
    return "\n\n".join(formatted_steps)


def analyze_trajectory_stuck(client: Optional["AzureOpenAI"], trajectory: List[Dict], task_id: str) -> Dict:
    """
    Analyze trajectory using GPT to identify stuck steps.
    """
    trajectory_text = format_trajectory_for_analysis(trajectory)

    system_prompt = """You are an expert AI agent evaluator. Your task is to analyze computer use agent trajectories and identify if the agent got stuck during execution.

An agent is considered "stuck" if:
1. It repeats the same action multiple times without progress
2. It enters an error loop or infinite loop
3. It failed to make meaningful progress for several steps

Analyze the trajectory and return a JSON response in this exact format:
{
  "is_stuck": true/false,
  "stuck_steps": [list of step numbers where agent appears stuck],
  "reasons": [list of reasons explaining why each step is stuck],
  "severity": "low/medium/high",
  "summary": "brief summary of the issue"
}

If the agent is not stuck, return:
{
  "is_stuck": false,
  "stuck_steps": [],
  "reasons": [],
  "severity": "none",
  "summary": "Agent completed task successfully"
}

"""

    user_prompt = f"""Analyze the following agent trajectory for task ID: {task_id}

Total steps: {len(trajectory)}

Trajectory:
{trajectory_text}

Provide your analysis in JSON format as specified."""

    try:
        content = _chat_completions_create(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trajectory_stuck_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_stuck": {
                                "type": "boolean",
                                "description": "Whether the agent got stuck during execution",
                            },
                            "stuck_steps": {
                                "type": "array",
                                "description": "List of step numbers where agent appears stuck",
                                "items": {"type": "integer"},
                            },
                            "reasons": {
                                "type": "array",
                                "description": "List of reasons explaining why each step is stuck",
                                "items": {"type": "string"},
                            },
                            "severity": {
                                "type": "string",
                                "description": "Severity level of the issue",
                                "enum": ["none", "low", "medium", "high"],
                            },
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of the issue or completion status",
                            },
                        },
                        "required": ["is_stuck", "stuck_steps", "reasons", "severity", "summary"],
                        "additionalProperties": False,
                    },
                },
            },
            max_completion_tokens=2000,
        )
        result = json.loads(content)
        return result

    except Exception as e:
        print(f"Error during GPT analysis: {e}")
        return {
            "error": str(e),
            "is_stuck": None,
            "stuck_steps": [],
            "reasons": [],
            "severity": "unknown",
            "summary": f"Analysis failed: {str(e)}",
        }


def analyze_trajectory_milestones(
    client: Optional["AzureOpenAI"],
    trajectory: List[Dict],
    task_id: str,
) -> Dict:
    """
    Label milestone steps in a GUI agent trajectory with reasoning.
    """
    trajectory_text = format_trajectory_for_analysis(trajectory)

    system_prompt = """You are labeling a GUI agent trajectory to identify milestone steps—steps where meaningful, verifiable progress is achieved.

1. If the trajectory is short/simple, like 5 or 6 steps, you may output only the final step as a milestone.
2. If the trajectory is long, you may list multiple milestones, but don't list too many, and each two milestones should be at least 3 steps apart.
3. If the trajectory becomes stuck (repetition/no progress), ignore steps inside the stuck region unless a milestone occurs later.

Rules:
- A milestone must be meaningful and verifiable from the given step text (action/response/done/fail).
- Prefer higher-level progress markers.
- Do NOT invent UI details you cannot support from the trajectory text.
- For EACH milestone step, provide a clear reasoning explaining why this step represents meaningful progress.
"""

    user_prompt = f"""Task ID: {task_id}
Total steps: {len(trajectory)}

Trajectory:
{trajectory_text}

Return JSON only, matching the schema. For each milestone step, provide the step number AND a reasoning explaining why it's a milestone."""

    try:
        content = _chat_completions_create(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trajectory_milestones",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "milestone_steps": {
                                "type": "array",
                                "description": "Milestone step numbers in chronological order.",
                                "items": {"type": "integer"},
                                "maxItems": 12
                            },
                            "milestone_reasons": {
                                "type": "array",
                                "description": "Reasoning for each milestone step (must have same length as milestone_steps).",
                                "items": {"type": "string"},
                                "maxItems": 12
                            },
                        },
                        "required": ["milestone_steps", "milestone_reasons"],
                        "additionalProperties": False,
                    },
                },
            },
            max_completion_tokens=2000,
        )
        result = json.loads(content)
        return result

    except Exception as e:
        print(f"Error during GPT analysis: {e}")
        return {
            "error": str(e),
            "milestone_steps": [],
            "milestone_reasons": [],
        }


def analyze_trajectory(
    client: Optional["AzureOpenAI"],
    trajectory: List[Dict],
    task_id: str,
    analysis_type: AnalysisType = "stuck",
) -> Dict:
    if analysis_type == "milestones":
        return analyze_trajectory_milestones(client, trajectory, task_id)
    return analyze_trajectory_stuck(client, trajectory, task_id)


def analyze_single_task(
    task_id: str,
    analysis_type: AnalysisType = "stuck",
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Analyze a single task trajectory.
    
    Args:
        task_id: The task ID to analyze
        verbose: Whether to print detailed output
        output_dir: Optional directory to save individual results
        
    Returns:
        Analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing task: {task_id}")
        print(f"{'='*60}")
    
    # Find trajectory file
    traj_file = find_trajectory_file(task_id)
    if not traj_file:
        error_msg = f"Trajectory file not found for task ID: {task_id}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return {
            "task_id": task_id,
            "error": error_msg,
            "file_path": None
        }
    
    if verbose:
        print(f"Found trajectory file: {traj_file}")
    
    # Load trajectory
    trajectory = load_trajectory(traj_file)
    if verbose:
        print(f"Loaded {len(trajectory)} steps")
    
    # Initialize client and analyze
    client = initialize_client()
    analysis = analyze_trajectory(
        client,
        trajectory,
        task_id,
        analysis_type=analysis_type,
    )
    
    # Add metadata
    analysis["task_id"] = task_id
    analysis["file_path"] = str(traj_file)
    analysis["total_steps"] = len(trajectory)
    analysis["analysis_type"] = analysis_type
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis Results:")
        print(f"{'='*60}")
        print(json.dumps(analysis, indent=2))
    
    # Save to individual file if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_id}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        if verbose:
            print(f"\nSaved to: {output_file}")
    
    return analysis


def analyze_all_tasks(
    analysis_type: AnalysisType = "stuck",
    output_dir: Optional[str] = None,
    summary_file: Optional[str] = None,
) -> List[Dict]:
    """
    Analyze all trajectory files in the base directory.
    
    Args:
        output_dir: Optional directory to save individual task results
        summary_file: Optional file path to save summary of all results
        
    Returns:
        List of analysis results for all tasks
    """
    if not BASE_DIR.exists():
        print(f"Base directory does not exist: {BASE_DIR}")
        return []
    
    results = []
    client = initialize_client()
    
    # Create output directory if specified
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_path}")
    
    # Find all trajectory files
    traj_files = []
    for app_dir in BASE_DIR.iterdir():
        if app_dir.is_dir():
            for task_dir in app_dir.iterdir():
                if task_dir.is_dir():
                    traj_file = task_dir / "traj.jsonl"
                    if traj_file.exists():
                        traj_files.append((task_dir.name, traj_file))
    
    print(f"Found {len(traj_files)} trajectory files to analyze\n")
    
    # Analyze each file
    for idx, (task_id, traj_file) in enumerate(traj_files, 1):
        print(f"[{idx}/{len(traj_files)}] Analyzing task: {task_id}")
        
        try:
            trajectory = load_trajectory(traj_file)
            analysis = analyze_trajectory(
                client,
                trajectory,
                task_id,
                analysis_type=analysis_type,
            )
            
            analysis["task_id"] = task_id
            analysis["file_path"] = str(traj_file)
            analysis["total_steps"] = len(trajectory)
            analysis["analysis_type"] = analysis_type
            
            results.append(analysis)
            
            # Save individual result file
            if output_path:
                task_output_file = output_path / f"{task_id}.json"
                with open(task_output_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
            
            # Print summary
            if analysis_type == "milestones":
                mcount = len(analysis.get("milestone_steps", []) or [])
                print(f"   ✓ OK - {mcount} milestone(s)")
                # Display milestone details if available
                m_steps = analysis.get("milestone_steps", [])
                m_reasons = analysis.get("milestone_reasons", [])
                if m_steps and m_reasons and len(m_steps) == len(m_reasons):
                    for step, reason in zip(m_steps, m_reasons):
                        print(f"      Step {step}: {reason}")
            else:
                status = "⚠️  STUCK" if analysis.get("is_stuck") else "✓ OK"
                print(f"   {status} - {analysis.get('summary', 'No summary')}")
            
        except Exception as e:
            print(f"   ERROR: Failed to analyze - {e}")
            error_result = {
                "task_id": task_id,
                "file_path": str(traj_file),
                "error": str(e)
            }
            results.append(error_result)
            
            # Save error result file
            if output_path:
                task_output_file = output_path / f"{task_id}.json"
                with open(task_output_file, 'w') as f:
                    json.dump(error_result, f, indent=2)
    
    # Save summary file if specified
    if summary_file:
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n\nSummary saved to: {summary_file}")
    
    # Print summary statistics
    stuck_count = sum(1 for r in results if r.get("is_stuck"))
    print(f"\n{'='*60}")
    print(f"Summary Statistics:")
    print(f"{'='*60}")
    print(f"Total tasks analyzed: {len(results)}")
    if analysis_type == "milestones":
        with_milestones = sum(
            1
            for r in results
            if (r.get("milestone_steps") is not None) and len(r.get("milestone_steps") or []) > 0
        )
        print(f"Trajectories with >=1 milestone: {with_milestones}")
        print(f"Trajectories with 0 milestones: {len(results) - with_milestones}")
    else:
        print(f"Tasks with issues: {stuck_count}")
        print(f"Tasks OK: {len(results) - stuck_count}")
    if output_path:
        print(f"\nIndividual results saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze agent trajectories (stuck-steps or milestone labeling)"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Analyze a single task by its task ID"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all trajectory files in the base directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save individual task results. "
            "Default depends on --analysis-type (stuck: analysis_results, milestones: milestone_results)."
        ),
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="File path to save summary of all results (optional, for --all mode)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        help="Override base directory for trajectory files"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files, only print to console"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        default="stuck",
        choices=["stuck", "milestones"],
        help="Type of analysis to run (default: stuck)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of times to run the analysis (results saved to separate folders: run1, run2, etc.)",
    )
    
    args = parser.parse_args()
    
    # Override base directory if specified
    if args.base_dir:
        global BASE_DIR
        BASE_DIR = Path(args.base_dir)
    
    # Determine output directory
    default_output_dir = "analysis_results" if args.analysis_type == "stuck" else "milestone_results"
    output_dir = None if args.no_save else (args.output_dir or default_output_dir)
    
    # Handle multiple runs
    if args.num_runs > 1:
        print(f"\n{'='*60}")
        print(f"Running analysis {args.num_runs} times")
        print(f"{'='*60}\n")
        
        all_runs_results = []
        
        for run_num in range(1, args.num_runs + 1):
            print(f"\n{'#'*60}")
            print(f"# RUN {run_num}/{args.num_runs}")
            print(f"{'#'*60}\n")
            
            # Create run-specific output directory
            if output_dir:
                run_output_dir = f"{output_dir}_run{run_num}"
            else:
                run_output_dir = None
            
            # Create run-specific summary file
            run_summary = None
            if args.summary:
                summary_path = Path(args.summary)
                run_summary = str(summary_path.parent / f"{summary_path.stem}_run{run_num}{summary_path.suffix}")
            
            if args.task_id:
                # Analyze single task
                output_path = Path(run_output_dir) if run_output_dir else None
                result = analyze_single_task(
                    args.task_id,
                    analysis_type=args.analysis_type,
                    verbose=True,
                    output_dir=output_path,
                )
                all_runs_results.append({
                    "run": run_num,
                    "result": result
                })
            
            elif args.all:
                # Analyze all tasks
                results = analyze_all_tasks(
                    analysis_type=args.analysis_type,
                    output_dir=run_output_dir,
                    summary_file=run_summary,
                )
                all_runs_results.append({
                    "run": run_num,
                    "results": results
                })
        
        # Save combined summary of all runs
        if args.summary and not args.no_save:
            summary_path = Path(args.summary)
            combined_summary = summary_path.parent / f"{summary_path.stem}_all_runs{summary_path.suffix}"
            with open(combined_summary, 'w') as f:
                json.dump(all_runs_results, f, indent=2)
            print(f"\n\n{'='*60}")
            print(f"Combined summary of all {args.num_runs} runs saved to: {combined_summary}")
            print(f"{'='*60}")
    
    else:
        # Single run (original behavior)
        if args.task_id:
            # Analyze single task
            output_path = Path(output_dir) if output_dir else None
            result = analyze_single_task(
                args.task_id,
                analysis_type=args.analysis_type,
                verbose=True,
                output_dir=output_path,
            )
        
        elif args.all:
            # Analyze all tasks
            results = analyze_all_tasks(
                analysis_type=args.analysis_type,
                output_dir=output_dir,
                summary_file=args.summary,
            )
        
        else:
            parser.print_help()
            print("\n\nExample usage:")
            print("  # Analyze a single task and save to folder")
            print("  python analyze_trajectories.py --task-id 2ad9387a-65d8-4e33-ad5b-7580065a27ca")
            print("\n  # Milestone labeling for a single task (new prompt)")
            print("  python analyze_trajectories.py --analysis-type milestones --task-id 2ad9387a-65d8-4e33-ad5b-7580065a27ca")
            print("\n  # Analyze all tasks and save each to separate files")
            print("  python analyze_trajectories.py --all")
            print("\n  # Milestone labeling for all tasks")
            print("  python analyze_trajectories.py --analysis-type milestones --all")
            print("\n  # Run milestone analysis 5 times and save to separate folders")
            print("  python analyze_trajectories.py --analysis-type milestones --all --num-runs 5")
            print("\n  # Analyze all tasks with custom output directory and summary")
            print("  python analyze_trajectories.py --all --output-dir my_results --summary summary.json")
            print("\n  # Analyze without saving files (console only)")
            print("  python analyze_trajectories.py --task-id TASK_ID --no-save")


if __name__ == "__main__":
    main()
