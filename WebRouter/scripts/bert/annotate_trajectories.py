#!/usr/bin/env python3
"""Analyze WebArena-verified agent trajectories to label stuck steps and milestone steps.

Adapted from 3rd_party/bert/analyze_trajectories.py for the WebArena experiment
format where trajectories are stored as separate step_N.json files rather than a
single traj.jsonl.

Usage examples:
    # Analyze a single task for stuck steps
    python annotate_trajectories.py --task-name webarena_verified.279.0.2 --analysis-type stuck

    # Analyze all tasks for milestones
    python annotate_trajectories.py --all --analysis-type milestones

    # Run milestone analysis 3 times for consensus
    python annotate_trajectories.py --all --analysis-type milestones --num-runs 3

    # Limit to first 10 tasks for testing
    python annotate_trajectories.py --all --analysis-type stuck --max-tasks 10
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from typing import Literal
except Exception:
    try:
        from typing_extensions import Literal  # type: ignore
    except Exception:
        Literal = object  # type: ignore

try:
    from openai import AzureOpenAI  # type: ignore

    _HAS_OPENAI_SDK = True
except Exception:
    AzureOpenAI = None  # type: ignore
    _HAS_OPENAI_SDK = False

# Import shared config
sys.path.insert(0, str(Path(__file__).parent))
from config import *  # noqa: F401, F403, E402


# ── Azure OpenAI Configuration ───────────────────────────────────────────────


def _load_dotenv_if_present() -> None:
    """Lightweight .env loader (no external dependencies).

    Looks for a .env in the repo root, scripts/bert/, or cwd.
    For each KEY=VALUE line, sets os.environ[KEY] if not already set.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
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
            pass


_load_dotenv_if_present()

ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "",
)
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4.1")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
SUBSCRIPTION_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "",
)
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Default results directory
DEFAULT_RESULTS_DIR = OSS20B_RESULTS_DIR

AnalysisType = Literal["stuck", "milestones"]


# ── Azure OpenAI client ──────────────────────────────────────────────────────


def initialize_client() -> Optional["AzureOpenAI"]:
    """Initialize Azure OpenAI client."""
    if not _HAS_OPENAI_SDK:
        return None
    return AzureOpenAI(
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
    """Create a chat completion using the OpenAI SDK or a REST fallback.

    Returns:
        Assistant message content (string).
    """
    force_rest = os.getenv("AZURE_OPENAI_FORCE_REST", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
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
            "No Azure OpenAI auth configured. "
            "Set either AZURE_OPENAI_API_KEY or AZURE_OPENAI_BEARER_TOKEN."
        )

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
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


# ── WebArena trajectory loading ──────────────────────────────────────────────


def find_experiment_dirs(results_dir: Path) -> List[Path]:
    """Return all experiment sub-directories that contain step JSON files.

    Args:
        results_dir: Top-level study results directory.

    Returns:
        Sorted list of experiment directory paths.
    """
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return []

    exp_dirs = []
    for child in results_dir.iterdir():
        if child.is_dir() and (child / "step_0.json").exists():
            exp_dirs.append(child)
    exp_dirs.sort(key=lambda p: p.name)
    return exp_dirs


def load_trajectory(exp_dir: Path) -> List[Dict]:
    """Load trajectory steps from an experiment directory.

    Reads step_0.json, step_1.json, ... in order and skips the terminal step
    (where action is null and think is null).

    Args:
        exp_dir: Path to the experiment directory containing step_N.json files.

    Returns:
        List of step dicts with keys: step_num, action, think, goal.
    """
    steps = []
    idx = 0
    while True:
        step_file = exp_dir / f"step_{idx}.json"
        if not step_file.exists():
            break
        try:
            with open(step_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: failed to read {step_file}: {e}")
            idx += 1
            continue

        action = data.get("action")
        think = data.get("think")

        # Skip terminal step (action=null, think=null)
        if action is None and think is None:
            idx += 1
            continue

        step_entry = {
            "step_num": data.get("step", idx),
            "action": action or "",
            "think": think or "",
            "goal": data.get("goal", "") if idx == 0 else "",
        }
        steps.append(step_entry)
        idx += 1

    return steps


# ── Trajectory formatting ────────────────────────────────────────────────────


def format_trajectory_for_analysis(trajectory: List[Dict]) -> str:
    """Format trajectory steps into a compact text representation for GPT.

    Each step is formatted as:
        Step N: [action] think_text_truncated_to_500_chars

    Args:
        trajectory: List of step dicts from load_trajectory().

    Returns:
        Multi-line formatted string.
    """
    lines = []
    for step in trajectory:
        action = step.get("action", "")
        think = step.get("think", "")
        # Truncate think text to 500 characters
        if len(think) > 500:
            think = think[:500] + "..."
        lines.append(f"Step {step['step_num']}: [{action}] {think}")
    return "\n".join(lines)


# ── GPT analysis functions ───────────────────────────────────────────────────


def analyze_trajectory_stuck(
    client: Optional["AzureOpenAI"],
    trajectory: List[Dict],
    task_name: str,
) -> Dict:
    """Analyze a trajectory for stuck steps using GPT.

    Args:
        client: Azure OpenAI client instance.
        trajectory: List of step dicts.
        task_name: WebArena task name for context.

    Returns:
        Dict with stuck analysis results.
    """
    trajectory_text = format_trajectory_for_analysis(trajectory)
    goal = ""
    for step in trajectory:
        if step.get("goal"):
            goal = step["goal"]
            break

    system_prompt = (
        "You are an expert AI agent evaluator. Your task is to analyze computer use agent "
        "trajectories and identify if the agent got stuck during execution.\n\n"
        'An agent is considered "stuck" if:\n'
        "1. It repeats the same action multiple times without progress\n"
        "2. It enters an error loop or infinite loop\n"
        "3. It failed to make meaningful progress for several steps\n\n"
        "Analyze the trajectory and return a JSON response in this exact format:\n"
        "{\n"
        '  "is_stuck": true/false,\n'
        '  "stuck_steps": [list of step numbers where agent appears stuck],\n'
        '  "reasons": [list of reasons explaining why each step is stuck],\n'
        '  "severity": "low/medium/high",\n'
        '  "summary": "brief summary of the issue"\n'
        "}\n\n"
        "If the agent is not stuck, return:\n"
        "{\n"
        '  "is_stuck": false,\n'
        '  "stuck_steps": [],\n'
        '  "reasons": [],\n'
        '  "severity": "none",\n'
        '  "summary": "Agent completed task successfully"\n'
        "}\n"
    )

    goal_line = f"\nGoal: {goal[:500]}\n" if goal else ""
    user_prompt = (
        f"Analyze the following agent trajectory for task: {task_name}\n"
        f"{goal_line}"
        f"\nTotal steps: {len(trajectory)}\n\n"
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Provide your analysis in JSON format as specified."
    )

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
                                "description": ("List of step numbers where agent appears stuck"),
                                "items": {"type": "integer"},
                            },
                            "reasons": {
                                "type": "array",
                                "description": (
                                    "List of reasons explaining why each step is stuck"
                                ),
                                "items": {"type": "string"},
                            },
                            "severity": {
                                "type": "string",
                                "description": "Severity level of the issue",
                                "enum": ["none", "low", "medium", "high"],
                            },
                            "summary": {
                                "type": "string",
                                "description": ("Brief summary of the issue or completion status"),
                            },
                        },
                        "required": [
                            "is_stuck",
                            "stuck_steps",
                            "reasons",
                            "severity",
                            "summary",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            max_completion_tokens=2000,
        )
        return json.loads(content)

    except Exception as e:
        print(f"  Error during GPT analysis: {e}")
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
    task_name: str,
) -> Dict:
    """Label milestone steps in a trajectory with reasoning.

    Args:
        client: Azure OpenAI client instance.
        trajectory: List of step dicts.
        task_name: WebArena task name for context.

    Returns:
        Dict with milestone analysis results.
    """
    trajectory_text = format_trajectory_for_analysis(trajectory)
    goal = ""
    for step in trajectory:
        if step.get("goal"):
            goal = step["goal"]
            break

    system_prompt = (
        "You are labeling a GUI agent trajectory to identify milestone steps -- steps where "
        "meaningful, verifiable progress is achieved.\n\n"
        "1. If the trajectory is short/simple, like 5 or 6 steps, you may output only the "
        "final step as a milestone.\n"
        "2. If the trajectory is long, you may list multiple milestones, but don't list too "
        "many, and each two milestones should be at least 3 steps apart.\n"
        "3. If the trajectory becomes stuck (repetition/no progress), ignore steps inside the "
        "stuck region unless a milestone occurs later.\n\n"
        "Rules:\n"
        "- A milestone must be meaningful and verifiable from the given step text "
        "(action/response/done/fail).\n"
        "- Prefer higher-level progress markers.\n"
        "- Do NOT invent UI details you cannot support from the trajectory text.\n"
        "- For EACH milestone step, provide a clear reasoning explaining why this step "
        "represents meaningful progress.\n"
    )

    goal_line = f"\nGoal: {goal[:500]}\n" if goal else ""
    user_prompt = (
        f"Task: {task_name}\n"
        f"{goal_line}"
        f"\nTotal steps: {len(trajectory)}\n\n"
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Return JSON only, matching the schema. For each milestone step, provide the step "
        f"number AND a reasoning explaining why it's a milestone."
    )

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
                                "description": ("Milestone step numbers in chronological order."),
                                "items": {"type": "integer"},
                                "maxItems": 12,
                            },
                            "milestone_reasons": {
                                "type": "array",
                                "description": (
                                    "Reasoning for each milestone step "
                                    "(must have same length as milestone_steps)."
                                ),
                                "items": {"type": "string"},
                                "maxItems": 12,
                            },
                        },
                        "required": ["milestone_steps", "milestone_reasons"],
                        "additionalProperties": False,
                    },
                },
            },
            max_completion_tokens=2000,
        )
        return json.loads(content)

    except Exception as e:
        print(f"  Error during GPT analysis: {e}")
        return {
            "error": str(e),
            "milestone_steps": [],
            "milestone_reasons": [],
        }


def analyze_trajectory(
    client: Optional["AzureOpenAI"],
    trajectory: List[Dict],
    task_name: str,
    analysis_type: AnalysisType = "stuck",
) -> Dict:
    """Dispatch to the appropriate analysis function.

    Args:
        client: Azure OpenAI client instance.
        trajectory: List of step dicts.
        task_name: WebArena task name.
        analysis_type: Either "stuck" or "milestones".

    Returns:
        Analysis result dict.
    """
    if analysis_type == "milestones":
        return analyze_trajectory_milestones(client, trajectory, task_name)
    return analyze_trajectory_stuck(client, trajectory, task_name)


# ── High-level analysis entry points ─────────────────────────────────────────


def analyze_single_task(
    task_name: str,
    results_dir: Path,
    analysis_type: AnalysisType = "stuck",
    output_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> Optional[Dict]:
    """Analyze trajectories for a single task.

    Finds the experiment directory matching *task_name* and runs the requested
    analysis.

    Args:
        task_name: WebArena task name (e.g. webarena_verified.279.0.2).
        results_dir: Study results directory.
        analysis_type: "stuck" or "milestones".
        output_dir: Where to save the output JSON.
        skip_existing: If True, skip tasks that already have output files.

    Returns:
        Analysis result dict, or None if not found / skipped.
    """
    # Check skip_existing
    if skip_existing and output_dir:
        output_file = output_dir / f"{task_name}.json"
        if output_file.exists():
            print(f"  Skipping {task_name} (already exists)")
            return None

    # Find matching experiment directory
    exp_dirs = find_experiment_dirs(results_dir)
    matching = []
    for exp_dir in exp_dirs:
        extracted = get_task_name_from_dir(exp_dir.name)
        if extracted == task_name:
            matching.append(exp_dir)

    if not matching:
        print(f"  ERROR: No experiment directory found for task: {task_name}")
        return {
            "task_name": task_name,
            "error": f"No experiment directory found for task: {task_name}",
            "exp_dir": None,
        }

    # Use the first matching directory
    exp_dir = matching[0]
    if len(matching) > 1:
        print(f"  Warning: {len(matching)} dirs match {task_name}, using first: {exp_dir.name}")

    # Load trajectory
    trajectory = load_trajectory(exp_dir)
    if not trajectory:
        print(f"  ERROR: No valid steps found in {exp_dir}")
        return {
            "task_name": task_name,
            "exp_dir": str(exp_dir),
            "error": "No valid trajectory steps found",
            "total_steps": 0,
        }

    # Run analysis
    client = initialize_client()
    analysis = analyze_trajectory(client, trajectory, task_name, analysis_type=analysis_type)

    # Add metadata
    analysis["task_name"] = task_name
    analysis["exp_dir"] = str(exp_dir)
    analysis["total_steps"] = len(trajectory)
    analysis["analysis_type"] = analysis_type

    # Save output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_name}.json"
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)

    return analysis


def analyze_all_tasks(
    results_dir: Path,
    analysis_type: AnalysisType = "stuck",
    output_dir: Optional[Path] = None,
    max_tasks: Optional[int] = None,
    skip_existing: bool = False,
    allowed_task_names: Optional[set[str]] = None,
) -> List[Dict]:
    """Analyze all experiment trajectories in a results directory.

    Args:
        results_dir: Study results directory containing experiment sub-dirs.
        analysis_type: "stuck" or "milestones".
        output_dir: Where to save per-task output JSONs.
        max_tasks: Limit the number of tasks to analyze (for testing).
        skip_existing: If True, skip tasks whose output JSON already exists.
        allowed_task_names: Optional allowlist of task names to analyze.

    Returns:
        List of analysis result dicts.
    """
    exp_dirs = find_experiment_dirs(results_dir)
    if not exp_dirs:
        print(f"No experiment directories found in: {results_dir}")
        return []

    # Build (task_name, exp_dir) pairs, deduplicating by task_name (first match wins)
    seen: Dict[str, Path] = {}
    for exp_dir in exp_dirs:
        task_name = get_task_name_from_dir(exp_dir.name)
        if allowed_task_names is not None and task_name not in allowed_task_names:
            continue
        if task_name and task_name not in seen:
            seen[task_name] = exp_dir

    tasks = sorted(seen.items())
    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    print(f"Found {len(tasks)} unique tasks to analyze (from {len(exp_dirs)} experiment dirs)")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    print()

    client = initialize_client()
    results = []
    success_count = 0
    error_count = 0
    skip_count = 0

    for idx, (task_name, exp_dir) in enumerate(tasks, 1):
        print(f"[{idx}/{len(tasks)}] {task_name}")

        # Check skip_existing
        if skip_existing and output_dir:
            output_file = output_dir / f"{task_name}.json"
            if output_file.exists():
                print(f"  Skipped (already exists)")
                skip_count += 1
                continue

        try:
            trajectory = load_trajectory(exp_dir)
            if not trajectory:
                print(f"  ERROR: No valid steps")
                error_count += 1
                error_result = {
                    "task_name": task_name,
                    "exp_dir": str(exp_dir),
                    "error": "No valid trajectory steps found",
                    "total_steps": 0,
                    "analysis_type": analysis_type,
                }
                results.append(error_result)
                if output_dir:
                    with open(output_dir / f"{task_name}.json", "w") as f:
                        json.dump(error_result, f, indent=2)
                continue

            analysis = analyze_trajectory(
                client, trajectory, task_name, analysis_type=analysis_type
            )

            # Add metadata
            analysis["task_name"] = task_name
            analysis["exp_dir"] = str(exp_dir)
            analysis["total_steps"] = len(trajectory)
            analysis["analysis_type"] = analysis_type

            results.append(analysis)
            success_count += 1

            # Save individual result
            if output_dir:
                with open(output_dir / f"{task_name}.json", "w") as f:
                    json.dump(analysis, f, indent=2)

            # Print summary line
            if analysis_type == "milestones":
                mcount = len(analysis.get("milestone_steps") or [])
                print(f"  OK - {mcount} milestone(s)")
                m_steps = analysis.get("milestone_steps", [])
                m_reasons = analysis.get("milestone_reasons", [])
                if m_steps and m_reasons and len(m_steps) == len(m_reasons):
                    for step, reason in zip(m_steps, m_reasons):
                        print(f"    Step {step}: {reason}")
            else:
                is_stuck = analysis.get("is_stuck")
                status = "STUCK" if is_stuck else "OK"
                print(f"  {status} - {analysis.get('summary', 'No summary')}")

        except Exception as e:
            print(f"  ERROR: {e}")
            error_count += 1
            error_result = {
                "task_name": task_name,
                "exp_dir": str(exp_dir),
                "error": str(e),
                "analysis_type": analysis_type,
            }
            results.append(error_result)
            if output_dir:
                with open(output_dir / f"{task_name}.json", "w") as f:
                    json.dump(error_result, f, indent=2)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Summary Statistics:")
    print(f"{'=' * 60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Analyzed successfully: {success_count}")
    print(f"Errors: {error_count}")
    if skip_existing:
        print(f"Skipped (already existed): {skip_count}")

    if analysis_type == "milestones":
        with_milestones = sum(1 for r in results if len(r.get("milestone_steps") or []) > 0)
        print(f"Trajectories with >=1 milestone: {with_milestones}")
        print(f"Trajectories with 0 milestones: {len(results) - with_milestones}")
    else:
        stuck_count = sum(1 for r in results if r.get("is_stuck"))
        print(f"Tasks stuck: {stuck_count}")
        print(f"Tasks OK: {success_count - stuck_count}")

    if output_dir:
        print(f"\nResults saved to: {output_dir}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze WebArena-verified agent trajectories for stuck steps "
            "or milestone labeling using Azure OpenAI GPT."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=(
            "Path to the study results directory containing experiment sub-dirs. "
            f"Default: {DEFAULT_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save per-task analysis JSONs. "
            "Default: results/data/<run-name>/stuck_analysis/ or "
            "results/data/<run-name>/milestone_analysis/"
        ),
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        default="stuck",
        choices=["stuck", "milestones"],
        help="Type of analysis to run (default: stuck)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Analyze a single task by its task name (e.g. webarena_verified.279.0.2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all tasks in the results directory",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help=(
            "Number of times to run the analysis (for consensus). "
            "Results saved to output_dir_run1, output_dir_run2, etc. (default: 1)"
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks to analyze (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have an output JSON file",
    )
    parser.add_argument(
        "--task-names-file",
        type=str,
        default=None,
        help=(
            "Optional text file with one task_name per line. "
            "Only those tasks are analyzed."
        ),
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    allowed_task_names = None
    if args.task_names_file:
        task_names_path = Path(args.task_names_file)
        if not task_names_path.exists():
            raise FileNotFoundError(f"Task names file not found: {task_names_path}")
        allowed_task_names = {
            line.strip()
            for line in task_names_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        print(
            f"Loaded {len(allowed_task_names)} task names from {task_names_path}"
        )

    # Determine default output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    elif args.analysis_type == "milestones":
        base_output_dir = get_milestone_analysis_dir(results_dir)
    else:
        base_output_dir = get_stuck_analysis_dir(results_dir)

    if not args.task_name and not args.all:
        parser.print_help()
        print("\n\nExample usage:")
        print("  # Stuck analysis for a single task")
        print(
            "  python annotate_trajectories.py "
            "--task-name webarena_verified.279.0.2 --analysis-type stuck"
        )
        print("\n  # Milestone labeling for all tasks")
        print("  python annotate_trajectories.py --all --analysis-type milestones")
        print("\n  # Run milestone analysis 3 times for consensus")
        print(
            "  python annotate_trajectories.py "
            "--all --analysis-type milestones --num-runs 3"
        )
        print("\n  # Test with first 10 tasks, skip already-analyzed")
        print(
            "  python annotate_trajectories.py "
            "--all --analysis-type stuck --max-tasks 10 --skip-existing"
        )
        return

    # Handle multiple runs
    if args.num_runs > 1:
        print(f"\n{'=' * 60}")
        print(f"Running analysis {args.num_runs} times")
        print(f"{'=' * 60}\n")

        for run_num in range(1, args.num_runs + 1):
            print(f"\n{'#' * 60}")
            print(f"# RUN {run_num}/{args.num_runs}")
            print(f"{'#' * 60}\n")

            run_output_dir = Path(f"{base_output_dir}_run{run_num}")

            if args.task_name:
                result = analyze_single_task(
                    args.task_name,
                    results_dir,
                    analysis_type=args.analysis_type,
                    output_dir=run_output_dir,
                    skip_existing=args.skip_existing,
                )
                if result:
                    print(json.dumps(result, indent=2))
            elif args.all:
                analyze_all_tasks(
                    results_dir,
                    analysis_type=args.analysis_type,
                    output_dir=run_output_dir,
                    max_tasks=args.max_tasks,
                    skip_existing=args.skip_existing,
                    allowed_task_names=allowed_task_names,
                )

    else:
        # Single run
        if args.task_name:
            result = analyze_single_task(
                args.task_name,
                results_dir,
                analysis_type=args.analysis_type,
                output_dir=base_output_dir,
                skip_existing=args.skip_existing,
            )
            if result:
                print(f"\n{'=' * 60}")
                print("Analysis Result:")
                print(f"{'=' * 60}")
                print(json.dumps(result, indent=2))
        elif args.all:
            analyze_all_tasks(
                results_dir,
                analysis_type=args.analysis_type,
                output_dir=base_output_dir,
                max_tasks=args.max_tasks,
                skip_existing=args.skip_existing,
                allowed_task_names=allowed_task_names,
            )


if __name__ == "__main__":
    main()
