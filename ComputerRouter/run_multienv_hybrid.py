"""
Hybrid Agent Runner: EvoCUA + Claude with BERT Stuck Detection

This script runs a hybrid agent that:
1. Starts execution with EvoCUA model
2. After each step, checks if the agent is stuck using a ModernBERT classifier
3. If stuck is detected, switches to Claude to continue execution

Environment Variables Required:
    export AWS_ACCESS_KEY_ID="xx"
    export AWS_SECRET_ACCESS_KEY="xx"
    export AWS_REGION="xx"
    export AWS_SECURITY_GROUP_ID="xx"
    export AWS_SUBNET_ID="xx"
    export OPENAI_API_KEY="xxxx"        # For EvoCUA
    export OPENAI_BASE_URL="xxxx"       # For EvoCUA
    export ANTHROPIC_API_KEY="xxxx"     # For Claude

Example Usage:
    python3 run_multienv_hybrid.py \
        --headless \
        --provider_name aws \
        --observation_type screenshot \
        --evocua_model EvoCUA-S2 \
        --claude_model claude-sonnet-4-5-20250929 \
        --result_dir ./hybrid_results \
        --test_all_meta_path evaluation_examples/test_nogdrive.json \
        --max_steps 50 \
        --num_envs 10 \
        --stuck_detector_path /gpfs/radev/scratch/cohan/jw3278/modernbert-stuck-detector \
        --stuck_threshold 0.5 \
        --min_steps_to_check 3
"""

from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import sys
import signal
import time
from typing import List, Dict, Optional, Set
from multiprocessing import Process, Manager, Queue
from multiprocessing import current_process
import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.evocua.evocua_agent import EvoCUAAgent
from mm_agents.anthropic import AnthropicAgent
from mm_agents.stuck_detector import create_stuck_detector
from mm_agents.milestone_detector import create_milestone_detector, MilestoneJudge

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False


# Thread-local storage for task context (works per-process in multiprocessing)
import threading
_task_context = threading.local()

def get_task_context():
    """Get current task context from thread-local storage."""
    return getattr(_task_context, 'context', {'domain': None, 'example_id': None})

def set_task_context(domain: str, example_id: str):
    """Set current task context in thread-local storage."""
    _task_context.context = {'domain': domain, 'example_id': example_id}

def clear_task_context():
    """Clear current task context."""
    if hasattr(_task_context, 'context'):
        delattr(_task_context, 'context')

class TaskContextFilter(logging.Filter):
    """Filter to add domain and example_id to log records."""
    def filter(self, record):
        ctx = get_task_context()
        domain = ctx.get('domain')
        example_id = ctx.get('example_id')
        if domain and example_id:
            record.domain = domain
            record.example_id = example_id
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                if not record.msg.startswith(f"[{domain}/{example_id}]"):
                    record.msg = f"[{domain}/{example_id}] {record.msg}"
        else:
            record.domain = domain or "N/A"
            record.example_id = example_id or "N/A"
        return True

# Load environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid agent evaluation with EvoCUA + Claude (stuck detection)"
    )

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=50)
    
    # Evaluation config
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")

    # EvoCUA model config
    parser.add_argument("--evocua_model", type=str, default="EvoCUA-S2", help="EvoCUA model name")
    parser.add_argument("--evocua_temperature", type=float, default=0.01)
    parser.add_argument("--evocua_top_p", type=float, default=0.9)
    parser.add_argument("--evocua_max_tokens", type=int, default=32768)
    parser.add_argument("--prompt_style", type=str, default="S2", choices=["S1", "S2"])
    parser.add_argument("--max_history_turns", type=int, default=4)
    parser.add_argument("--coordinate_type", type=str, default="relative")
    parser.add_argument("--resize_factor", type=int, default=32)
    parser.add_argument("--password", type=str, default="osworld-public-evaluation")

    # Claude model config
    parser.add_argument("--claude_model", type=str, default="claude-sonnet-4-5-20250929", help="Claude model name")
    parser.add_argument("--claude_temperature", type=float, default=None)
    parser.add_argument("--claude_top_p", type=float, default=None)
    parser.add_argument("--claude_max_tokens", type=int, default=4096)
    parser.add_argument("--claude_max_trajectory_length", type=int, default=10)
    parser.add_argument("--no-thinking", action="store_true", help="Disable Claude thinking mode")
    parser.add_argument("--use-isp", action="store_true", help="Use Claude interleaved scratchpad mode")

    # Stuck detector config
    parser.add_argument(
        "--stuck_detector_path", 
        type=str, 
        default="/gpfs/radev/scratch/cohan/jw3278/modernbert-stuck-detector",
        help="Path to the fine-tuned ModernBERT stuck detector model"
    )
    parser.add_argument(
        "--stuck_detector_device",
        type=str,
        default=None,
        help="Device for ModernBERT stuck detector inference. Examples: 'cpu', 'cuda', 'cuda:2'. "
             "If unset, auto-detects ('cuda' if available else 'cpu'). You can also set "
             "env var STUCK_DETECTOR_DEVICE.",
    )
    parser.add_argument("--stuck_threshold", type=float, default=0.5, help="Probability threshold for stuck detection")
    parser.add_argument("--min_steps_to_check", type=int, default=3, help="Minimum steps before checking for stuck")
    parser.add_argument("--use_dummy_detector", action="store_true", help="Use dummy detector (no actual stuck detection)")

    # Milestone detector config
    parser.add_argument(
        "--milestone_detector_path",
        type=str,
        default="/gpfs/radev/scratch/cohan/jw3278/modernbert-milestone-detector",
        help="Path to the fine-tuned ModernBERT milestone detector model"
    )
    parser.add_argument(
        "--milestone_detector_device",
        type=str,
        default=None,
        help="Device for ModernBERT milestone detector inference. Examples: 'cpu', 'cuda', 'cuda:2'. "
             "If unset, uses the same device as stuck detector.",
    )
    parser.add_argument("--milestone_threshold", type=float, default=0.5, help="Probability threshold for milestone detection")
    parser.add_argument("--milestone_context_steps", type=int, default=5, help="Number of previous steps to include as context for milestone detection")
    parser.add_argument("--use_dummy_milestone_detector", action="store_true", help="Use dummy milestone detector (no actual milestone detection)")
    parser.add_argument("--disable_milestone_detection", action="store_true", help="Completely disable milestone detection")

    # Example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--skip_domains",
        type=str,
        nargs="+",
        default=[],
        help=(
            "One or more domain names to skip (e.g. --skip_domains chrome libreoffice_calc). "
            "All tasks belonging to the specified domains will be excluded."
        ),
    )
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help=(
            "Run a single task by its ID. The domain will be looked up from --test_all_meta_path. "
            "Example: --task_id 035f41ba-6653-43ab-aa63-c86d449d62e5"
        ),
    )
    parser.add_argument(
        "--task_ids_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file specifying the exact tasks to run. "
            "Supported formats: "
            "(1) list of {\"domain\": <str>, \"task_id\": <str>} objects (like rerun_taskids.json); "
            "(2) dict mapping domain -> [task_id, ...]; "
            "(3) list of task_id strings (will be mapped to domains using --test_all_meta_path). "
            "If set, this overrides domain selection from --test_all_meta_path."
        ),
    )

    # Logging config
    parser.add_argument("--result_dir", type=str, default="./hybrid_results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    # AWS config
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region for the VM")
    parser.add_argument("--provider_name", type=str, default="aws", 
                       choices=["aws", "virtualbox", "vmware", "docker", "azure"], help="Provider name")
    parser.add_argument("--client_password", type=str, default="", help="Client password")
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")
    
    args = parser.parse_args()
    return args


args = config()

logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "hybrid-normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "hybrid-debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

# Add task context filter
task_filter = TaskContextFilter()
file_handler.addFilter(task_filter)
debug_handler.addFilter(task_filter)
stdout_handler.addFilter(task_filter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def load_task_ids_from_json(path: str, *, meta_lookup_path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load a task selection JSON into the test_all_meta-like dict format.

    Returns:
        Dict[str, List[str]]: mapping domain -> list of task IDs (example IDs).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow wrapper object like {"tasks": [...]}
    if isinstance(data, dict) and "tasks" in data and len(data) == 1:
        data = data["tasks"]

    # Format: dict mapping domain -> [task_id, ...]
    if isinstance(data, dict):
        out: Dict[str, List[str]] = {}
        for domain, ids in data.items():
            if not isinstance(ids, list):
                raise ValueError(
                    f"Invalid task_ids_json format: expected list for domain '{domain}', got {type(ids).__name__}"
                )
            seen: Set[str] = set()
            out_ids: List[str] = []
            for tid in ids:
                tid_s = str(tid)
                if tid_s not in seen:
                    seen.add(tid_s)
                    out_ids.append(tid_s)
            out[str(domain)] = out_ids
        return out

    # Format: list
    if isinstance(data, list):
        if len(data) == 0:
            return {}

        # List of strings -> map to domains using meta_lookup_path
        if all(isinstance(x, str) for x in data):
            if not meta_lookup_path:
                raise ValueError(
                    "task_ids_json is a list of task_id strings; provide --test_all_meta_path for domain lookup"
                )
            with open(meta_lookup_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            id_to_domain: Dict[str, str] = {}
            for domain, ids in meta.items():
                for tid in ids:
                    id_to_domain[str(tid)] = str(domain)

            out: Dict[str, List[str]] = {}
            seen_global: Set[str] = set()
            for tid in data:
                if tid in seen_global:
                    continue
                seen_global.add(tid)
                domain = id_to_domain.get(tid)
                if not domain:
                    raise ValueError(f"Task ID '{tid}' not found in meta file '{meta_lookup_path}'")
                out.setdefault(domain, []).append(tid)
            return out

        # List of objects -> expect {domain, task_id}
        if all(isinstance(x, dict) for x in data):
            out: Dict[str, List[str]] = {}
            seen_per_domain: Dict[str, Set[str]] = {}
            for obj in data:
                domain = obj.get("domain")
                task_id = obj.get("task_id") or obj.get("example_id") or obj.get("taskid")
                if not domain or not task_id:
                    raise ValueError(
                        f"Invalid entry in task_ids_json '{path}': expected keys 'domain' and 'task_id', got {obj}"
                    )
                domain_s = str(domain)
                tid_s = str(task_id)
                seen = seen_per_domain.setdefault(domain_s, set())
                if tid_s in seen:
                    continue
                seen.add(tid_s)
                out.setdefault(domain_s, []).append(tid_s)
            return out

        raise ValueError(
            f"Invalid task_ids_json list format in '{path}'. "
            "Expected list of strings or list of objects with {domain, task_id}."
        )

    raise ValueError(
        f"Invalid task_ids_json root type in '{path}': expected list or dict, got {type(data).__name__}"
    )


def run_env_tasks(task_queue: Queue, args: argparse.Namespace, shared_scores: list):
    active_environments = []
    env = None
    stuck_detector = None
    milestone_detector = None
    milestone_judge = None
    
    try:
        REGION = args.region
        screen_size = (args.screen_width, args.screen_height)
        
        # Determine snapshot based on provider
        snapshot_name = "init_state"
        if args.provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP
            ami_id = IMAGE_ID_MAP[REGION].get(screen_size, IMAGE_ID_MAP[REGION].get((1920, 1080)))
            snapshot_name = ami_id

        # Create environment
        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=REGION,
            snapshot_name=snapshot_name,
            screen_size=screen_size,
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=True,
            client_password=args.client_password
        )
        active_environments.append(env)
        
        # Initialize stuck detector (once per process)
        logger.info("Initializing stuck detector...")
        stuck_device = args.stuck_detector_device or os.environ.get("STUCK_DETECTOR_DEVICE")
        stuck_detector = create_stuck_detector(
            model_path=args.stuck_detector_path,
            use_dummy=args.use_dummy_detector,
            device=stuck_device,
            stuck_threshold=args.stuck_threshold,
            min_steps_to_check=args.min_steps_to_check,
        )
        logger.info("Stuck detector initialized.")
        
        # Initialize milestone detector (once per process)
        if not getattr(args, 'disable_milestone_detection', False):
            logger.info("Initializing milestone detector...")
            milestone_device = args.milestone_detector_device or stuck_device or os.environ.get("MILESTONE_DETECTOR_DEVICE")
            milestone_detector = create_milestone_detector(
                model_path=args.milestone_detector_path,
                use_dummy=getattr(args, 'use_dummy_milestone_detector', False),
                device=milestone_device,
                milestone_threshold=args.milestone_threshold,
                context_steps=args.milestone_context_steps,
            )
            logger.info("Milestone detector initialized.")
            
            # Initialize milestone judge (uses Claude to verify milestones)
            logger.info("Initializing milestone judge...")
            try:
                milestone_judge = MilestoneJudge(
                    model=args.claude_model,
                    max_tokens=2048,
                )
                logger.info("Milestone judge initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize milestone judge: {e}")
                milestone_judge = None
        else:
            logger.info("Milestone detection is disabled.")
        
        logger.info(f"Process {current_process().name} started.")
        
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
                
            domain, example_id = item
            set_task_context(domain, example_id)
            
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                    
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {example['instruction']}")
                
                # Use combined model name for result directory
                model_name = f"Hybrid-{args.evocua_model}-{args.claude_model}"
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    model_name,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)
                
                # Initialize EvoCUA Agent
                evocua_agent = EvoCUAAgent(
                    model=args.evocua_model,
                    max_tokens=args.evocua_max_tokens,
                    top_p=args.evocua_top_p,
                    temperature=args.evocua_temperature,
                    action_space=args.action_space,
                    observation_type=args.observation_type,
                    max_steps=args.max_steps,
                    prompt_style=args.prompt_style,
                    max_history_turns=args.max_history_turns,
                    screen_size=screen_size,
                    coordinate_type=args.coordinate_type,
                    password=args.password,
                    resize_factor=args.resize_factor,
                )
                
                # Initialize Claude Agent
                claude_agent = AnthropicAgent(
                    env=env,
                    model=args.claude_model,
                    max_tokens=args.claude_max_tokens,
                    top_p=args.claude_top_p,
                    temperature=args.claude_temperature,
                    action_space="claude_computer_use",  # Claude uses its own action space
                    observation_type=args.observation_type,
                    max_trajectory_length=args.claude_max_trajectory_length,
                    provider_name=args.provider_name,
                    screen_size=screen_size,
                    no_thinking=getattr(args, 'no_thinking', False),
                    use_isp=getattr(args, 'use_isp', False),
                )
                
                try:
                    lib_run_single.run_single_example_hybrid(
                        evocua_agent=evocua_agent,
                        claude_agent=claude_agent,
                        stuck_detector=stuck_detector,
                        env=env,
                        example=example,
                        max_steps=args.max_steps,
                        instruction=example["instruction"],
                        args=args,
                        example_result_dir=example_result_dir,
                        scores=shared_scores,
                        milestone_detector=milestone_detector,
                        milestone_judge=milestone_judge,
                    )
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    
                    try:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")

                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}))
                        f.write("\n")
                        
            except Exception as e:
                logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                clear_task_context()
                
    except Exception as e:
        logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes
    
    if is_terminating:
        return
    
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    
    for env in active_environments:
        try:
            logger.info(f"Closing environment...")
            env.close()
            logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    
    time.sleep(1)
    
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig
                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")
    
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
        num_envs = args.num_envs
        processes = []
        
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores),
                name=f"HybridEnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
            
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores),
                            name=f"HybridEnvProcess-Restart-{idx+1}"
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise
        scores = list(shared_scores)
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    switch_count = 0
    total_count = 0

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        total_count += 1
                        try:
                            all_result.append(
                                float(open(os.path.join(example_path, "result.txt"), "r").read())
                            )
                        except:
                            all_result.append(0.0)
                        
                        # Check if there was a switch
                        if "hybrid_summary.json" in os.listdir(example_path):
                            try:
                                with open(os.path.join(example_path, "hybrid_summary.json"), "r") as f:
                                    summary = json.load(f)
                                    if summary.get("switched_at_step") is not None:
                                        switch_count += 1
                            except:
                                pass

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print(f"Current Success Rate: {sum(all_result) / len(all_result) * 100:.2f}%")
        print(f"Total tasks: {total_count}, Switched to Claude: {switch_count} ({switch_count/total_count*100:.2f}%)")
        return all_result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        args = config()
        
        # Use combined model name for result directory
        model_name = f"Hybrid-{args.evocua_model}-{args.claude_model}"
        
        path_to_args = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            model_name,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)

        if args.task_id:
            # Single task mode: look up domain from meta file
            logger.info(f"Running single task: {args.task_id}")
            with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
                full_meta = json.load(f)
            
            # Find the domain for this task_id
            found_domain = None
            for domain, task_ids in full_meta.items():
                if args.task_id in task_ids:
                    found_domain = domain
                    break
            
            if found_domain is None:
                logger.error(f"Task ID '{args.task_id}' not found in {args.test_all_meta_path}")
                sys.exit(1)
            
            logger.info(f"Found task in domain: {found_domain}")
            test_all_meta = {found_domain: [args.task_id]}
        elif args.task_ids_json:
            logger.info(f"Loading tasks from task list JSON: {args.task_ids_json}")
            test_all_meta = load_task_ids_from_json(
                args.task_ids_json, meta_lookup_path=args.test_all_meta_path
            )
        else:
            with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
                test_all_meta = json.load(f)

        if args.domain != "all" and not args.task_id:
            if args.domain not in test_all_meta:
                logger.error(f"Domain '{args.domain}' not found in selected tasks.")
                sys.exit(1)
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        # Filter out skipped domains
        if args.skip_domains:
            skipped = []
            for skip_d in args.skip_domains:
                if skip_d in test_all_meta:
                    skipped.append(skip_d)
                    del test_all_meta[skip_d]
                else:
                    logger.warning(f"Skip domain '{skip_d}' not found in task list, ignoring.")
            if skipped:
                logger.info(f"Skipping domains: {', '.join(skipped)}")

        test_file_list = get_unfinished(
            args.action_space,
            model_name,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info(f"Left tasks:\n{left_info}")

        get_result(
            args.action_space,
            model_name,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        test(args, test_file_list)
        
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        signal_handler(signal.SIGTERM, None)
    finally:
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    logger.info("Closing environment in final cleanup...")
                    env.close()
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")
        
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
        
        time.sleep(1)
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")
