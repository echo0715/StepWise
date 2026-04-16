#!/usr/bin/env python3
"""Run WebArena-verified with BERT-based model selection.

Two modes:
  1. --mode static  (default): Uses a pre-computed predictions file to assign
     each task to gpt-oss-20b or gpt-5-mini before running.
  2. --mode builtin: Uses BERTSwitchingAgent which loads the BERT model and
     decides per-task at runtime (no prediction file needed).

Usage:
    # Built-in switching (recommended) — BERT is loaded inside the agent:
    python scripts/bert/run_with_routing.py --mode builtin \
        --stuck-threshold 0.1 --milestone-threshold 0.1

    # Static routing — uses pre-computed prediction file:
    python scripts/bert/run_with_routing.py --mode static \
        --predictions scripts/bert/output/routing_predictions.json

    # Specify sites, task range, parallelism:
    python scripts/bert/run_with_routing.py --mode builtin \
        --sites shopping,reddit --start 0 --end 400 --n-jobs 8
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import warnings
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *

from agentlab.agents.generic_agent.agent_configs import (
    FLAGS_GPT_4o,
    AGENT_AZURE_5,
    AGENT_AZURE_5_2,
    AGENT_AZURE_5_MINI,
)
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.exp_utils import RESULTS_DIR, add_dependencies
from agentlab.experiments.loop import ExpArgs, EnvArgs
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import SelfHostedModelArgs

from switching_agent import BERTSwitchingAgentArgs

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

ALL_SITES = ["shopping", "shopping_admin", "reddit", "gitlab", "map", "wikipedia"]
DEFAULT_SITES = ["shopping", "shopping_admin", "reddit", "gitlab"]

MODEL_OSS20B = "gpt-oss-20b"
MODEL_GPT5 = "gpt-5"
MODEL_GPT52 = "gpt-5.2"
MODEL_GPT5MINI = "gpt-5-mini"
DEFAULT_SMALL_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_SMALL_MODEL_URL = f"http://localhost:{VLLM_PORT}/v1"
DEFAULT_SMALL_MAX_TOTAL_TOKENS = 131_072
DEFAULT_SMALL_MAX_NEW_TOKENS = 4096
LARGE_AGENT_BY_NAME = {
    MODEL_GPT5: AGENT_AZURE_5,
    MODEL_GPT52: AGENT_AZURE_5_2,
    MODEL_GPT5MINI: AGENT_AZURE_5_MINI,
}


def format_threshold_tag(value: float) -> str:
    """Format threshold values consistently for run naming."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------


def build_small_agent(
    model_name: str = DEFAULT_SMALL_MODEL_NAME,
    model_url: str = DEFAULT_SMALL_MODEL_URL,
    max_total_tokens: int = DEFAULT_SMALL_MAX_TOTAL_TOKENS,
    max_new_tokens: int = DEFAULT_SMALL_MAX_NEW_TOKENS,
) -> GenericAgentArgs:
    """Build GenericAgentArgs for the self-hosted small model."""
    chat_model_args = SelfHostedModelArgs(
        model_name=model_name,
        model_url=model_url,
        backend="vllm",
        temperature=1.0,
        top_p=1.0,
        max_total_tokens=max_total_tokens,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
    )
    flags = FLAGS_GPT_4o.copy()
    flags.extra_instructions = "Reasoning: medium"
    return GenericAgentArgs(
        chat_model_args=chat_model_args,
        flags=flags,
    )


def build_large_agent(model_name: str) -> GenericAgentArgs:
    """Build GenericAgentArgs for the configured Azure large model."""
    agent = LARGE_AGENT_BY_NAME.get(model_name)
    if agent is None:
        raise ValueError(f"Unsupported large model: {model_name}")
    return deepcopy(agent)


def build_switching_agent(
    stuck_bert_dir: str = None,
    milestone_bert_dir: str = None,
    small_model_name: str = DEFAULT_SMALL_MODEL_NAME,
    small_model_url: str = DEFAULT_SMALL_MODEL_URL,
    small_max_total_tokens: int = DEFAULT_SMALL_MAX_TOTAL_TOKENS,
    small_max_new_tokens: int = DEFAULT_SMALL_MAX_NEW_TOKENS,
    large_model_name: str = MODEL_GPT5MINI,
    *,
    stuck_threshold: float,
    milestone_threshold: float,
) -> BERTSwitchingAgentArgs:
    """Build BERTSwitchingAgentArgs with stuck + milestone BERTs.

    Uses the small model by default. Switches to the large model
    (gpt-5-mini) when stuck is detected or milestone verification fails.
    """
    small_args = SelfHostedModelArgs(
        model_name=small_model_name,
        model_url=small_model_url,
        backend="vllm",
        temperature=1.0,
        top_p=1.0,
        max_total_tokens=small_max_total_tokens,
        max_new_tokens=small_max_new_tokens,
        max_input_tokens=small_max_total_tokens - small_max_new_tokens,
    )
    large_args = build_large_agent(large_model_name).chat_model_args

    flags = FLAGS_GPT_4o.copy()
    flags.extra_instructions = "Reasoning: medium"

    return BERTSwitchingAgentArgs(
        small_model_args=small_args,
        large_model_args=large_args,
        flags=flags,
        stuck_bert_dir=stuck_bert_dir or str(BERT_OUTPUT_ROOT / "modernbert-stuck-detector"),
        milestone_bert_dir=milestone_bert_dir or str(BERT_OUTPUT_ROOT / "modernbert-milestone-detector"),
        stuck_threshold=stuck_threshold,
        milestone_threshold=milestone_threshold,
    )


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def load_routing_map(predictions_path: str) -> dict[str, str]:
    """Load routing predictions from JSON.

    Supports two formats:
      1. routing_map.json: ``{task_name: model_name, ...}``
      2. routing_predictions.json: ``[{"task_name": ..., "predicted_model": ...}, ...]``

    Parameters
    ----------
    predictions_path : str
        Path to the JSON predictions file.

    Returns
    -------
    dict[str, str]
        Mapping from task_name to model name (``"gpt-oss-20b"`` or ``"gpt-5-mini"``).
    """
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Format 1: direct mapping {task_name: model_name}
        return data
    elif isinstance(data, list):
        # Format 2: list of dicts with task_name and predicted_model
        routing_map = {}
        for entry in data:
            task_name = entry.get("task_name")
            model = entry.get("predicted_model")
            if task_name and model:
                routing_map[task_name] = model
        return routing_map
    else:
        raise ValueError(f"Unexpected JSON format in {path}: expected dict or list of dicts")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def get_vllm_served_models(model_url: str) -> set[str] | None:
    """Return the set of model IDs served by the vLLM endpoint."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{model_url.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
        return {
            item.get("id")
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }
    except Exception:
        return None


def require_vllm_model(model_url: str, model_name: str) -> None:
    """Fail fast unless the expected self-hosted model is actually served."""
    served_models = get_vllm_served_models(model_url)
    if served_models is None:
        raise RuntimeError(
            f"vLLM server is NOT reachable at {model_url}, but tasks require self-hosted "
            f"model '{model_name}'. Start the server first or use the correct --small-model-url."
        )
    if model_name not in served_models:
        served_list = ", ".join(sorted(served_models)) if served_models else "<none>"
        raise RuntimeError(
            f"vLLM server at {model_url} is reachable but does not serve required model "
            f"'{model_name}'. Served models: {served_list}. Stop the existing server or "
            f"use a different --small-model-url."
        )


def check_azure_key() -> bool:
    """Check if the Azure OpenAI API key is set."""
    return bool(os.environ.get("AZURE_OPENAI_API_KEY"))


def preflight_checks(
    routing_map: dict[str, str],
    task_names: set[str],
    small_model_name: str,
    small_model_url: str,
) -> None:
    """Validate required model availability before launching work.

    Parameters
    ----------
    routing_map : dict[str, str]
        The task-to-model routing map.
    task_names : set[str]
        The set of task names that will be evaluated.
    small_model_name : str
        The required self-hosted model ID.
    small_model_url : str
        Base URL for the self-hosted endpoint.
    """
    routed_models = {routing_map.get(tn, MODEL_GPT5MINI) for tn in task_names}

    if MODEL_OSS20B in routed_models:
        require_vllm_model(small_model_url, small_model_name)

    if MODEL_GPT5MINI in routed_models and not check_azure_key():
        n_mini = sum(
            1 for tn in task_names if routing_map.get(tn, MODEL_GPT5MINI) == MODEL_GPT5MINI
        )
        warnings.warn(
            f"AZURE_OPENAI_API_KEY is not set, but {n_mini} tasks are routed to {MODEL_GPT5MINI}. "
            f"Those tasks will fail at runtime unless the key is configured.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Study info printer (adapted from main_webarena_verified.py)
# ---------------------------------------------------------------------------


def print_study_info(study_dir, benchmark_name, include_sites, start, end):
    """Print task filtering summary and progress."""
    import pandas as pd
    import browsergym.experiments

    pkg_dir = Path(browsergym.experiments.__file__).parent
    metadata_csv = pkg_dir / "benchmark" / "metadata" / f"{benchmark_name}.csv"
    if not metadata_csv.exists():
        print(f"  Metadata CSV not found: {metadata_csv}")
        return

    md = pd.read_csv(metadata_csv)
    all_task_names = set(md["task_name"])

    mask = pd.Series(True, index=md.index)
    if include_sites:
        inc = set(include_sites)
        mask &= md["sites"].apply(lambda s: all(x in inc for x in s.split()))
    task_ids = set(range(start, end + 1))
    mask &= md["task_id"].isin(task_ids)

    expected_tasks = set(md[mask]["task_name"])
    skipped_tasks = all_task_names - expected_tasks
    expected_md = md[md["task_name"].isin(expected_tasks)]
    skipped_md = md[md["task_name"].isin(skipped_tasks)]

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Total tasks in benchmark: {len(all_task_names)}")
    print(f"Expected: {len(expected_tasks)}  |  Skipped: {len(skipped_tasks)}")
    print(f"Task ID range: {start}-{end}")
    if include_sites:
        print(f"Include sites: {include_sites}")
    print(f"\nExpected tasks by site:")
    for site, count in expected_md["sites"].value_counts().items():
        print(f"  {site}: {count}")
    if skipped_tasks:
        print(f"\nSkipped tasks by site:")
        for site, count in skipped_md["sites"].value_counts().items():
            print(f"  {site}: {count}")

    # --- Scan study dir for progress ---
    study_path = Path(study_dir)
    if not study_path.exists():
        print(f"\nStudy dir does not exist yet. Will create on first run.")
        print(f"{'=' * 60}\n")
        return

    exp_dirs = [
        d
        for d in study_path.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]

    task_status = {}
    n_success = 0
    for d in exp_dirs:
        m = re.search(r"_on_(.+)_\d+$", d.name)
        if not m:
            continue
        tn = m.group(1)
        sf = d / "summary_info.json"
        if sf.exists():
            try:
                info = json.loads(sf.read_text())
                if info.get("err_msg") is not None:
                    task_status[tn] = "error"
                elif info.get("terminated") or info.get("truncated"):
                    task_status[tn] = "done"
                    if (info.get("cum_reward") or 0) > 0:
                        n_success += 1
                else:
                    task_status[tn] = "incomplete"
            except Exception:
                task_status[tn] = "incomplete"
        else:
            task_status[tn] = "incomplete"

    n_done = sum(1 for s in task_status.values() if s == "done")
    n_err = sum(1 for s in task_status.values() if s == "error")
    not_started = expected_tasks - set(task_status.keys())
    to_run = len(not_started) + sum(1 for s in task_status.values() if s != "done")

    print(
        f"\nProgress: {n_done}/{len(expected_tasks)} done, {n_err} errors, "
        f"{len(not_started)} not started, {n_success} successes"
    )
    print(f"Tasks to re-run/start: {to_run}")

    remaining = [
        (t, task_status.get(t, "not_started"))
        for t in sorted([t for t, s in task_status.items() if s != "done"] + list(not_started))
    ]
    if 0 < len(remaining) <= 50:
        for t, s in remaining:
            print(f"  [{s}] {t}")
    elif len(remaining) > 50:
        print(f"  (showing first 20 of {len(remaining)})")
        for t, s in remaining[:20]:
            print(f"  [{s}] {t}")
        print(f"  ... and {len(remaining) - 20} more")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Run WebArena-verified with BERT-based model selection."
    )
    p.add_argument(
        "--mode",
        default="builtin",
        choices=["builtin", "static"],
        help=(
            "Routing mode. 'builtin': BERTSwitchingAgent with BERT loaded in-process "
            "(no prediction file needed). 'static': uses a pre-computed prediction file. "
            "Default: builtin"
        ),
    )
    p.add_argument(
        "--stuck-bert-dir",
        default=str(BERT_OUTPUT_ROOT / "modernbert-stuck-detector"),
        help="Path to stuck-detector BERT (for --mode builtin).",
    )
    p.add_argument(
        "--milestone-bert-dir",
        default=str(BERT_OUTPUT_ROOT / "modernbert-milestone-detector"),
        help="Path to milestone-detector BERT (for --mode builtin).",
    )
    p.add_argument(
        "--predictions",
        default=str(PREDICTIONS_JSON),
        help=(
            "Path to routing predictions JSON (for --mode static). "
            f"Default: {PREDICTIONS_JSON}"
        ),
    )
    p.add_argument(
        "--sites",
        default=",".join(DEFAULT_SITES),
        help=f"Comma-separated sites (default: {','.join(DEFAULT_SITES)})",
    )
    p.add_argument(
        "--small-model-name",
        default=DEFAULT_SMALL_MODEL_NAME,
        help=f"Self-hosted small-model name (default: {DEFAULT_SMALL_MODEL_NAME})",
    )
    p.add_argument(
        "--small-model-url",
        default=DEFAULT_SMALL_MODEL_URL,
        help=f"Self-hosted small-model URL (default: {DEFAULT_SMALL_MODEL_URL})",
    )
    p.add_argument(
        "--small-max-total-tokens",
        type=int,
        default=DEFAULT_SMALL_MAX_TOTAL_TOKENS,
        help=f"Max total tokens for the small model (default: {DEFAULT_SMALL_MAX_TOTAL_TOKENS})",
    )
    p.add_argument(
        "--small-max-new-tokens",
        type=int,
        default=DEFAULT_SMALL_MAX_NEW_TOKENS,
        help=f"Max new tokens for the small model (default: {DEFAULT_SMALL_MAX_NEW_TOKENS})",
    )
    p.add_argument(
        "--large-model-name",
        default=MODEL_GPT5MINI,
        choices=sorted(LARGE_AGENT_BY_NAME),
        help=f"Azure large model to switch to (default: {MODEL_GPT5MINI})",
    )
    p.add_argument(
        "--stuck-threshold",
        type=float,
        help=(
            "Positive-class probability threshold for stuck-triggered switching "
            "(required for --mode builtin)"
        ),
    )
    p.add_argument(
        "--milestone-threshold",
        type=float,
        help=(
            "Positive-class probability threshold for milestone-triggered verification "
            "(required for --mode builtin)"
        ),
    )
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("--start", type=int, default=0, help="Start task ID (default: 0)")
    p.add_argument("--end", type=int, default=812, help="End task ID (default: 812)")
    p.add_argument("--study-dir", help="Override study directory name")
    p.add_argument(
        "--default-model",
        default=MODEL_GPT5MINI,
        choices=[MODEL_OSS20B, MODEL_GPT5MINI],
        help=f"Model for tasks not in predictions (default: {MODEL_GPT5MINI})",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing results dir and start fresh",
    )
    p.add_argument(
        "--backend",
        default="ray",
        choices=["ray", "sequential", "joblib"],
        help="Parallel backend (default: ray). Use 'sequential' for debugging.",
    )
    args = p.parse_args()
    if args.mode == "builtin":
        missing = []
        if args.stuck_threshold is None:
            missing.append("--stuck-threshold")
        if args.milestone_threshold is None:
            missing.append("--milestone-threshold")
        if missing:
            p.error(
                "builtin mode requires explicit threshold args: "
                + ", ".join(missing)
            )
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_benchmark_and_filter(args):
    """Load WebArena-verified benchmark and filter by sites/task ID range."""
    import bgym

    include_sites = [s.strip() for s in args.sites.split(",")]
    benchmark_name = "webarena_verified"
    benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name]()

    # Filter by sites
    if include_sites:
        include_set = set(include_sites)
        md = benchmark.task_metadata
        allowed_tasks = set(
            md[md["sites"].apply(lambda s: all(x in include_set for x in s.split()))]["task_name"]
        )
        benchmark.env_args_list = [
            ea for ea in benchmark.env_args_list if ea.task_name in allowed_tasks
        ]

    # Filter by task ID range
    task_ids = set(range(args.start, args.end + 1))
    benchmark.env_args_list = [
        ea for ea in benchmark.env_args_list if int(ea.task_name.split(".")[-2]) in task_ids
    ]

    return benchmark, benchmark_name, include_sites


def _get_study_dir(args):
    """Determine the study directory path."""
    results_root = PROJECT_ROOT / "results"
    if args.study_dir:
        return results_root / args.study_dir
    small_name = args.small_model_name.split("/")[-1]
    large_name = args.large_model_name if args.mode == "builtin" else MODEL_GPT5MINI
    suffix = (
        (
            f"BERTSwitch-{format_threshold_tag(args.stuck_threshold)}-"
            f"{format_threshold_tag(args.milestone_threshold)}-{small_name}+{large_name}"
        )
        if args.mode == "builtin"
        else f"BERTRouted-{small_name}+{large_name}"
    )
    return results_root / f"{suffix}_webarena-verified_{args.start}-{args.end}"


def _handle_relaunch(study, exp_args_list, benchmark):
    """Handle relaunch if study dir already exists."""
    full_list = {ea.env_args.task_name: ea for ea in exp_args_list}
    study.find_incomplete(include_errors=True)

    # Drop tasks from disk that are not in the current filter
    dropped = [
        ea.env_args.task_name
        for ea in study.exp_args_list
        if ea.env_args.task_name not in full_list
    ]
    if dropped:
        print(f"Dropping {len(dropped)} tasks not in current include_sites.")
        study.exp_args_list = [
            ea for ea in study.exp_args_list if ea.env_args.task_name in full_list
        ]

    # Add new tasks not yet on disk
    on_disk = {ea.env_args.task_name for ea in study.exp_args_list}
    new_tasks = [ea for name, ea in full_list.items() if name not in on_disk]
    if new_tasks:
        print(f"Found {len(new_tasks)} new tasks not yet on disk, adding to study.")
        study.exp_args_list.extend(new_tasks)

    # Deduplicate
    seen = set()
    deduped = []
    for ea in study.exp_args_list:
        tn = ea.env_args.task_name
        if tn not in seen:
            seen.add(tn)
            deduped.append(ea)
    if len(deduped) < len(study.exp_args_list):
        print(f"Deduped {len(study.exp_args_list) - len(deduped)} duplicate task entries.")
    study.exp_args_list = deduped

    # Re-add dependencies
    task_deps = benchmark.dependency_graph_over_tasks()
    if task_deps:
        add_dependencies(study.exp_args_list, task_deps)


# ---------------------------------------------------------------------------
# Mode: builtin — BERTSwitchingAgent decides per-task at runtime
# ---------------------------------------------------------------------------


def run_builtin_mode(args):
    """Run with BERTSwitchingAgent: stuck + milestone BERTs loaded in agent.

    A single BERTSwitchingAgent is used for all tasks. At each step it runs
    stuck and milestone BERTs on the trajectory. If stuck → switch to large
    model. If milestone → verify with large model, switch if verification fails.
    """
    benchmark, benchmark_name, include_sites = _load_benchmark_and_filter(args)
    study_dir = _get_study_dir(args)

    if args.overwrite and study_dir.exists():
        print(f"Overwrite: deleting {study_dir}")
        shutil.rmtree(study_dir)

    relaunch = study_dir.exists()

    require_vllm_model(args.small_model_url, args.small_model_name)

    # Build the switching agent
    switching_agent = build_switching_agent(
        args.stuck_bert_dir,
        args.milestone_bert_dir,
        args.small_model_name,
        args.small_model_url,
        args.small_max_total_tokens,
        args.small_max_new_tokens,
        args.large_model_name,
        stuck_threshold=args.stuck_threshold,
        milestone_threshold=args.milestone_threshold,
    )
    switching_agent.set_benchmark(benchmark, demo_mode=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BERT Switching Agent — WebArena-Verified")
    print(f"{'=' * 60}")
    print(f"Mode:           builtin (stuck + milestone BERTs)")
    print(f"Stuck BERT:     {args.stuck_bert_dir}")
    print(f"Milestone BERT: {args.milestone_bert_dir}")
    print(f"Stuck threshold: {args.stuck_threshold}")
    print(f"Milestone threshold: {args.milestone_threshold}")
    print(f"Small model:    {args.small_model_name} (default)")
    print(f"Large model:    {args.large_model_name} (on switch)")
    print(f"Sites:          {include_sites}")
    print(f"Task range:     {args.start}-{args.end}")
    print(f"Total tasks:    {len(benchmark.env_args_list)}")
    print(f"Study dir:      {study_dir}")
    print(f"N jobs:         {args.n_jobs}")
    print(f"Relaunch:       {relaunch}")
    print(f"{'=' * 60}")

    print_study_info(study_dir, benchmark_name, include_sites, args.start, args.end)

    # Create study with the single switching agent
    study = Study(
        agent_args=[switching_agent],
        benchmark=benchmark,
        logging_level_stdout=logging.WARNING,
    )
    study.dir = study_dir

    if relaunch:
        _handle_relaunch(study, study.exp_args_list, benchmark)

    study.run(
        n_jobs=args.n_jobs,
        parallel_backend=args.backend,
        strict_reproducibility=False,
        n_relaunch=1 if relaunch else 3,
    )


# ---------------------------------------------------------------------------
# Mode: static — pre-computed prediction file assigns tasks to models
# ---------------------------------------------------------------------------


def run_static_mode(args):
    """Run with static routing: each task is pre-assigned to a model."""
    benchmark, benchmark_name, include_sites = _load_benchmark_and_filter(args)
    study_dir = _get_study_dir(args)

    if args.overwrite and study_dir.exists():
        print(f"Overwrite: deleting {study_dir}")
        shutil.rmtree(study_dir)

    relaunch = study_dir.exists()

    # Load routing predictions
    routing_map = load_routing_map(args.predictions)
    logger.info(f"Loaded routing predictions for {len(routing_map)} tasks from {args.predictions}")

    # Build agents
    oss20b_agent = build_small_agent(
        args.small_model_name,
        args.small_model_url,
        args.small_max_total_tokens,
        args.small_max_new_tokens,
    )
    gpt5mini_agent = build_large_agent(MODEL_GPT5MINI)
    agent_lookup = {MODEL_OSS20B: oss20b_agent, MODEL_GPT5MINI: gpt5mini_agent}

    # Pre-flight checks
    task_names = {ea.task_name for ea in benchmark.env_args_list}
    preflight_checks(routing_map, task_names, args.small_model_name, args.small_model_url)

    # Set benchmark on both agents
    oss20b_agent.set_benchmark(benchmark, demo_mode=False)
    gpt5mini_agent.set_benchmark(benchmark, demo_mode=False)

    # Route tasks
    exp_args_list = []
    n_oss20b, n_gpt5mini, n_default = 0, 0, 0
    for env_args in benchmark.env_args_list:
        predicted = routing_map.get(env_args.task_name, args.default_model)
        if env_args.task_name not in routing_map:
            n_default += 1

        agent = agent_lookup.get(predicted)
        if agent is None:
            logger.warning(
                f"Unknown model '{predicted}' for task {env_args.task_name}, "
                f"falling back to {args.default_model}"
            )
            agent = agent_lookup[args.default_model]
            predicted = args.default_model

        if predicted == MODEL_OSS20B:
            n_oss20b += 1
        else:
            n_gpt5mini += 1

        exp_args_list.append(
            ExpArgs(
                agent_args=agent,
                env_args=env_args,
                logging_level=logging.DEBUG,
                logging_level_stdout=logging.WARNING,
            )
        )

    for i, ea in enumerate(exp_args_list):
        ea.order = i

    task_deps = benchmark.dependency_graph_over_tasks()
    if task_deps:
        exp_args_list = add_dependencies(exp_args_list, task_deps)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BERT Static Routing — WebArena-Verified")
    print(f"{'=' * 60}")
    print(f"Mode:           static (pre-computed predictions)")
    print(f"Predictions:    {args.predictions}")
    print(f"Sites:          {include_sites}")
    print(f"Task range:     {args.start}-{args.end}")
    print(f"Total tasks:    {len(exp_args_list)}")
    print(f"  -> {MODEL_OSS20B}:  {n_oss20b}")
    print(f"  -> {MODEL_GPT5MINI}: {n_gpt5mini}")
    if n_default > 0:
        print(f"  (default fallback: {n_default} tasks not in predictions)")
    print(f"Study dir:      {study_dir}")
    print(f"N jobs:         {args.n_jobs}")
    print(f"Relaunch:       {relaunch}")
    print(f"Default model:  {args.default_model}")
    print(f"{'=' * 60}")

    print_study_info(study_dir, benchmark_name, include_sites, args.start, args.end)

    # Create study
    study = Study(
        agent_args=[gpt5mini_agent],
        benchmark=benchmark,
        logging_level_stdout=logging.WARNING,
    )
    study.dir = study_dir
    study.exp_args_list = exp_args_list

    if relaunch:
        _handle_relaunch(study, exp_args_list, benchmark)

    study.run(
        n_jobs=args.n_jobs,
        parallel_backend=args.backend,
        strict_reproducibility=False,
        n_relaunch=1 if relaunch else 3,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.mode == "builtin":
        run_builtin_mode(args)
    else:
        run_static_mode(args)


if __name__ == "__main__":
    main()
