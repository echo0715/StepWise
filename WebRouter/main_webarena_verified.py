"""
Unified experiment runner for WebArena Verified benchmark.

Supports Azure models (pre-configured) and vLLM-served models.

Usage:
  # Azure models (pre-configured agents)
  python main_webarena_verified.py --model azure-gpt-5-mini
  python main_webarena_verified.py --model azure-gpt-5.2

  # vLLM models (requires vLLM server running)
  python main_webarena_verified.py --model gpt-oss-20b
  python main_webarena_verified.py --model gpt-oss-120b
  python main_webarena_verified.py --model agenttrek-32b

  # Options
  python main_webarena_verified.py --model gpt-oss-20b --sites shopping,reddit --n-jobs 2 --relaunch
"""

import argparse
import logging
from pathlib import Path
import os

from agentlab.agents.browser_use_agent.browser_use_agent import BrowserUseAgentArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.llm.chat_api import SelfHostedModelArgs
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)


# ─── Model definitions ─────────────────────────────────────────────────

VLLM_MODELS = {
    "gpt-oss-20b": {
        "model_name": "openai/gpt-oss-20b",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "max_total_tokens": 131_072,
        "max_new_tokens": 4096,
        "extra_instructions": "Reasoning: medium",
    },
    "gpt-oss-120b": {
        "model_name": "openai/gpt-oss-120b",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "max_total_tokens": 131_072,
        "max_new_tokens": 4096,
        "extra_instructions": "Reasoning: medium",
    },
    "agenttrek-32b": {
        "model_name": "xlangai/AgentTrek-1.0-32B",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_total_tokens": 32_768,
        "max_new_tokens": 512,
        "extra_instructions": None,
    },
    "bu-30b": {
        "model_name": "browser-use/bu-30b-a3b-preview",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": None,
        "max_total_tokens": 65_536,
        "max_new_tokens": 4096,
        "agent_type": "browser_use",
    },
}

AZURE_MODELS = {
    "azure-gpt-5-mini": "AGENT_AZURE_5_MINI",
    "azure-gpt-5.2": "AGENT_AZURE_5_2",
    "azure-gpt-5": "AGENT_AZURE_5",
    "azure-gpt-5-nano": "AGENT_AZURE_5_NANO",
}

ALL_SITES = ["shopping", "shopping_admin", "reddit", "gitlab", "map", "wikipedia"]
DEFAULT_SITES = ["shopping", "shopping_admin", "reddit", "gitlab"]


def build_agent(model_key: str) -> tuple[GenericAgentArgs, str]:
    """Build agent args and a study directory suffix from model key."""
    if model_key in AZURE_MODELS:
        import agentlab.agents.generic_agent.agent_configs as configs

        agent = getattr(configs, AZURE_MODELS[model_key])
        suffix = f"{agent.agent_name}"
        return agent, suffix

    if model_key in VLLM_MODELS:
        cfg = VLLM_MODELS[model_key]
        max_input = cfg["max_total_tokens"] - cfg["max_new_tokens"]
        model_args = SelfHostedModelArgs(
            model_name=cfg["model_name"],
            backend="vllm",
            max_total_tokens=cfg["max_total_tokens"],
            max_input_tokens=max_input,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
        )

        if cfg.get("agent_type") == "browser_use":
            agent = BrowserUseAgentArgs(chat_model_args=model_args)
        else:
            flags = cfg.get("flags", FLAGS_GPT_4o).copy()
            if cfg.get("extra_instructions"):
                flags.extra_instructions = cfg["extra_instructions"]
            agent = GenericAgentArgs(chat_model_args=model_args, flags=flags)

        t, tp, tk = cfg["temperature"], cfg["top_p"], cfg["top_k"]
        mn, mt = cfg["max_new_tokens"], cfg["max_total_tokens"]
        suffix = f"{agent.agent_name}"
        # Append vLLM params to study dir (after benchmark name, matching legacy format)
        agent._vllm_params_suffix = f"_{t}-{tp}-{tk}-{mn}-{mt}"
        return agent, suffix

    raise ValueError(
        f"Unknown model: {model_key}. Available: {list(AZURE_MODELS) + list(VLLM_MODELS)}"
    )


def print_study_info(study_dir, benchmark_name, include_sites, start, end):
    """Print task filtering summary and progress. Fast: no study/benchmark loading."""
    import json
    import re

    import pandas as pd
    import browsergym.experiments

    pkg_dir = Path(browsergym.experiments.__file__).parent
    metadata_csv = pkg_dir / "benchmark" / "metadata" / f"{benchmark_name}.csv"
    if not metadata_csv.exists():
        print(f"  Metadata CSV not found: {metadata_csv}")
        return

    md = pd.read_csv(metadata_csv)
    all_task_names = set(md["task_name"])

    # Filter by include_sites and task_id range to get expected tasks
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

    print(f"\n{'='*60}")
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
        print(f"{'='*60}\n")
        return

    exp_dirs = [
        d for d in study_path.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]

    task_status = {}  # task_name -> "done" | "error" | "incomplete"
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

    print(f"\nProgress: {n_done}/{len(expected_tasks)} done, {n_err} errors, "
          f"{len(not_started)} not started, {n_success} successes")
    print(f"Tasks to re-run/start: {to_run}")

    remaining = [(t, task_status.get(t, "not_started")) for t in sorted(
        [t for t, s in task_status.items() if s != "done"] + list(not_started)
    )]
    if 0 < len(remaining) <= 50:
        for t, s in remaining:
            print(f"  [{s}] {t}")
    elif len(remaining) > 50:
        print(f"  (showing first 20 of {len(remaining)})")
        for t, s in remaining[:20]:
            print(f"  [{s}] {t}")
        print(f"  ... and {len(remaining) - 20} more")
    print(f"{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Run WebArena Verified experiments")
    p.add_argument(
        "--model",
        required=True,
        choices=list(AZURE_MODELS) + list(VLLM_MODELS),
        help="Model to evaluate",
    )
    p.add_argument(
        "--sites",
        default=",".join(DEFAULT_SITES),
        help=f"Comma-separated sites (default: {','.join(DEFAULT_SITES)})",
    )
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("--start", type=int, default=0, help="Start task ID (default: 0)")
    p.add_argument("--end", type=int, default=812, help="End task ID (default: 812)")
    p.add_argument("--overwrite", action="store_true", help="Delete existing results dir and start fresh")
    p.add_argument("--study-dir", help="Override study directory name")
    return p.parse_args()


def main():
    args = parse_args()
    agent, suffix = build_agent(args.model)
    agent_args = [agent]

    include_sites = [s.strip() for s in args.sites.split(",")]
    benchmark_name = "webarena_verified"

    vllm_suffix = getattr(agent, "_vllm_params_suffix", "")
    if args.study_dir:
        study_dir = Path(os.environ["AGENTLAB_EXP_ROOT"]) / "results" / args.study_dir
    else:
        study_dir = (
            Path(os.environ["AGENTLAB_EXP_ROOT"])
            / "results"
            / f"{suffix}_{benchmark_name.replace('_', '-')}{vllm_suffix}_{args.start}-{args.end}"
        )

    import bgym

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
        ea
        for ea in benchmark.env_args_list
        if int(ea.task_name.split(".")[-2]) in task_ids
    ]

    # Delete existing results if --overwrite
    if args.overwrite and study_dir.exists():
        import shutil
        print(f"Overwrite: deleting {study_dir}")
        shutil.rmtree(study_dir)

    # Auto-detect relaunch: if study dir exists, continue from where we left off
    relaunch = study_dir.exists()

    print(f"Model:     {args.model}")
    print(f"Sites:     {include_sites}")
    print(f"Tasks:     {len(benchmark.env_args_list)}")
    print(f"Study dir: {study_dir}")
    print(f"N jobs:    {args.n_jobs}")
    print(f"Relaunch:  {relaunch} ({'dir exists' if study_dir.exists() else 'fresh'})")

    print_study_info(study_dir, benchmark_name, include_sites, args.start, args.end)

    study = Study(
        agent_args,
        benchmark,
        logging_level_stdout=logging.WARNING,
    )
    study.dir = study_dir

    if relaunch:
        full_list = {ea.env_args.task_name: ea for ea in study.exp_args_list}
        study.find_incomplete(include_errors=True)

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

        on_disk = {ea.env_args.task_name for ea in study.exp_args_list}
        new_tasks = [ea for name, ea in full_list.items() if name not in on_disk]
        if new_tasks:
            print(f"Found {len(new_tasks)} new tasks not yet on disk, adding to study.")
            study.exp_args_list.extend(new_tasks)

        from agentlab.experiments.exp_utils import add_dependencies

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

        task_deps = benchmark.dependency_graph_over_tasks()
        if task_deps:
            add_dependencies(study.exp_args_list, task_deps)

    study.run(
        n_jobs=args.n_jobs,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=1 if relaunch else 3,
    )


if __name__ == "__main__":
    main()
