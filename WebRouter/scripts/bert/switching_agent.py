#!/usr/bin/env python3
"""Multi-BERT model switching agent for WebArena-verified.

Uses two BERT classifiers (stuck detector + milestone detector) to dynamically
decide when to switch from a small/cheap model (gpt-oss-20b) to a large/expensive
model (gpt-5-mini) during task execution.

Logic per step:
  1. Run stuck BERT on recent trajectory context
  2. If stuck → switch to large model immediately, stay there
  3. Run milestone BERT on recent trajectory context
  4. If milestone → ask the large model to verify steps since last milestone
     - If verification fails → switch to large model
     - If verification passes → stay on small model
  5. Otherwise → continue with current model

Once switched to the large model, the agent stays on it for the rest of the task.
On reset() (new task), it reverts to the small model.

Usage:
    from scripts.bert.switching_agent import BERTSwitchingAgentArgs
    agent_args = BERTSwitchingAgentArgs(
        small_model_args=oss20b_args,
        large_model_args=gpt5mini_args,
        flags=FLAGS_GPT_4o,
        stuck_bert_dir="path/to/stuck-detector",
        milestone_bert_dir="path/to/milestone-detector",
        stuck_threshold=0.1,
        milestone_threshold=0.1,
    )
"""

import logging
import re
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from bgym import Benchmark
from browsergym.experiments.agent import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from agentlab.agents.generic_agent.generic_agent import GenericAgent
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt

logger = logging.getLogger(__name__)

# BERT label conventions (both stuck and milestone use 1 = positive)
_LABEL_POSITIVE = 1  # stuck=1 means stuck, milestone=1 means milestone

# Context window for BERT input
_CONTEXT_WINDOW = 5

# Default paths
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_STUCK_DIR = _SCRIPT_DIR / "output" / "modernbert-stuck-detector"
_DEFAULT_MILESTONE_DIR = _SCRIPT_DIR / "output" / "modernbert-milestone-detector"


def _format_threshold(value: float) -> str:
    """Format thresholds consistently for names and logs."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def _extract_goal_short(goal: str) -> str:
    """Extract core task instruction from the full goal text."""
    if not goal:
        return ""
    for sep in ["\n---", "\n===", "\nYour response should"]:
        if sep in goal:
            return goal[: goal.index(sep)].strip()
    return goal[:500].strip()


def _build_trajectory_text(
    goal_short: str, step_history: list, context_window: int = 5, include_goal: bool = True
) -> str:
    """Build BERT input: optional goal + sliding window of recent steps.

    Format matches 3rd_party/bert training data::

        Task: Get the top-1 best-selling product...
        Step 0: [click('227')] We need to view the order history...
        Step 1: [click('1530')] Navigating to orders page...

    When include_goal is False, the "Task: ..." line is omitted (used for stuck
    detection, which should be goal-agnostic).
    """
    parts = []
    if include_goal:
        parts.append(f"Task: {goal_short}")
    start = max(0, len(step_history) - context_window)
    for step_num, action, think in step_history[start:]:
        think_text = think or ""
        if think_text:
            parts.append(f"Step {step_num}: [{action}] {think_text}")
        else:
            parts.append(f"Step {step_num}: [{action}]")
    return "\n".join(parts)


def _load_trained_max_length(model_dir: str | Path, tokenizer) -> int:
    """Load the max sequence length used during fine-tuning.

    Prefer the explicit value recorded by train_router.py. Fall back to the
    saved tokenizer's model_max_length, but ignore sentinel "very large" values
    used by some tokenizers.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            max_length = int(config.get("max_length", 0))
            if max_length > 0:
                return max_length
        except Exception as exc:
            logger.warning("Failed to read %s: %s", config_path, exc)

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 100_000:
        return tokenizer_max

    return 512


VERIFICATION_PROMPT = """\
You are verifying whether an AI web agent's recent actions were correct.

Task the agent is trying to accomplish:
{goal}

Steps taken since the last verified checkpoint:
{steps}

Were these steps reasonable and making correct progress toward the task? \
Consider whether the agent is navigating to the right pages, clicking the right \
elements, and filling in correct information.

Answer with a single word: CORRECT or INCORRECT"""


# ── Agent Args (serializable config) ────────────────────────────────────────


@dataclass
class BERTSwitchingAgentArgs(AgentArgs):
    """Agent that uses stuck + milestone BERTs to decide when to switch models.

    Attributes
    ----------
    small_model_args : BaseModelArgs
        Config for the small/cheap model (gpt-oss-20b). Used by default.
    large_model_args : BaseModelArgs
        Config for the large/expensive model (gpt-5-mini). Used when switching.
    flags : GenericPromptFlags
        Prompt configuration flags (same as GenericAgent).
    stuck_bert_dir : str
        Path to the trained stuck-detector BERT model directory.
    milestone_bert_dir : str
        Path to the trained milestone-detector BERT model directory.
    stuck_threshold : float
        Positive-class probability threshold for stuck-triggered switching.
    milestone_threshold : float
        Positive-class probability threshold for milestone-triggered verification.
    max_retry : int
        Max LLM retries per step.
    """

    small_model_args: BaseModelArgs = None
    large_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    stuck_bert_dir: str = ""
    milestone_bert_dir: str = ""
    stuck_threshold: float | None = None
    milestone_threshold: float | None = None
    max_retry: int = 4

    def __post_init__(self):
        if self.stuck_threshold is None or self.milestone_threshold is None:
            raise ValueError(
                "stuck_threshold and milestone_threshold are required for "
                "BERTSwitchingAgentArgs"
            )
        if not 0.0 <= float(self.stuck_threshold) <= 1.0:
            raise ValueError(
                f"stuck_threshold must be in [0,1], got {self.stuck_threshold}"
            )
        if not 0.0 <= float(self.milestone_threshold) <= 1.0:
            raise ValueError(
                "milestone_threshold must be in [0,1], got "
                f"{self.milestone_threshold}"
            )
        try:
            small_name = self.small_model_args.model_name.replace("/", "_")
            large_name = self.large_model_args.model_name.replace("/", "_")
            stuck_tag = _format_threshold(self.stuck_threshold)
            milestone_tag = _format_threshold(self.milestone_threshold)
            self.agent_name = f"BERTSwitch-{stuck_tag}-{milestone_tag}-{small_name}-{large_name}"
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        """Adapt flags to benchmark (same logic as GenericAgentArgs)."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True
        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.small_model_args.temperature = 0
        self.large_model_args.temperature = 0

    def prepare(self):
        self.small_model_args.prepare_server()
        self.large_model_args.prepare_server()

    def close(self):
        self.small_model_args.close_server()
        self.large_model_args.close_server()

    def make_agent(self):
        return BERTSwitchingAgent(
            small_model_args=self.small_model_args,
            large_model_args=self.large_model_args,
            flags=self.flags,
            stuck_bert_dir=self.stuck_bert_dir or str(_DEFAULT_STUCK_DIR),
            milestone_bert_dir=self.milestone_bert_dir or str(_DEFAULT_MILESTONE_DIR),
            stuck_threshold=self.stuck_threshold,
            milestone_threshold=self.milestone_threshold,
            max_retry=self.max_retry,
        )


# ── Agent (runtime) ─────────────────────────────────────────────────────────


class BERTSwitchingAgent(GenericAgent):
    """Agent that uses stuck + milestone BERTs for online model switching.

    Starts every task with the small model. Before each step (except step 0),
    runs the stuck BERT and milestone BERT on the trajectory so far. If stuck
    is detected, switches to the large model permanently. If a milestone is
    detected, verifies the recent steps with the large model — if verification
    fails, switches permanently.
    """

    def __init__(
        self,
        small_model_args: BaseModelArgs,
        large_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        stuck_bert_dir: str,
        milestone_bert_dir: str,
        stuck_threshold: float,
        milestone_threshold: float,
        max_retry: int = 4,
    ):
        # ── Create both LLMs ──
        self.small_llm = small_model_args.make_model()
        self.large_llm = large_model_args.make_model()
        self.small_model_args = small_model_args
        self.large_model_args = large_model_args

        # ── Load BERT classifiers ──
        import torch

        # Monkey-patch torch.compile to a no-op on Python 3.12+ where dynamo
        # is unsupported.  ModernBERT uses @torch.compile at class-definition
        # time, which crashes before any model code runs.
        import sys

        if sys.version_info >= (3, 12):
            _orig_compile = torch.compile

            def _noop_compile(fn=None, *args, **kwargs):
                if fn is not None:
                    return fn
                return lambda f: f

            torch.compile = _noop_compile

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        # Use CPU for BERT — the installed PyTorch may not support the GPU arch
        # (e.g. Blackwell sm_120), and BERT is small enough for CPU inference.
        self._bert_device = torch.device("cpu")

        logger.info("Loading stuck BERT from %s", stuck_bert_dir)
        self.stuck_tokenizer = AutoTokenizer.from_pretrained(stuck_bert_dir)
        self.stuck_bert = AutoModelForSequenceClassification.from_pretrained(stuck_bert_dir)
        self.stuck_bert.eval()
        self.stuck_bert.to(self._bert_device)
        self._stuck_max_length = _load_trained_max_length(stuck_bert_dir, self.stuck_tokenizer)
        logger.info("Stuck BERT max_length=%d", self._stuck_max_length)

        logger.info("Loading milestone BERT from %s", milestone_bert_dir)
        self.milestone_tokenizer = AutoTokenizer.from_pretrained(milestone_bert_dir)
        self.milestone_bert = AutoModelForSequenceClassification.from_pretrained(milestone_bert_dir)
        self.milestone_bert.eval()
        self.milestone_bert.to(self._bert_device)
        self._milestone_max_length = _load_trained_max_length(
            milestone_bert_dir, self.milestone_tokenizer
        )
        logger.info("Milestone BERT max_length=%d", self._milestone_max_length)
        self._stuck_threshold = float(stuck_threshold)
        self._milestone_threshold = float(milestone_threshold)
        if not 0.0 <= self._stuck_threshold <= 1.0:
            raise ValueError(f"stuck_threshold must be in [0,1], got {self._stuck_threshold}")
        if not 0.0 <= self._milestone_threshold <= 1.0:
            raise ValueError(
                f"milestone_threshold must be in [0,1], got {self._milestone_threshold}"
            )
        logger.info(
            "BERT thresholds: stuck>=%.3f milestone>=%.3f",
            self._stuck_threshold,
            self._milestone_threshold,
        )

        # ── Initialize GenericAgent internals ──
        self.chat_llm = self.small_llm
        self.chat_model_args = small_model_args
        self.max_retry = max_retry
        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self._check_flag_constancy()

        # ── Switching state ──
        self._using_large = False
        self._switch_reason = None
        self._switch_step = None
        self._goal_short = ""
        self._step_history = []  # list of (step_num, action_str, think_str)
        self._last_milestone_idx = -1  # index in _step_history of last milestone
        self._bert_log = []  # per-step log of BERT decisions
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
        # ── Extract goal on first step ──
        if not self._goal_short:
            self._goal_short = _extract_goal_short(obs.get("goal", ""))

        # ── Before taking action: check BERTs (if we have history) ──
        step_bert_info = {"stuck": None, "milestone": None, "verification": None}

        if self._step_history and not self._using_large:
            step_bert_info = self._check_and_maybe_switch()

        # ── Take action with current model ──
        self.obs_history.append(obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        try:
            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        # ── Record this step ──
        action_str = ans_dict.get("action") or ""
        think_str = ans_dict.get("think") or ""
        step_num = len(self._step_history)
        self._step_history.append((step_num, str(action_str), think_str))

        # ── Build stats ──
        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]
        stats["bert_using_large"] = self._using_large
        stats["bert_switch_reason"] = self._switch_reason
        stats["bert_switch_step"] = self._switch_step
        stats["bert_stuck_pred"] = step_bert_info.get("stuck")
        stats["bert_milestone_pred"] = step_bert_info.get("milestone")
        stats["bert_verification"] = step_bert_info.get("verification")
        stats["bert_current_model"] = (
            self.large_model_args.model_name if self._using_large
            else self.small_model_args.model_name
        )

        self._bert_log.append(step_bert_info)

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        from dataclasses import asdict

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={
                "chat_model_args": asdict(self.chat_model_args),
                "bert_using_large": self._using_large,
                "bert_switch_reason": self._switch_reason,
                "bert_switch_step": self._switch_step,
                "bert_log": step_bert_info,
            },
        )
        return ans_dict["action"], agent_info

    # ── BERT checking logic ──────────────────────────────────────────────

    def _check_and_maybe_switch(self) -> dict:
        """Run stuck + milestone BERTs and apply switching logic.

        Returns a dict describing what happened for logging.
        """
        info = {"stuck": None, "milestone": None, "verification": None}

        # 1. Check stuck (no goal — stuck detection is goal-agnostic)
        stuck_pos_prob = self._run_bert(
            self.stuck_bert,
            self.stuck_tokenizer,
            self._stuck_max_length,
            include_goal=False,
        )
        stuck_triggered = stuck_pos_prob >= self._stuck_threshold
        info["stuck"] = {
            "positive_prob": stuck_pos_prob,
            "threshold": self._stuck_threshold,
            "triggered": stuck_triggered,
        }

        if stuck_triggered:
            logger.info(
                "STUCK detected (pos_prob=%.3f, threshold=%.3f) at step %d -> switching to large model",
                stuck_pos_prob, self._stuck_threshold, len(self._step_history),
            )
            self._switch_to_large("stuck_detected")
            return info

        # 2. Check milestone (with goal — milestone detection is goal-aware)
        milestone_pos_prob = self._run_bert(
            self.milestone_bert,
            self.milestone_tokenizer,
            self._milestone_max_length,
            include_goal=True,
        )
        milestone_triggered = milestone_pos_prob >= self._milestone_threshold
        info["milestone"] = {
            "positive_prob": milestone_pos_prob,
            "threshold": self._milestone_threshold,
            "triggered": milestone_triggered,
        }

        if milestone_triggered:
            # Verify steps since last milestone with large model
            verified, verification_response = self._verify_steps_with_large_model()
            info["verification"] = {
                "correct": verified,
                "response": verification_response[:200],
            }

            if not verified:
                logger.info(
                    "MILESTONE verification FAILED at step %d → switching to large model",
                    len(self._step_history),
                )
                self._switch_to_large("milestone_verification_failed")
            else:
                logger.info(
                    "MILESTONE verified CORRECT at step %d (pos_prob=%.3f, threshold=%.3f) -> staying on small model",
                    len(self._step_history),
                    milestone_pos_prob,
                    self._milestone_threshold,
                )

            # Update last milestone regardless of verification result
            self._last_milestone_idx = len(self._step_history) - 1

        return info

    def _run_bert(self, model, tokenizer, max_length: int, include_goal: bool) -> float:
        """Run a BERT classifier on the current trajectory context.

        Returns the positive-class probability.
        """
        torch = self._torch
        text = _build_trajectory_text(
            self._goal_short, self._step_history, _CONTEXT_WINDOW, include_goal=include_goal
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self._bert_device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            positive_prob = probs[0, _LABEL_POSITIVE].item()

        return positive_prob

    def _verify_steps_with_large_model(self) -> tuple[bool, str]:
        """Ask the large model to verify steps since the last milestone.

        Returns (is_correct, raw_response_text).
        """
        # Gather steps since last milestone
        start_idx = self._last_milestone_idx + 1
        recent_steps = self._step_history[start_idx:]

        if not recent_steps:
            return True, "no_steps_to_verify"

        # Format steps for the verification prompt
        step_lines = []
        for step_num, action, think in recent_steps:
            think_trunc = (think[:200] + "...") if think and len(think) > 200 else (think or "")
            if think_trunc:
                step_lines.append(f"Step {step_num}: [{action}] {think_trunc}")
            else:
                step_lines.append(f"Step {step_num}: [{action}]")

        prompt_text = VERIFICATION_PROMPT.format(
            goal=self._goal_short,
            steps="\n".join(step_lines),
        )

        # Call the large model for verification
        try:
            from agentlab.llm.llm_utils import HumanMessage

            messages = Discussion([
                SystemMessage("You are a helpful assistant that verifies web agent actions."),
                HumanMessage(prompt_text),
            ])
            response = self.large_llm(messages)

            # Extract text from response
            if isinstance(response, dict):
                response_text = response.get("content", str(response))
            else:
                response_text = str(response)

            # Parse: look for CORRECT or INCORRECT
            upper = response_text.upper()
            if "INCORRECT" in upper:
                return False, response_text
            elif "CORRECT" in upper:
                return True, response_text
            else:
                # Ambiguous → assume incorrect (conservative)
                logger.warning("Ambiguous verification response: %s", response_text[:100])
                return False, response_text

        except Exception as e:
            logger.error("Verification call failed: %s", e)
            # On error, don't switch (conservative for cost)
            return True, f"error: {e}"

    def _switch_to_large(self, reason: str):
        """Permanently switch to the large model."""
        self._using_large = True
        self._switch_reason = reason
        self._switch_step = len(self._step_history)
        self.chat_llm = self.large_llm
        self.chat_model_args = self.large_model_args
        logger.info(
            "SWITCHED to large model at step %d (reason: %s) for task: %s",
            self._switch_step, reason, self._goal_short[:80],
        )

    def reset(self, seed=None):
        """Reset for a new task. Reverts to the small model."""
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []
        # Revert to small model
        self.chat_llm = self.small_llm
        self.chat_model_args = self.small_model_args
        self._using_large = False
        self._switch_reason = None
        self._switch_step = None
        self._goal_short = ""
        self._step_history = []
        self._last_milestone_idx = -1
        self._bert_log = []
