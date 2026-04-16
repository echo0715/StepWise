"""
BrowserUseAgent implementation for AgentLab.

A dedicated agent for browser-use fine-tuned models (e.g. bu-30b-a3b-preview).
Instead of shoehorning the BU model into GenericAgent's AXTree + <action> tag format,
this agent uses the model's native prompt format:
  - System prompt from browser-use's system_prompt_browser_use.md
  - DOM elements formatted as [numeric_index]<tag attrs /> (not AXTree [bid] role)
  - JSON output: {"evaluation_previous_goal":..., "memory":..., "next_goal":..., "action":[...]}
  - History formatted as <agent_history> / <agent_state> / <browser_state> XML sections
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import bgym
from browsergym.experiments.agent import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    count_tokens,
    retry,
)
from agentlab.llm.tracking import cost_tracker_decorator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts – exact copies of browser_use 0.12.0 templates for
# browser-use fine-tuned models.  We load from the library when available
# and fall back to these if the installed version doesn't support
# is_browser_use_model (e.g. older releases or dependency conflicts).
# ---------------------------------------------------------------------------


def _load_bu_system_prompt(use_thinking: bool) -> str:
    """Load the browser-use system prompt, preferring the library when possible."""
    try:
        from browser_use.agent.prompts import SystemPrompt as BUSystemPrompt

        sp = BUSystemPrompt(is_browser_use_model=True, use_thinking=use_thinking)
        return sp.get_system_message().content
    except (ImportError, TypeError):
        # Fallback: hardcoded copies of system_prompt_browser_use[_no_thinking].md
        pass

    if use_thinking:
        return (
            "You are a browser-use agent operating in thinking mode. "
            "You automate browser tasks by outputting structured JSON actions.\n\n"
            "<output>\n"
            "You must ALWAYS respond with a valid JSON in this exact format:\n"
            "{{\n"
            '  "thinking": "A structured reasoning block analyzing: current page state, '
            'what was attempted, what worked/failed, and strategic planning for next steps.",\n'
            '  "evaluation_previous_goal": "Concise one-sentence analysis of your last action. '
            'Clearly state success, failure, or uncertain.",\n'
            '  "memory": "1-3 sentences of specific memory of this step and overall progress. '
            'Track items found, pages visited, forms filled, etc.",\n'
            '  "next_goal": "State the next immediate goal and action to achieve it, '
            'in one clear sentence.",\n'
            '  "action": [{{"action_name": {{...params...}}}}]\n'
            "}}\n"
            "Action list should NEVER be empty.\n"
            "</output>"
        )
    else:
        return (
            "You are a browser-use agent. "
            "You automate browser tasks by outputting structured JSON actions.\n\n"
            "<output>\n"
            "You must ALWAYS respond with a valid JSON in this exact format:\n"
            "{{\n"
            '  "evaluation_previous_goal": "Concise one-sentence analysis of your last action. '
            'Clearly state success, failure, or uncertain.",\n'
            '  "memory": "1-3 sentences of specific memory of this step and overall progress. '
            'Track items found, pages visited, forms filled, etc.",\n'
            '  "next_goal": "State the next immediate goal and action to achieve it, '
            'in one clear sentence.",\n'
            '  "action": [{{"action_name": {{...params...}}}}]\n'
            "}}\n"
            "Action list should NEVER be empty.\n"
            "</output>"
        )


# ---------------------------------------------------------------------------
# ARIA role → HTML tag mapping
# ---------------------------------------------------------------------------

ROLE_TO_TAG = {
    "link": "a",
    "button": "button",
    "textbox": "input",
    "searchbox": "input",
    "combobox": "select",
    "checkbox": "input",
    "radio": "input",
    "heading": "h2",
    "img": "img",
    "image": "img",
    "navigation": "nav",
    "main": "main",
    "form": "form",
    "list": "ul",
    "listitem": "li",
    "table": "table",
    "row": "tr",
    "cell": "td",
    "columnheader": "th",
    "textarea": "textarea",
    "tab": "div",
    "tabpanel": "div",
    "dialog": "dialog",
    "menu": "menu",
    "menubar": "menu",
    "separator": "hr",
    "slider": "input",
    "spinbutton": "input",
    "option": "option",
    "menuitem": "menuitem",
    "menuitemcheckbox": "menuitem",
    "menuitemradio": "menuitem",
    "treeitem": "li",
    "switch": "input",
    "progressbar": "progress",
    "article": "article",
    "banner": "header",
    "complementary": "aside",
    "contentinfo": "footer",
    "region": "section",
    "group": "div",
    "paragraph": "p",
    "generic": "div",
    "Section": "section",
}


def _convert_axtree_to_bu_dom(axtree_txt: str) -> tuple[str, dict[int, str]]:
    """Convert BrowserGym AXTree text to browser-use DOM format.

    BrowserGym AXTree format:
        [a1b2] button 'Submit'
        [c3d4] link 'Home page'

    Browser-use DOM format:
        [1]<button /> Submit
        [2]<a /> Home page

    Returns:
        (formatted_text, index_to_bid) where index_to_bid maps numeric indices
        back to original BrowserGym bids for action translation.
    """
    lines = axtree_txt.split("\n")
    output_lines = []
    index_to_bid = {}
    current_index = 1

    # Pattern: [bid] role 'name' or [bid] role "name" or [bid] role
    # BrowserGym bid is alphanumeric, role follows, optional quoted name
    bid_pattern = re.compile(
        r"^(\s*)"  # leading whitespace (indentation)
        r"\[([a-zA-Z0-9]+)\]\s+"  # [bid]
        r"(\S+)"  # role
        r"(?:\s+'([^']*)')?"  # optional 'name' (single quotes)
        r'(?:\s+"([^"]*)")?'  # optional "name" (double quotes)
        r"(.*)"  # rest of line
    )
    # Pattern for lines without bids (plain text content)
    text_line_pattern = re.compile(r"^(\s+)(.+)$")

    for line in lines:
        m = bid_pattern.match(line)
        if m:
            indent = m.group(1)
            bid = m.group(2)
            role = m.group(3)
            name_single = m.group(4)  # from 'name'
            name_double = m.group(5)  # from "name"
            rest = m.group(6).strip()

            name = name_single if name_single is not None else (name_double or "")

            # Map role to HTML tag
            tag = ROLE_TO_TAG.get(role, role)

            # Build attributes string from any extra info in rest
            attrs = ""
            if role in ("textbox", "searchbox") and rest:
                attrs = f' type="text" placeholder="{rest.strip()}"'
            elif role == "checkbox":
                if "checked" in rest.lower():
                    attrs = ' type="checkbox" checked'
                else:
                    attrs = ' type="checkbox"'
            elif role == "radio":
                attrs = ' type="radio"'
            elif role == "slider":
                attrs = ' type="range"'
            elif role == "spinbutton":
                attrs = ' type="number"'
            elif role == "heading":
                # Try to detect heading level from rest or default to h2
                level_match = re.search(r"level\s*(\d)", rest, re.IGNORECASE)
                if level_match:
                    tag = f"h{level_match.group(1)}"

            # Assign numeric index and record mapping
            index_to_bid[current_index] = bid

            # Format: [N]<tag attrs /> Name
            if name:
                output_lines.append(f"[{current_index}]<{tag}{attrs} /> {name}")
            else:
                output_lines.append(f"[{current_index}]<{tag}{attrs} />")

            current_index += 1
        else:
            # Non-bid lines (plain text, structural markers, etc.)
            stripped = line.strip()
            if stripped:
                output_lines.append(stripped)

    return "\n".join(output_lines), index_to_bid


def _build_bu_messages(
    obs: dict,
    history_items: list[dict],
    step_num: int,
    max_steps: int,
    task: str,
    use_thinking: bool,
    bu_dom_text: str,
) -> Discussion:
    """Build browser-use format messages for the LLM.

    Returns a Discussion with system + user messages matching browser-use's
    native <agent_history>, <agent_state>, <browser_state> format.
    """
    # System prompt – matches the browser_use library's template for BU models
    sys_prompt = _load_bu_system_prompt(use_thinking)

    # Build agent_history section
    history_text = ""
    for item in history_items:
        history_text += "<step>\n"
        if item.get("evaluation"):
            history_text += f"Eval: {item['evaluation']}\n"
        if item.get("memory"):
            history_text += f"Memory: {item['memory']}\n"
        if item.get("goal"):
            history_text += f"Next goal: {item['goal']}\n"
        if item.get("result"):
            history_text += f"Result: {item['result']}\n"
        history_text += "</step>\n"

    # Build agent_state section
    date_str = datetime.now().strftime("%Y-%m-%d")
    agent_state = f"""<user_request>
{task}
</user_request>
<step_info>Step {step_num + 1} maximum:{max_steps}
Today:{date_str}</step_info>"""

    # Build browser_state section
    url = obs.get("url", "")
    page_title = obs.get("page_title", "")

    # Tab info
    open_pages = obs.get("open_pages_urls", [])
    open_titles = obs.get("open_pages_titles", [])
    active_idx = obs.get("active_page_index", 0)
    tabs_text = ""
    for i, (u, t) in enumerate(zip(open_pages, open_titles)):
        active = " (active)" if i == active_idx else ""
        tabs_text += f"Tab {i}{active}\n  URL: {u}\n  Title: {t}\n"

    if not tabs_text:
        tabs_text = f"Tab 0 (active)\n  URL: {url}\n  Title: {page_title}\n"

    # Scroll info
    scroll_position = obs.get("scroll_position", None)
    scroll_text = ""
    if scroll_position is not None:
        x, y = scroll_position.get("x", 0), scroll_position.get("y", 0)
        if y > 0:
            scroll_text = f"\nScroll position: {x}px right, {y}px down"

    # Build browser_state content
    browser_state = f"""Current tab: Tab {active_idx} (active)
  URL: {url}
  Title: {page_title}
{scroll_text}
Interactive elements:
{bu_dom_text}"""

    # Assemble user message
    user_content = f"""<agent_history>
{history_text}</agent_history>

<agent_state>
{agent_state}
</agent_state>

<browser_state>
{browser_state}
</browser_state>"""

    messages = Discussion([SystemMessage(sys_prompt)])
    messages.append(HumanMessage(user_content))

    return messages


def _parse_bu_response(
    text: str, index_to_bid: dict[int, str]
) -> dict[str, Any]:
    """Parse browser-use model JSON response and convert actions to BrowserGym format.

    Handles:
    - <think>...</think> tags (stripped)
    - JSON extraction from potentially malformed responses
    - Action conversion from BU numeric indices to BrowserGym bids

    Returns dict with keys: action, think, evaluation, memory, goal
    Raises ParseError if response cannot be parsed.
    """
    # Strip <think>...</think> blocks
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Extract JSON from response
    data = None

    # Try to find a JSON object in the response
    # Look for the outermost { ... } block
    brace_depth = 0
    json_start = None
    json_end = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and json_start is not None:
                json_end = i + 1
                break

    if json_start is not None and json_end is not None:
        json_str = cleaned[json_start:json_end]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common issues: trailing commas, etc.
            fixed = re.sub(r",\s*}", "}", json_str)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                pass

    if data is None:
        # Regex fallback for completely malformed JSON
        raise ParseError(
            f"Could not parse JSON from response. Please respond with valid JSON.\n"
            f"Your response: {cleaned[:500]}"
        )

    # Extract fields
    evaluation = data.get("evaluation_previous_goal", "")
    memory = data.get("memory", "")
    goal = data.get("next_goal", "")
    thinking = data.get("thinking", think_text or "")
    actions = data.get("action", [])

    if not actions:
        raise ParseError(
            "The 'action' list is empty. You must provide at least one action."
        )

    # Convert the first action to BrowserGym format
    if isinstance(actions, list):
        action_entry = actions[0]
    else:
        action_entry = actions

    action_str = _convert_bu_action_to_bgym(action_entry, index_to_bid, data)

    if action_str is None:
        raise ParseError(
            f"Could not convert action to BrowserGym format: {action_entry}. "
            f"Available actions: click, input_text, scroll_down, scroll_up, "
            f"go_to_url, go_back, send_keys, select_option, done"
        )

    return {
        "action": action_str,
        "think": thinking or think_text,
        "evaluation": evaluation,
        "memory": memory,
        "goal": goal,
    }


def _convert_bu_action_to_bgym(
    action_entry: dict, index_to_bid: dict[int, str], full_data: dict = None
) -> str | None:
    """Convert a single browser-use action dict to a BrowserGym action string.

    Maps numeric indices from the BU DOM format back to BrowserGym bids.
    """
    if not isinstance(action_entry, dict):
        return None

    # action_entry is like {"click": {"index": 5}} or {"scroll_down": {}}
    for action_name, params in action_entry.items():
        if not isinstance(params, dict):
            params = {}

        if action_name == "click":
            idx = params.get("index")
            if idx is not None:
                bid = index_to_bid.get(int(idx), str(idx))
                return f'click("{bid}")'

        elif action_name == "input_text":
            idx = params.get("index")
            text_val = params.get("text", "")
            if idx is not None:
                bid = index_to_bid.get(int(idx), str(idx))
                # Escape quotes in text
                text_val = text_val.replace("\\", "\\\\").replace('"', '\\"')
                return f'fill("{bid}", "{text_val}")'

        elif action_name == "scroll_down":
            return "scroll(0, 300)"

        elif action_name == "scroll_up":
            return "scroll(0, -300)"

        elif action_name == "go_to_url":
            url = params.get("url", "")
            return f'goto("{url}")'

        elif action_name == "go_back":
            return "go_back()"

        elif action_name == "send_keys":
            keys = params.get("keys", "")
            return f'press("{keys}")'

        elif action_name == "select_option":
            idx = params.get("index")
            text_val = params.get("text", params.get("option", ""))
            if idx is not None:
                bid = index_to_bid.get(int(idx), str(idx))
                return f'select_option("{bid}", "{text_val}")'

        elif action_name == "done":
            text_val = params.get("text", "Task completed.")
            return f'send_msg_to_user("{text_val}")'

        elif action_name == "hover":
            idx = params.get("index")
            if idx is not None:
                bid = index_to_bid.get(int(idx), str(idx))
                return f'hover("{bid}")'

        elif action_name == "new_tab":
            return "new_tab()"

        elif action_name == "close_tab":
            return "tab_close()"

        elif action_name == "tab_focus":
            tab_idx = params.get("index", 0)
            return f"tab_focus({tab_idx})"

    return None


# ---------------------------------------------------------------------------
# Agent Args & Agent
# ---------------------------------------------------------------------------


@dataclass
class BrowserUseAgentArgs(AgentArgs):
    """Configuration for the BrowserUseAgent.

    Uses browser-use native prompt format instead of GenericAgent's AXTree + <action> tags.
    """

    agent_name: str = "BrowserUseAgent"
    chat_model_args: BaseModelArgs = None
    use_thinking: bool = True
    use_screenshot: bool = False
    max_retry: int = 4

    def __post_init__(self):
        try:
            self.agent_name = f"BrowserUseAgent-{self.chat_model_args.model_name}".replace(
                "/", "_"
            )
        except AttributeError:
            pass

    def set_benchmark(self, benchmark, demo_mode):
        pass

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return BrowserUseAgent(
            chat_model_args=self.chat_model_args,
            use_thinking=self.use_thinking,
            use_screenshot=self.use_screenshot,
            max_retry=self.max_retry,
        )


class BrowserUseAgent(bgym.Agent):
    """Agent that uses browser-use's native prompt format.

    Converts BrowserGym observations (AXTree) to browser-use's DOM format,
    builds messages in browser-use's XML structure, and parses the model's
    JSON output back to BrowserGym actions.
    """

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        use_thinking: bool = True,
        use_screenshot: bool = False,
        max_retry: int = 4,
    ):
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.use_thinking = use_thinking
        self.use_screenshot = use_screenshot
        self.max_retry = max_retry

        # Use standard BrowserGym high-level actions with bid addressing
        self.action_set = bgym.HighLevelActionSet(["bid"], multiaction=False)

        # Observation preprocessor: we need axtree_txt
        obs_flags = dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=False,
            use_error_logs=True,
            use_history=False,
            use_past_error_logs=False,
            use_action_history=False,
            use_think_history=False,
            use_diff=False,
            use_screenshot=use_screenshot,
            use_som=False,
            extract_visible_tag=False,
            extract_clickable_tag=False,
            extract_coords="False",
            filter_visible_elements_only=False,
        )
        self._obs_preprocessor = dp.make_obs_preprocessor(obs_flags)

        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def _get_max_input_tokens(self) -> int | None:
        """Derive max input tokens from chat_model_args, matching GenericAgent logic."""
        args = self.chat_model_args
        candidates = [getattr(args, "max_input_tokens", None)]
        if getattr(args, "max_input_tokens", None) is None and getattr(args, "max_total_tokens", None) is not None:
            max_new = getattr(args, "max_new_tokens", None) or 0
            candidates.append(args.max_total_tokens - max_new)
        candidates = [c for c in candidates if c is not None]
        return min(candidates) if candidates else None

    def _truncate_dom(
        self,
        bu_dom_text: str,
        index_to_bid: dict[int, str],
        obs: dict,
        step_num: int,
        max_steps: int,
        task: str,
    ) -> tuple[str, dict[int, str]]:
        """Truncate DOM text to fit within the model's token budget."""
        max_input_tokens = self._get_max_input_tokens()
        if max_input_tokens is None:
            return bu_dom_text, index_to_bid

        # Build messages with a placeholder to measure overhead
        placeholder = "DOM_PLACEHOLDER"
        test_msgs = _build_bu_messages(
            obs=obs,
            history_items=self.history_items,
            step_num=step_num,
            max_steps=max_steps,
            task=task,
            use_thinking=self.use_thinking,
            bu_dom_text=placeholder,
        )
        overhead_text = str(test_msgs)
        model_name = self.chat_model_args.model_name
        overhead_tokens = count_tokens(overhead_text, model=model_name)
        # Budget for DOM text. Use a large margin (2000 tokens) because
        # count_tokens uses a generic tokenizer that may differ from the model's
        # actual tokenizer (e.g. GPT-4 tiktoken vs Qwen tokenizer).
        dom_budget = max_input_tokens - overhead_tokens - 2000

        if dom_budget <= 0:
            logger.warning("Token budget exhausted by message overhead alone")
            return "", {}

        dom_tokens = count_tokens(bu_dom_text, model=model_name)
        if dom_tokens <= dom_budget:
            return bu_dom_text, index_to_bid

        # Truncate DOM lines from the bottom
        lines = bu_dom_text.split("\n")
        # Binary search for the right number of lines
        lo, hi = 0, len(lines)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = "\n".join(lines[:mid])
            if count_tokens(candidate, model=model_name) <= dom_budget:
                lo = mid
            else:
                hi = mid - 1

        truncated_text = "\n".join(lines[:lo])
        truncated_text += f"\n... [{len(lines) - lo} more elements truncated to fit context window]"
        logger.info(f"Truncated DOM from {len(lines)} to {lo} lines ({dom_tokens} -> ~{dom_budget} tokens)")

        # Rebuild index_to_bid mapping for remaining lines only
        # Parse the max index present in the truncated text
        max_idx = 0
        for line in lines[:lo]:
            m = re.match(r"\[(\d+)\]", line)
            if m:
                max_idx = int(m.group(1))
        truncated_mapping = {k: v for k, v in index_to_bid.items() if k <= max_idx}

        return truncated_text, truncated_mapping

    def reset(self, seed=None):
        self.seed = seed
        self.history_items = []
        self._last_index_to_bid = {}

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, dict]:
        task = obs.get("goal", "")
        max_steps = obs.get("goal_object", {}).get("max_steps", 30) if isinstance(
            obs.get("goal_object"), dict
        ) else 30
        step_num = len(self.history_items)

        # Convert AXTree to browser-use DOM format
        axtree_txt = obs.get("axtree_txt", "")
        bu_dom_text, index_to_bid = _convert_axtree_to_bu_dom(axtree_txt)
        self._last_index_to_bid = index_to_bid

        # Truncate DOM text to fit within token budget
        bu_dom_text, index_to_bid = self._truncate_dom(bu_dom_text, index_to_bid, obs, step_num, max_steps, task)
        self._last_index_to_bid = index_to_bid

        # Build messages in browser-use format
        chat_messages = _build_bu_messages(
            obs=obs,
            history_items=self.history_items,
            step_num=step_num,
            max_steps=max_steps,
            task=task,
            use_thinking=self.use_thinking,
            bu_dom_text=bu_dom_text,
        )

        # Add screenshot if available and enabled
        if self.use_screenshot and obs.get("screenshot") is not None:
            from agentlab.llm.llm_utils import image_to_jpg_base64_url

            chat_messages.last_message.add_image(obs["screenshot"])

        # Create parser closure with the index_to_bid mapping
        def parser(response_text: str) -> dict:
            return _parse_bu_response(response_text, index_to_bid)

        try:
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=parser,
            )
            n_retry = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = {"action": None, "think": None}
            n_retry = self.max_retry + 1

        # Update history with this step's info
        action_str = ans_dict.get("action")
        self.history_items.append(
            {
                "evaluation": ans_dict.get("evaluation", ""),
                "memory": ans_dict.get("memory", ""),
                "goal": ans_dict.get("goal", ""),
                "result": f"executed action: {action_str}" if action_str else "action failed",
            }
        )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = n_retry

        agent_info = AgentInfo(
            think=ans_dict.get("think"),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )

        return action_str, agent_info
