import datetime
import json
import logging
import os
import time
import re
import uuid
import base64
from io import BytesIO
from PIL import Image
from wrapt_timeout_decorator import *
from lib_results_logger import log_task_completion

logger = logging.getLogger("desktopenv.experiment")


def _resize_screenshot_for_claude(screenshot_b64: str, target_size: tuple = (1280, 720)) -> str:
    """
    Resize a base64 encoded screenshot to Claude's expected size.
    
    Args:
        screenshot_b64: Base64 encoded screenshot string
        target_size: Target size (width, height), defaults to Claude's 1280x720
        
    Returns:
        Resized screenshot as base64 string
    """
    try:
        # Decode base64 to bytes
        screenshot_bytes = base64.b64decode(screenshot_b64)
        
        # Open as PIL Image
        img = Image.open(BytesIO(screenshot_bytes))
        
        # Resize
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert back to base64
        output_buffer = BytesIO()
        resized_img.save(output_buffer, format='PNG')
        return base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to resize screenshot: {e}, returning original")
        return screenshot_b64


def _extract_reasoning_from_evocua_response(response: str) -> str:
    """
    Extract only the reasoning/thinking content from an EvoCUA response,
    removing EvoCUA-specific formatting like </think> tags and <tool_call> blocks.
    
    This prevents Claude from seeing and mimicking EvoCUA's output format.
    
    Args:
        response: Raw EvoCUA response string
        
    Returns:
        Cleaned reasoning text suitable for Claude's message history
    """
    if not response:
        return ""
    
    reasoning = response
    
    # Remove <tool_call> blocks and their contents
    reasoning = re.sub(r'<tool_call>.*?</tool_call>', '', reasoning, flags=re.DOTALL)
    reasoning = re.sub(r'<tool_call>\s*\{.*?\}\s*', '', reasoning, flags=re.DOTALL)
    
    # Remove </think> tags (EvoCUA uses these to separate thinking from action)
    reasoning = reasoning.replace('</think>', '')
    reasoning = reasoning.replace('<think>', '')
    
    # Remove "Action:" lines that describe what was done
    reasoning = re.sub(r'\nAction:.*$', '', reasoning, flags=re.MULTILINE)
    
    # Remove any remaining JSON-like tool call patterns
    reasoning = re.sub(r'\{"name":\s*"computer_use".*?\}', '', reasoning, flags=re.DOTALL)
    
    # Clean up excessive whitespace
    reasoning = re.sub(r'\n{3,}', '\n\n', reasoning)
    reasoning = reasoning.strip()
    
    return reasoning


def _convert_evocua_action_to_claude_input(action: str) -> dict:
    """
    Convert an EvoCUA action string (pyautogui code) to Claude's computer tool input format.
    
    Args:
        action: EvoCUA action string (e.g., "pyautogui.click(100, 200)")
        
    Returns:
        Dictionary in Claude's computer tool input format
    """
    action_lower = action.lower() if action else ""
    
    # Default to screenshot action
    input_dict = {"action": "screenshot"}
    
    try:
        # Parse pyautogui commands
        if "pyautogui.click(" in action_lower:
            input_dict = {"action": "left_click"}
            # Extract coordinates
            coords = re.findall(r'click\s*\(\s*(\d+)\s*,\s*(\d+)', action_lower)
            if coords:
                # Scale from original resolution to Claude's 1280x720
                # EvoCUA typically uses actual screen resolution, Claude uses 1280x720
                x, y = int(coords[0][0]), int(coords[0][1])
                # Assume 1920x1080 -> 1280x720 scaling
                input_dict["coordinate"] = [int(x * 1280 / 1920), int(y * 720 / 1080)]
        
        elif "pyautogui.rightclick(" in action_lower:
            input_dict = {"action": "right_click"}
            coords = re.findall(r'rightclick\s*\(\s*(\d+)\s*,\s*(\d+)', action_lower)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                input_dict["coordinate"] = [int(x * 1280 / 1920), int(y * 720 / 1080)]
        
        elif "pyautogui.doubleclick(" in action_lower:
            input_dict = {"action": "double_click"}
            coords = re.findall(r'doubleclick\s*\(\s*(\d+)\s*,\s*(\d+)', action_lower)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                input_dict["coordinate"] = [int(x * 1280 / 1920), int(y * 720 / 1080)]
        
        elif "pyautogui.press(" in action_lower:
            # Extract key from press('key')
            key_match = re.search(r"press\s*\(\s*['\"]([^'\"]+)['\"]", action_lower)
            if key_match:
                key = key_match.group(1)
                input_dict = {"action": "type", "text": key}
            else:
                input_dict = {"action": "key", "text": "enter"}
        
        elif "pyautogui.hotkey(" in action_lower:
            # Extract keys from hotkey('key1', 'key2')
            keys = re.findall(r"['\"]([^'\"]+)['\"]", action)
            if keys:
                input_dict = {"action": "key", "text": "+".join(keys)}
            else:
                input_dict = {"action": "key", "text": "enter"}
        
        elif "pyautogui.keydown(" in action_lower or "pyautogui.keyup(" in action_lower:
            key_match = re.search(r"key(?:down|up)\s*\(\s*['\"]([^'\"]+)['\"]", action_lower)
            if key_match:
                input_dict = {"action": "key", "text": key_match.group(1)}
        
        elif "pyautogui.scroll(" in action_lower:
            # Extract scroll amount
            scroll_match = re.search(r'scroll\s*\(\s*(-?\d+)', action_lower)
            if scroll_match:
                scroll_amount = int(scroll_match.group(1))
                direction = "up" if scroll_amount > 0 else "down"
                input_dict = {
                    "action": "scroll",
                    "scroll_direction": direction,
                    "scroll_amount": abs(scroll_amount)
                }
            else:
                input_dict = {"action": "scroll", "scroll_direction": "down", "scroll_amount": 3}
        
        elif "pyautogui.moveto(" in action_lower:
            input_dict = {"action": "mouse_move"}
            coords = re.findall(r'moveto\s*\(\s*(\d+)\s*,\s*(\d+)', action_lower)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                input_dict["coordinate"] = [int(x * 1280 / 1920), int(y * 720 / 1080)]
        
        elif "pyautogui.dragto(" in action_lower:
            input_dict = {"action": "left_click_drag"}
            coords = re.findall(r'dragto\s*\(\s*(\d+)\s*,\s*(\d+)', action_lower)
            if coords:
                x, y = int(coords[0][0]), int(coords[0][1])
                input_dict["coordinate"] = [int(x * 1280 / 1920), int(y * 720 / 1080)]
        
        elif action.upper() in ("DONE", "FAIL", "WAIT"):
            input_dict = {"action": action.lower()}
        
    except Exception as e:
        logger.warning(f"Failed to parse action '{action}': {e}")
    
    return input_dict


def inject_evocua_history_to_claude(
    claude_agent,
    evocua_agent,
    instruction: str,
    current_obs: dict = None,
    max_history_steps: int = 30,
    screen_size: tuple = (1920, 1080),
):
    """
    Inject EvoCUA's step history into Claude agent's message history.
    
    This function converts EvoCUA's action history into Claude's expected message format,
    allowing Claude to understand what has been attempted before and continue from there.
    
    Args:
        claude_agent: The Claude/Anthropic agent instance
        evocua_agent: The EvoCUA agent instance with history
        instruction: The original task instruction
        current_obs: Current observation dict containing the latest screenshot (after last EvoCUA action)
        max_history_steps: Maximum number of recent steps to inject (to avoid context overflow)
        screen_size: Original screen size for coordinate scaling
    """
    # Get EvoCUA history
    screenshots = evocua_agent.screenshots  # base64 encoded screenshots
    actions = evocua_agent.actions  # low-level instructions (descriptions)
    responses = evocua_agent.responses  # raw LLM responses
    
    if not screenshots or not actions:
        logger.warning("No EvoCUA history to inject into Claude")
        return
    
    # Limit history to avoid context overflow
    total_steps = min(len(actions), max_history_steps)
    start_idx = max(0, len(actions) - max_history_steps)
    
    logger.info(f"Injecting {total_steps} EvoCUA steps into Claude history (starting from step {start_idx + 1})")
    
    messages = []
    
    # First message: instruction + first screenshot (from the history window)
    first_screenshot_idx = start_idx
    if first_screenshot_idx < len(screenshots):
        first_screenshot_b64 = _resize_screenshot_for_claude(screenshots[first_screenshot_idx])
        
        # Build context message explaining the handoff
        context_text = f"""Task: {instruction}"""

# [CONTEXT: You are continuing from a previous agent that got stuck. Below is the history of what was attempted. The agent may stuck due to grounding error, or could not find the correct logic to complete the task. Please complete the task from where it left off.]
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": first_screenshot_b64,
                    },
                },
                {"type": "text", "text": context_text},
            ]
        })
    
    # For each step in the history window, create assistant (with tool_use) and user (with tool_result) messages
    for i in range(start_idx, start_idx + total_steps):
        rel_idx = i - start_idx  # Relative index within the history window
        
        action_desc = actions[i] if i < len(actions) else ""
        response = responses[i] if i < len(responses) else ""
        
        # Generate a unique tool use ID
        tool_use_id = f"toolu_{uuid.uuid4().hex[:24]}"
        
        # Parse the actual pyautogui action from the response or action
        # EvoCUA's response contains the action, we need to extract the pyautogui code
        pyautogui_action = ""
        if response:
            # Look for pyautogui code in the response
            if "pyautogui." in response:
                # Extract pyautogui lines
                lines = response.split('\n')
                for line in lines:
                    if 'pyautogui.' in line:
                        pyautogui_action = line.strip()
                        break
            elif "```" in response:
                # Try to extract from code blocks
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
                for block in code_blocks:
                    if 'pyautogui.' in block:
                        lines = block.strip().split('\n')
                        for line in lines:
                            if 'pyautogui.' in line:
                                pyautogui_action = line.strip()
                                break
                        if pyautogui_action:
                            break
        
        # If we couldn't extract from response, use the action description
        if not pyautogui_action:
            pyautogui_action = action_desc
        
        # Convert to Claude's tool input format
        tool_input = _convert_evocua_action_to_claude_input(pyautogui_action)
        
        # Create assistant message with reasoning and tool_use
        assistant_content = []
        
        # Add cleaned reasoning text from response (strip EvoCUA-specific formatting)
        if response:
            # Extract only the reasoning, removing EvoCUA's </think> and <tool_call> formatting
            cleaned_reasoning = _extract_reasoning_from_evocua_response(response)
            # Truncate long responses
            if len(cleaned_reasoning) > 800:
                cleaned_reasoning = cleaned_reasoning[:800] + "..."
            if cleaned_reasoning:
                assistant_content.append({
                    "type": "text", 
                    "text": f"Step {i + 1} reasoning: {cleaned_reasoning}"
                })
            else:
                # If no reasoning extracted, use action description
                assistant_content.append({
                    "type": "text",
                    "text": f"Step {i + 1}: {action_desc}"
                })
        else:
            assistant_content.append({
                "type": "text",
                "text": f"Step {i + 1}: {action_desc}"
            })
        
        assistant_content.append({
            "type": "tool_use",
            "id": tool_use_id,
            "name": "computer",
            "input": tool_input
        })
        
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        # User message with tool_result + next screenshot
        tool_result_content = [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": f"Previous step {i + 1} completed: {action_desc}"}]
        }]
        
        # Add the next screenshot
        # For the last step, use the current observation's screenshot (shows current state)
        # For earlier steps, use the next screenshot from EvoCUA's history
        is_last_step = (i == start_idx + total_steps - 1)
        next_screenshot_b64 = None
        
        if is_last_step and current_obs and "screenshot" in current_obs:
            # Use the current screenshot (after the last EvoCUA action)
            current_screenshot_bytes = current_obs["screenshot"]
            current_screenshot_b64 = base64.b64encode(current_screenshot_bytes).decode('utf-8')
            next_screenshot_b64 = _resize_screenshot_for_claude(current_screenshot_b64)
            logger.info("Using current observation screenshot for last step")
        else:
            # Use the next screenshot from EvoCUA's history
            next_screenshot_idx = i + 1
            if next_screenshot_idx < len(screenshots):
                next_screenshot_b64 = _resize_screenshot_for_claude(screenshots[next_screenshot_idx])
        
        if next_screenshot_b64:
            tool_result_content[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": next_screenshot_b64
                }
            })
        
        messages.append({
            "role": "user",
            "content": tool_result_content
        })
    
    # Set Claude's message history
    claude_agent.messages = messages
    logger.info(f"Successfully injected {len(messages)} messages into Claude's history (including current screenshot: {current_obs is not None})")


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)

    # Reset environment first to get fresh VM IP
    env.reset(task_config=example)

    # Reset agent with fresh VM IP (for snapshot reverts)
    
    # try:
    #     agent.reset(runtime_logger, vm_ip=env.vm_ip)
    # except Exception as e:
    #     agent.reset(vm_ip=env.vm_ip)

    # For Qwen3VLAgent, only logger is needed
    try: 
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()
    
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    time.sleep(20) # Wait for the environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    
    # Log task completion to results.json
    log_task_completion(example, result, example_result_dir, args)
    
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger

def run_single_example_human(env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    
    # Save initial screenshot
    with open(os.path.join(example_result_dir, "initial_state.png"), "wb") as _f:
        _f.write(obs['screenshot'])
    
    # Save trajectory information
    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
        f.write(json.dumps({
            "instruction": instruction,
            "initial_state": "initial_state.png"
        }))
        f.write("\n")
    
    # Evaluate the result
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")



def run_single_example_kimi(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0

    step_token_data = []
    kimi_token_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_request_time": 0.0,
        "num_requests": 0,
    }

    env.controller.start_recording()
    task_start_time = time.time()

    while not done and step_idx < max_steps:
        response, actions, info_dict = agent.predict(instruction, obs)

        logger.info(f"Got Action: {actions}")

        step_info = getattr(agent, 'last_step_info', {})
        if step_info:
            step_token_record = {
                "step": step_idx + 1,
                "request_time": step_info.get("request_time", 0),
                "input_tokens": step_info.get("input_tokens", 0),
                "output_tokens": step_info.get("output_tokens", 0),
                "total_tokens": step_info.get("total_tokens", 0),
            }
            step_token_data.append(step_token_record)
            kimi_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
            kimi_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
            kimi_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
            kimi_token_totals["total_request_time"] += step_info.get("request_time", 0)
            kimi_token_totals["num_requests"] += 1

        if not actions or len(actions)==0 or actions[0]=="" or actions[0].lower().startswith("error"): 
            break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info(f"Action {action} executed, reward: {reward}, done: {done}")
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            log_entry = {
                "step_num": step_idx + 1,
                "action": action,
                "natural_language_action": info_dict.get("action"),
                "action_timestamp": action_timestamp,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
            }
            if step_token_data and step_token_data[-1]["step"] == step_idx + 1:
                log_entry["token_info"] = step_token_data[-1]

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    task_end_time = time.time()
    time.sleep(30) # Wait for the environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

    avg_request_time = (
        kimi_token_totals["total_request_time"] / kimi_token_totals["num_requests"]
        if kimi_token_totals["num_requests"] > 0 else 0
    )
    summary = {
        "result": result,
        "total_steps": step_idx,
        "task_wall_time": task_end_time - task_start_time,
        "token_usage": {
            "total_input_tokens": kimi_token_totals["input_tokens"],
            "total_output_tokens": kimi_token_totals["output_tokens"],
            "total_tokens": kimi_token_totals["total_tokens"],
            "total_request_time": kimi_token_totals["total_request_time"],
            "avg_request_time": avg_request_time,
            "total_requests": kimi_token_totals["num_requests"],
        },
        "per_step": step_token_data,
    }
    with open(os.path.join(example_result_dir, "kimi_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def run_single_example_agi(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )

        done = not response.get('state_correct', False)

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info, step_info = agent.step(action)

            if not done:
                if not response.get('state_correct', False):
                    done = True

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            # Remove pending checks if they exist which will cause issues with json serialization
            if action.get('pending_checks', None):
                del action['pending_checks']

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def run_single_example_openaicua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )

        done = not response.get('state_correct', False)

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info, step_info = agent.step(action)

            if not done:
                if not response.get('state_correct', False):
                    done = True

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            # Remove pending checks if they exist which will cause issues with json serialization
            if action.get('pending_checks', None):
                del action['pending_checks']

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_opencua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions, info_dict = agent.predict(instruction, obs)

        logger.info(f"Got Action: {actions}")
        # Breack if no actions
        if not actions or len(actions)==0 or actions[0]=="" or actions[0].lower().startswith("error"): 
            break

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info(f"Action {action} executed, reward: {reward}, done: {done}")
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action": action,
                    "natural_language_action": info_dict.get("action"),
                    "action_timestamp": action_timestamp,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }, ensure_ascii=False))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time.sleep(20) # Wait for the environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_autoglm(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)
    
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
                
            if done:
                logger.info("The episode is done.")
                break
        
        # Invalid Action
        if not actions:
            obs = env._get_obs() # update observation
            
        step_idx += 1
    
    if not done: # not completed the task yet
        env.action_history.append('FAIL')
    
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_mano(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    
    with open(os.path.join(example_result_dir, f"step_0.png"),
      "wb") as _f:
        _f.write(obs['screenshot'])
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        if len(actions) > 1:
            if (("pyautogui.hotkey('shift')" in actions[0] or "pyautogui.hotkey('ctrl')" in actions[0]) 
                and "pyautogui.click" in actions[1]):
                hotkey_type = 'shift' if "shift" in actions[0] else 'ctrl'
                action = f"pyautogui.keyDown('{hotkey_type}')\n{actions[1]}\npyautogui.keyUp('{hotkey_type}')"
                actions = [action]  
                
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                    "response":response
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
    
def run_single_example_uipath(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)

    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs,
            args,
            step_idx
        )
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


from mm_agents.os_symphony.utils.common_utils import draw_coordinates
from mm_agents.os_symphony.utils.process_context import set_current_result_dir


logger = logging.getLogger("desktopenv.experiment")

def run_single_example_os_symphony(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    set_current_result_dir(example_result_dir)
    
    agent.reset(result_dir=example_result_dir)
    env.reset(task_config=example)
    time.sleep(30) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    # env.controller.start_recording()
    start_time = time.time()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs,
            step_idx == max_steps - 1
        )
        for action in actions:
            # Save screenshot and trajectory information
            if "reflection" in response and response["reflection"].get("is_milestone"):
                img_name = f"step_{step_idx + 1}_milestone.png"
            else:
                img_name = f"step_{step_idx + 1}.png"
                
            with open(os.path.join(example_result_dir, img_name),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            if "coordinates" in response and response["coordinates"]:
                draw_coordinates(
                    image_bytes=obs['screenshot'], 
                    coordinates=response["coordinates"], 
                    save_path=os.path.join(example_result_dir, img_name[:-4] + "_draw.png")
                )

            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            logger.info("Done: %s", done)

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "instruction": instruction,
                    "step_num": step_idx + 1,
                    "action": action,
                    "response": response,
                    "done": done,
                    "info": info,
                    "screenshot_file": img_name
                }))
                f.write("\n")
            with open(os.path.join(example_result_dir, f"traj_{step_idx+1}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "step_num": step_idx + 1,
                    "action": action,
                    "response": response,
                    "done": done,
                    "info": info,
                    "screenshot_file": img_name
                }, f, indent=4, ensure_ascii=False)
            if done:
                logger.info("The episode is done.")
                time.sleep(60)
                break
        step_idx += 1
    end_time = time.time()
    result = float(env.evaluate())
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    with open(os.path.join(example_result_dir, "time.txt"), "w", encoding="utf-8") as f:
        f.write(f"{end_time-start_time:.2f}\n")


def run_single_example_evocua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    """
    Unified run function for EvoCUAAgent (supporting both S1 and S2 modes).
    """
    runtime_logger = setup_logger(example, example_result_dir)
    
    # Reset Environment
    env.reset(task_config=example)
    
    # Reset Agent
    # Handle agent reset signature differences if any
    try:
        agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        try:
            agent.reset(runtime_logger)
        except Exception:
            agent.reset()

    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0

    env.controller.start_recording()
    while not done and step_idx < max_steps:
        # EvoCUAAgent.predict unified signature: returns (response, actions)
        # It handles both modes internally.
        predict_res = agent.predict(instruction, obs)
        
        # Check return signature logic
        if len(predict_res) == 3:
            # Compatibility with S1 original signature if agent was updated to match
            response, actions, info_dict = predict_res
        else:
            response, actions = predict_res
            info_dict = {}

        logger.info(f"Step {step_idx + 1} Actions: {actions}")
        
        # Break if no actions (fail-safe)
        if not actions or (len(actions) == 1 and (actions[0] == "" or "error" in actions[0].lower())):
             # Allow "FAIL" or "DONE" to process through execution loop if agent outputs them as actions
             if not (actions and actions[0] in ["FAIL", "DONE"]):
                 logger.warning("No valid actions returned. Breaking loop.")
                 break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Executing action: %s", action)
            
            # Execute
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            
            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            
            # Save screenshot
            screenshot_file = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_file), "wb") as _f:
                _f.write(obs['screenshot'])
            
            # Log Trajectory
            log_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": screenshot_file
            }
            # Add natural language info if available (S1 style)
            if info_dict:
                log_entry["natural_language_action"] = info_dict.get("action")
            
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")
                
            if done:
                logger.info("The episode is done.")
                break
        
        step_idx += 1
        
    time.sleep(20) # Wait for environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    
    log_task_completion(example, result, example_result_dir, args)

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def run_single_example_hybrid(
    evocua_agent,
    claude_agent,
    stuck_detector,
    env,
    example,
    max_steps,
    instruction,
    args,
    example_result_dir,
    scores,
    milestone_detector=None,
    milestone_judge=None,
):
    """
    Hybrid run function that uses EvoCUA first and switches to Claude when stuck or milestone fails.
    
    This function:
    1. Starts execution with EvoCUA agent
    2. After each step, checks if the agent is stuck using the BERT stuck detector
    3. After each step, checks if the step is a milestone using the BERT milestone detector
    4. If milestone detected, sends to Claude for judgment:
       - If milestone failed, switches to Claude to continue execution
    5. If stuck is detected, switches to Claude to continue execution
    
    Args:
        evocua_agent: The EvoCUA agent instance
        claude_agent: The Claude/Anthropic agent instance
        stuck_detector: StuckDetector instance for detecting stuck loops
        env: The desktop environment
        example: The task example configuration
        max_steps: Maximum number of steps allowed
        instruction: The task instruction
        args: Command line arguments
        example_result_dir: Directory to save results
        scores: Shared list to append results
        milestone_detector: Optional MilestoneDetector instance for detecting milestones
        milestone_judge: Optional MilestoneJudge instance for judging milestones
    """
    from mm_agents.milestone_detector import MilestoneTracker
    
    runtime_logger = setup_logger(example, example_result_dir)
    
    # Reset Environment
    env.reset(task_config=example)
    
    # Reset Agents
    try:
        evocua_agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        try:
            evocua_agent.reset(runtime_logger)
        except Exception:
            evocua_agent.reset()
    
    try:
        claude_agent.reset(runtime_logger)
    except Exception:
        claude_agent.reset()

    time.sleep(60)  # Wait for the environment to be ready
    obs = env._get_obs()  # Get the initial observation
    done = False
    step_idx = 0
    
    # Track which agent is currently active
    current_agent = "evocua"
    switched_at_step = None
    switch_reason = None  # 'stuck' or 'milestone_failed'
    
    # History tracking for stuck detection
    step_responses = []
    step_actions = []
    
    # Initialize milestone tracker
    milestone_tracker = MilestoneTracker()
    
    # Set initial screenshot for milestone comparison
    if obs and 'screenshot' in obs:
        initial_screenshot_b64 = base64.b64encode(obs['screenshot']).decode('utf-8')
        milestone_tracker.set_initial_screenshot(initial_screenshot_b64)
        logger.info("Initial screenshot captured for milestone tracking")
    
    # Additional metadata for logging
    stuck_detections = []
    milestone_detections = []
    
    # Token and timing tracking (per-step and cumulative, separated by agent)
    step_token_data = []  # Per-step records
    evocua_token_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_request_time": 0.0,
        "num_requests": 0,
    }
    claude_token_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_request_time": 0.0,
        "num_requests": 0,
    }

    env.controller.start_recording()
    
    while not done and step_idx < max_steps:
        if current_agent == "evocua":
            # Use EvoCUA agent
            predict_res = evocua_agent.predict(instruction, obs)
            
            if len(predict_res) == 3:
                response, actions, info_dict = predict_res
            else:
                response, actions = predict_res
                info_dict = {}
            
            logger.info(f"[EvoCUA] Step {step_idx + 1} Actions: {actions}")
            
            # Capture EvoCUA token/timing info
            step_info = getattr(evocua_agent, 'last_step_info', {})
            if step_info:
                step_token_record = {
                    "step": step_idx + 1,
                    "agent": "evocua",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                }
                step_token_data.append(step_token_record)
                evocua_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                evocua_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                evocua_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                evocua_token_totals["total_request_time"] += step_info.get("request_time", 0)
                evocua_token_totals["num_requests"] += 1
            
            # Handle empty/error actions
            if not actions or (len(actions) == 1 and (actions[0] == "" or "error" in str(actions[0]).lower())):
                if not (actions and actions[0] in ["FAIL", "DONE"]):
                    logger.warning("[EvoCUA] No valid actions returned. Breaking loop.")
                    break
            
            # Track history for stuck detection
            response_str = response if isinstance(response, str) else json.dumps(response)
            action_str = str(actions[0]) if actions else ""
            step_responses.append(response_str)
            step_actions.append(action_str)
            
            # Add to milestone tracker
            milestone_tracker.add_step(step_idx + 1, response_str)
            
        else:
            # Use Claude agent
            response, actions = claude_agent.predict(instruction, obs)
            
            logger.info(f"[Claude] Step {step_idx + 1} Actions: {actions}")
            
            # Capture Claude token/timing info
            step_info = getattr(claude_agent, 'last_step_info', {})
            if step_info:
                step_token_record = {
                    "step": step_idx + 1,
                    "agent": "claude",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                    "cache_creation_input_tokens": step_info.get("cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": step_info.get("cache_read_input_tokens", 0),
                }
                step_token_data.append(step_token_record)
                claude_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                claude_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                claude_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                claude_token_totals["cache_creation_input_tokens"] += step_info.get("cache_creation_input_tokens", 0)
                claude_token_totals["cache_read_input_tokens"] += step_info.get("cache_read_input_tokens", 0)
                claude_token_totals["total_request_time"] += step_info.get("request_time", 0)
                claude_token_totals["num_requests"] += 1
            
            if actions is None:
                actions = []
            
            # Track history (for logging, not stuck detection after switch)
            response_str = str(response) if response else ""
            action_str = ""
            if actions:
                first_action = actions[0]
                if isinstance(first_action, dict):
                    action_str = first_action.get("command", str(first_action))
                else:
                    action_str = str(first_action)
            step_responses.append(response_str)
            step_actions.append(action_str)
        
        # Execute actions (matching original run_single_example pattern)
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info(f"[{current_agent.upper()}] Step {step_idx + 1}: {action}")
            
            # Execute - pass action directly to env.step() like the original
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            
            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            
            # Save screenshot
            screenshot_file = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_file), "wb") as _f:
                _f.write(obs['screenshot'])
            
            # Log Trajectory
            log_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "response": response_str,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": screenshot_file,
                "current_agent": current_agent,
                "switched_at_step": switched_at_step,
                "switch_reason": switch_reason,
            }
            
            # Add token/timing info for this step
            if step_token_data and step_token_data[-1]["step"] == step_idx + 1:
                log_entry["token_info"] = step_token_data[-1]
            
            if info_dict and current_agent == "evocua":
                log_entry["natural_language_action"] = info_dict.get("action")
            
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")
            
            if done:
                logger.info("The episode is done.")
                break
        
        # Check for stuck/milestone conditions AFTER executing the action (only if still using EvoCUA)
        should_switch = False
        switch_to_claude_reason = None
        
        if not done and current_agent == "evocua" and step_idx >= 1:
            # 1. Check for stuck condition
            is_stuck, stuck_prob, formatted_history = stuck_detector.check_if_stuck(
                step_responses=step_responses,
                step_actions=step_actions,
                current_step=step_idx + 1,
                max_history_steps=6,
            )
            
            if is_stuck:
                logger.warning(f"STUCK DETECTED at step {step_idx + 1}! Probability: {stuck_prob:.4f}")
                should_switch = True
                switch_to_claude_reason = "stuck"
                
                stuck_detections.append({
                    "step": step_idx + 1,
                    "probability": stuck_prob,
                    "history_summary": formatted_history[:500] + "..." if len(formatted_history) > 500 else formatted_history,
                })
            
            # 2. Check for milestone condition (only if not already switching due to stuck)
            if not should_switch and milestone_detector is not None:
                is_milestone, milestone_prob, formatted_text = milestone_detector.check_if_milestone(
                    task_description=instruction,
                    step_responses=step_responses,
                    current_step=step_idx + 1,
                )
                
                if is_milestone:
                    logger.info(f"MILESTONE DETECTED at step {step_idx + 1}! Probability: {milestone_prob:.4f}")
                    
                    # Get the current screenshot as base64
                    current_screenshot_b64 = base64.b64encode(obs['screenshot']).decode('utf-8')
                    
                    # Get the previous milestone screenshot (or initial screenshot)
                    previous_milestone_screenshot_b64 = milestone_tracker.get_previous_milestone_screenshot()
                    
                    # Get reasoning since last milestone
                    reasoning_history = milestone_tracker.get_reasoning_since_last_milestone()
                    
                    # Judge the milestone using Claude
                    if milestone_judge is not None:
                        judgment = milestone_judge.judge_milestone(
                            task_description=instruction,
                            reasoning_history=reasoning_history,
                            current_screenshot_b64=current_screenshot_b64,
                            previous_milestone_screenshot_b64=previous_milestone_screenshot_b64,
                        )
                        
                        logger.info(f"Milestone judgment: success={judgment.get('success')}, "
                                   f"milestone={judgment.get('inferred_milestone', 'Unknown')}")
                        
                        # Record the milestone (with screenshot for future comparisons)
                        milestone_tracker.record_milestone(
                            step_num=step_idx + 1,
                            milestone_prob=milestone_prob,
                            judgment=judgment,
                            screenshot_file=screenshot_file,
                            screenshot_b64=current_screenshot_b64,
                        )
                        
                        milestone_detections.append({
                            "step": step_idx + 1,
                            "probability": milestone_prob,
                            "inferred_milestone": judgment.get("inferred_milestone", "Unknown"),
                            "success": judgment.get("success", True),
                            "reasoning": judgment.get("reasoning", ""),
                        })
                        
                        # If milestone failed, switch to Claude
                        if not judgment.get("success", True):
                            logger.warning(f"MILESTONE FAILED at step {step_idx + 1}!")
                            logger.info(f"Reason: {judgment.get('reasoning', 'No reason provided')}")
                            should_switch = True
                            switch_to_claude_reason = "milestone_failed"
                    else:
                        # No judge available, just record the milestone detection
                        milestone_tracker.record_milestone(
                            step_num=step_idx + 1,
                            milestone_prob=milestone_prob,
                            judgment={"success": True, "inferred_milestone": "Unknown", "reasoning": "No judge available"},
                            screenshot_file=screenshot_file,
                            screenshot_b64=current_screenshot_b64,
                        )
                        
                        milestone_detections.append({
                            "step": step_idx + 1,
                            "probability": milestone_prob,
                            "inferred_milestone": "Unknown",
                            "success": True,
                            "reasoning": "No milestone judge available",
                        })
            
            # 3. Switch to Claude if needed
            if should_switch:
                logger.info(f"Switching from EvoCUA to Claude (reason: {switch_to_claude_reason})...")
                
                # Record the switch
                switched_at_step = step_idx + 1
                switch_reason = switch_to_claude_reason
                current_agent = "claude"
                
                # Format context for Claude based on switch reason
                if switch_to_claude_reason == "stuck":
                    handoff_context = stuck_detector.format_context_for_claude(
                        instruction=instruction,
                        step_responses=step_responses,
                        step_actions=step_actions,
                        stuck_reason=f"Stuck probability: {stuck_prob:.4f}",
                    )
                else:  # milestone_failed
                    last_milestone = milestone_tracker.get_last_milestone()
                    handoff_context = f"""TASK HANDOFF - MILESTONE FAILED
                    
Task: {instruction}

Failed Milestone: {last_milestone.get('inferred_milestone', 'Unknown') if last_milestone else 'Unknown'}
Reason: {last_milestone.get('judgment_reasoning', 'Unknown') if last_milestone else 'Unknown'}

The previous agent attempted a milestone but it was not successful.
Please continue from here and complete the task correctly.
"""
                
                # Log the handoff
                with open(os.path.join(example_result_dir, "handoff.txt"), "w", encoding="utf-8") as f:
                    f.write(handoff_context)
                
                logger.info(f"Handoff context prepared. Continuing with Claude agent.")
                
                # Reset Claude agent and inject EvoCUA's history into Claude's message format
                try:
                    claude_agent.reset(runtime_logger)
                except Exception:
                    claude_agent.reset()
                
                # Inject EvoCUA's step history into Claude's message history
                # This allows Claude to understand what has been attempted and continue from there
                current_screen_size = (args.screen_width, args.screen_height) if hasattr(args, 'screen_width') else (1920, 1080)
                inject_evocua_history_to_claude(
                    claude_agent=claude_agent,
                    evocua_agent=evocua_agent,
                    instruction=instruction,
                    current_obs=obs,  # Pass current observation with latest screenshot
                    max_history_steps=10,  # Limit to avoid context overflow
                    screen_size=current_screen_size,
                )
                
                logger.info(f"Injected EvoCUA history ({len(evocua_agent.actions)} steps) into Claude's message history")
                
                # Note: We don't modify the instruction anymore since the context is now in the message history
                # The inject function already adds context about the handoff
        
        step_idx += 1
    
    time.sleep(20)  # Wait for environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    
    # Save result
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    
    # Save hybrid run summary
    hybrid_summary = {
        "total_steps": step_idx,
        "final_agent": current_agent,
        "switched_at_step": switched_at_step,
        "switch_reason": switch_reason,
        "stuck_detections": stuck_detections,
        "milestone_detections": milestone_detections,
        "all_milestones": milestone_tracker.get_all_milestones(),
        "result": result,
        "token_usage": {
            "evocua": evocua_token_totals,
            "claude": claude_token_totals,
            "combined": {
                "total_input_tokens": evocua_token_totals["input_tokens"] + claude_token_totals["input_tokens"],
                "total_output_tokens": evocua_token_totals["output_tokens"] + claude_token_totals["output_tokens"],
                "total_tokens": evocua_token_totals["total_tokens"] + claude_token_totals["total_tokens"],
                "total_request_time": evocua_token_totals["total_request_time"] + claude_token_totals["total_request_time"],
                "total_requests": evocua_token_totals["num_requests"] + claude_token_totals["num_requests"],
            },
            "per_step": step_token_data,
        },
    }
    with open(os.path.join(example_result_dir, "hybrid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(hybrid_summary, f, indent=2)
    
    log_task_completion(example, result, example_result_dir, args)
    
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def inject_kimi_history_to_evocua(
    evocua_agent,
    kimi_agent,
    instruction: str,
    current_obs: dict = None,
):
    """
    Append Kimi's step history to EvoCUA's internal state so that EvoCUA
    can see what happened during a Kimi burst when it resumes.

    EvoCUA stores history in parallel lists:
      - self.screenshots  (list of base64-encoded image strings)
      - self.actions       (list of low-level instruction strings)
      - self.responses     (list of raw LLM response strings)
      - self.observations  (list of obs dicts)
      - self.cots          (list of dicts, used by S1 style)

    We convert Kimi's recorded history into EvoCUA's format and append.
    """
    from mm_agents.evocua.utils import encode_image, process_image

    kimi_observations = kimi_agent.observations
    kimi_actions = kimi_agent.actions
    kimi_cots = kimi_agent.cots

    if not kimi_actions:
        logger.warning("No Kimi history to inject into EvoCUA")
        return

    # inject_evocua_history_to_kimi pre-populates Kimi with EvoCUA history
    # and records how many entries it injected.  The Kimi-own steps start
    # after that prefix.
    injected_prefix_len = getattr(kimi_agent, '_injected_prefix_len', 0)
    start_idx = injected_prefix_len
    if start_idx >= len(kimi_actions):
        logger.info("No new Kimi steps beyond pre-injected EvoCUA history to inject")
        return

    steps_to_inject = len(kimi_actions) - start_idx
    logger.info(
        f"Injecting {steps_to_inject} Kimi steps into EvoCUA history "
        f"(Kimi indices {start_idx}..{len(kimi_actions) - 1})"
    )

    for i in range(start_idx, len(kimi_actions)):
        # Screenshot: Kimi stores raw bytes in observations; convert to EvoCUA's b64 format
        screenshot_bytes = b""
        if i < len(kimi_observations) and kimi_observations[i]:
            screenshot_bytes = kimi_observations[i].get("screenshot", b"")

        if screenshot_bytes:
            if evocua_agent.prompt_style == "S2":
                b64_img, _, _ = process_image(screenshot_bytes, factor=evocua_agent.resize_factor)
            else:
                b64_img = encode_image(screenshot_bytes)
        else:
            b64_img = ""

        evocua_agent.screenshots.append(b64_img)

        action_desc = kimi_actions[i] if i < len(kimi_actions) else ""
        evocua_agent.actions.append(action_desc)

        cot = kimi_cots[i] if i < len(kimi_cots) else {}
        thought = cot.get("thought", "")
        response_text = f"[Kimi step] {thought}" if thought else f"[Kimi step] {action_desc}"
        evocua_agent.responses.append(response_text)

        if i < len(kimi_observations) and kimi_observations[i]:
            evocua_agent.observations.append(kimi_observations[i])
        else:
            evocua_agent.observations.append({})

        evocua_agent.cots.append({"thought": thought, "action": action_desc})

    logger.info(
        f"Successfully injected {steps_to_inject} Kimi steps into EvoCUA's history "
        f"(EvoCUA now has {len(evocua_agent.actions)} total steps)"
    )


def inject_evocua_history_to_kimi(
    kimi_agent,
    evocua_agent,
    instruction: str,
    current_obs: dict = None,
    max_history_steps: int = 30,
):
    """
    Inject EvoCUA's step history into KimiAgent's internal state so that
    Kimi's predict() method automatically includes the previous steps.

    KimiAgent stores history in three parallel lists:
      - self.observations  (list of obs dicts with 'screenshot' bytes)
      - self.actions        (list of action description strings)
      - self.cots           (list of dicts with at least 'thought' and 'action')

    We populate these from EvoCUA's recorded history.
    """
    screenshots = evocua_agent.screenshots   # base64 encoded
    actions = evocua_agent.actions            # low-level instruction strings
    responses = evocua_agent.responses        # raw LLM responses

    if not screenshots or not actions:
        logger.warning("No EvoCUA history to inject into Kimi")
        return

    total_steps = min(len(actions), max_history_steps)
    start_idx = max(0, len(actions) - max_history_steps)

    logger.info(
        f"Injecting {total_steps} EvoCUA steps into Kimi history "
        f"(starting from step {start_idx + 1})"
    )

    kimi_agent.observations = []
    kimi_agent.actions = []
    kimi_agent.cots = []

    for i in range(start_idx, start_idx + total_steps):
        # Build obs dict that KimiAgent expects ('screenshot' as raw bytes)
        if i < len(screenshots):
            screenshot_bytes = base64.b64decode(screenshots[i])
        else:
            screenshot_bytes = b""
        kimi_agent.observations.append({"screenshot": screenshot_bytes})

        action_desc = actions[i] if i < len(actions) else ""
        kimi_agent.actions.append(action_desc)

        response_text = responses[i] if i < len(responses) else ""
        thought = _extract_reasoning_from_evocua_response(response_text)
        kimi_agent.cots.append({"thought": thought, "action": action_desc})

    kimi_agent._injected_prefix_len = len(kimi_agent.actions)
    logger.info(
        f"Successfully injected {len(kimi_agent.actions)} steps into Kimi's history"
    )


def run_single_example_hybrid_kimi(
    evocua_agent,
    kimi_agent,
    stuck_detector,
    env,
    example,
    max_steps,
    instruction,
    args,
    example_result_dir,
    scores,
    milestone_detector=None,
    milestone_judge=None,
):
    """
    Hybrid run: EvoCUA base model + Kimi K2.5 advanced model.

    Mirrors run_single_example_hybrid() but switches to KimiAgent instead of Claude.
    """
    from mm_agents.milestone_detector import MilestoneTracker

    runtime_logger = setup_logger(example, example_result_dir)

    env.reset(task_config=example)

    try:
        evocua_agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        try:
            evocua_agent.reset(runtime_logger)
        except Exception:
            evocua_agent.reset()

    try:
        kimi_agent.reset(runtime_logger)
    except Exception:
        kimi_agent.reset()

    time.sleep(60)
    obs = env._get_obs()
    done = False
    step_idx = 0

    current_agent = "evocua"
    switched_at_step = None
    switch_reason = None

    step_responses = []
    step_actions = []

    milestone_tracker = MilestoneTracker()

    if obs and "screenshot" in obs:
        initial_screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
        milestone_tracker.set_initial_screenshot(initial_screenshot_b64)
        logger.info("Initial screenshot captured for milestone tracking")

    stuck_detections = []
    milestone_detections = []

    step_token_data = []
    evocua_token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_request_time": 0.0, "num_requests": 0,
    }
    kimi_token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_request_time": 0.0, "num_requests": 0,
    }

    env.controller.start_recording()

    while not done and step_idx < max_steps:
        info_dict = {}

        if current_agent == "evocua":
            predict_res = evocua_agent.predict(instruction, obs)

            if len(predict_res) == 3:
                response, actions_list, info_dict = predict_res
            else:
                response, actions_list = predict_res

            logger.info(f"[EvoCUA] Step {step_idx + 1} Actions: {actions_list}")

            step_info = getattr(evocua_agent, "last_step_info", {})
            if step_info:
                step_token_data.append({
                    "step": step_idx + 1, "agent": "evocua",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                })
                evocua_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                evocua_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                evocua_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                evocua_token_totals["total_request_time"] += step_info.get("request_time", 0)
                evocua_token_totals["num_requests"] += 1

            if not actions_list or (
                len(actions_list) == 1
                and (actions_list[0] == "" or "error" in str(actions_list[0]).lower())
            ):
                if not (actions_list and actions_list[0] in ["FAIL", "DONE"]):
                    logger.warning("[EvoCUA] No valid actions returned. Breaking loop.")
                    break

            response_str = response if isinstance(response, str) else json.dumps(response)
            action_str = str(actions_list[0]) if actions_list else ""
            step_responses.append(response_str)
            step_actions.append(action_str)
            milestone_tracker.add_step(step_idx + 1, response_str)

        else:
            # Kimi agent
            response, actions_list, info_dict = kimi_agent.predict(instruction, obs)

            logger.info(f"[Kimi] Step {step_idx + 1} Actions: {actions_list}")

            if (
                not actions_list
                or len(actions_list) == 0
                or actions_list[0] == ""
                or actions_list[0].lower().startswith("error")
            ):
                break

            response_str = str(response) if response else ""
            action_str = str(actions_list[0]) if actions_list else ""
            step_responses.append(response_str)
            step_actions.append(action_str)

        # Execute actions
        for action in actions_list:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info(f"[{current_agent.upper()}] Step {step_idx + 1}: {action}")

            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            screenshot_file = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_file), "wb") as _f:
                _f.write(obs["screenshot"])

            log_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "response": response_str,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": screenshot_file,
                "current_agent": current_agent,
                "switched_at_step": switched_at_step,
                "switch_reason": switch_reason,
            }

            if step_token_data and step_token_data[-1]["step"] == step_idx + 1:
                log_entry["token_info"] = step_token_data[-1]

            if info_dict and current_agent == "evocua":
                log_entry["natural_language_action"] = info_dict.get("action")

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")

            if done:
                logger.info("The episode is done.")
                break

        # ---- Stuck / milestone detection (only while using EvoCUA) ----
        should_switch = False
        switch_to_kimi_reason = None

        if not done and current_agent == "evocua" and step_idx >= 1:
            # 1. Check stuck
            is_stuck, stuck_prob, formatted_history = stuck_detector.check_if_stuck(
                step_responses=step_responses,
                step_actions=step_actions,
                current_step=step_idx + 1,
                max_history_steps=6,
            )

            if is_stuck:
                logger.warning(
                    f"STUCK DETECTED at step {step_idx + 1}! Probability: {stuck_prob:.4f}"
                )
                should_switch = True
                switch_to_kimi_reason = "stuck"
                stuck_detections.append({
                    "step": step_idx + 1,
                    "probability": stuck_prob,
                    "history_summary": (
                        formatted_history[:500] + "..."
                        if len(formatted_history) > 500
                        else formatted_history
                    ),
                })

            # 2. Check milestone
            if not should_switch and milestone_detector is not None:
                is_milestone, milestone_prob, formatted_text = milestone_detector.check_if_milestone(
                    task_description=instruction,
                    step_responses=step_responses,
                    current_step=step_idx + 1,
                )

                if is_milestone:
                    logger.info(
                        f"MILESTONE DETECTED at step {step_idx + 1}! "
                        f"Probability: {milestone_prob:.4f}"
                    )

                    current_screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
                    previous_milestone_screenshot_b64 = milestone_tracker.get_previous_milestone_screenshot()
                    reasoning_history = milestone_tracker.get_reasoning_since_last_milestone()

                    if milestone_judge is not None:
                        judgment = milestone_judge.judge_milestone(
                            task_description=instruction,
                            reasoning_history=reasoning_history,
                            current_screenshot_b64=current_screenshot_b64,
                            previous_milestone_screenshot_b64=previous_milestone_screenshot_b64,
                        )

                        logger.info(
                            f"Milestone judgment: success={judgment.get('success')}, "
                            f"milestone={judgment.get('inferred_milestone', 'Unknown')}"
                        )

                        milestone_tracker.record_milestone(
                            step_num=step_idx + 1,
                            milestone_prob=milestone_prob,
                            judgment=judgment,
                            screenshot_file=screenshot_file,
                            screenshot_b64=current_screenshot_b64,
                        )

                        milestone_detections.append({
                            "step": step_idx + 1,
                            "probability": milestone_prob,
                            "inferred_milestone": judgment.get("inferred_milestone", "Unknown"),
                            "success": judgment.get("success", True),
                            "reasoning": judgment.get("reasoning", ""),
                        })

                        if not judgment.get("success", True):
                            logger.warning(f"MILESTONE FAILED at step {step_idx + 1}!")
                            logger.info(f"Reason: {judgment.get('reasoning', 'No reason provided')}")
                            should_switch = True
                            switch_to_kimi_reason = "milestone_failed"
                    else:
                        milestone_tracker.record_milestone(
                            step_num=step_idx + 1,
                            milestone_prob=milestone_prob,
                            judgment={
                                "success": True,
                                "inferred_milestone": "Unknown",
                                "reasoning": "No judge available",
                            },
                            screenshot_file=screenshot_file,
                            screenshot_b64=current_screenshot_b64,
                        )
                        milestone_detections.append({
                            "step": step_idx + 1,
                            "probability": milestone_prob,
                            "inferred_milestone": "Unknown",
                            "success": True,
                            "reasoning": "No milestone judge available",
                        })

            # 3. Switch to Kimi if needed
            if should_switch:
                logger.info(
                    f"Switching from EvoCUA to Kimi (reason: {switch_to_kimi_reason})..."
                )

                switched_at_step = step_idx + 1
                switch_reason = switch_to_kimi_reason
                current_agent = "kimi"

                if switch_to_kimi_reason == "stuck":
                    handoff_context = stuck_detector.format_context_for_claude(
                        instruction=instruction,
                        step_responses=step_responses,
                        step_actions=step_actions,
                        stuck_reason=f"Stuck probability: {stuck_prob:.4f}",
                    )
                else:
                    last_milestone = milestone_tracker.get_last_milestone()
                    handoff_context = (
                        f"TASK HANDOFF - MILESTONE FAILED\n\n"
                        f"Task: {instruction}\n\n"
                        f"Failed Milestone: "
                        f"{last_milestone.get('inferred_milestone', 'Unknown') if last_milestone else 'Unknown'}\n"
                        f"Reason: "
                        f"{last_milestone.get('judgment_reasoning', 'Unknown') if last_milestone else 'Unknown'}\n\n"
                        f"The previous agent attempted a milestone but it was not successful.\n"
                        f"Please continue from here and complete the task correctly.\n"
                    )

                with open(os.path.join(example_result_dir, "handoff.txt"), "w", encoding="utf-8") as f:
                    f.write(handoff_context)

                logger.info("Handoff context prepared. Continuing with Kimi agent.")

                try:
                    kimi_agent.reset(runtime_logger)
                except Exception:
                    kimi_agent.reset()

                inject_evocua_history_to_kimi(
                    kimi_agent=kimi_agent,
                    evocua_agent=evocua_agent,
                    instruction=instruction,
                    current_obs=obs,
                    max_history_steps=10,
                )

                logger.info(
                    f"Injected EvoCUA history ({len(evocua_agent.actions)} steps) "
                    f"into Kimi's internal state"
                )

        step_idx += 1

    time.sleep(30)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    hybrid_summary = {
        "total_steps": step_idx,
        "final_agent": current_agent,
        "switched_at_step": switched_at_step,
        "switch_reason": switch_reason,
        "stuck_detections": stuck_detections,
        "milestone_detections": milestone_detections,
        "all_milestones": milestone_tracker.get_all_milestones(),
        "result": result,
        "token_usage": {
            "evocua": evocua_token_totals,
            "kimi": kimi_token_totals,
            "combined": {
                "total_input_tokens": evocua_token_totals["input_tokens"] + kimi_token_totals["input_tokens"],
                "total_output_tokens": evocua_token_totals["output_tokens"] + kimi_token_totals["output_tokens"],
                "total_tokens": evocua_token_totals["total_tokens"] + kimi_token_totals["total_tokens"],
                "total_request_time": evocua_token_totals["total_request_time"] + kimi_token_totals["total_request_time"],
                "total_requests": evocua_token_totals["num_requests"] + kimi_token_totals["num_requests"],
            },
            "per_step": step_token_data,
        },
    }
    with open(os.path.join(example_result_dir, "hybrid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(hybrid_summary, f, indent=2)

    log_task_completion(example, result, example_result_dir, args)

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def run_single_example_hybrid_kimi_bounce(
    evocua_agent,
    kimi_agent,
    stuck_detector,
    env,
    example,
    max_steps,
    instruction,
    args,
    example_result_dir,
    scores,
    milestone_detector=None,
    milestone_judge=None,
    kimi_steps_per_burst=5,
):
    """
    Hybrid run with bounce-back: EvoCUA + Kimi K2.5.

    Like run_single_example_hybrid_kimi, but when a switch to Kimi is triggered
    (stuck detection or milestone failure), Kimi runs for only `kimi_steps_per_burst`
    steps and then control returns to EvoCUA.  Subsequent stuck/milestone triggers
    can switch to Kimi again, creating multiple short Kimi bursts within a single
    episode.
    """
    from mm_agents.milestone_detector import MilestoneTracker

    runtime_logger = setup_logger(example, example_result_dir)

    env.reset(task_config=example)

    try:
        evocua_agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        try:
            evocua_agent.reset(runtime_logger)
        except Exception:
            evocua_agent.reset()

    try:
        kimi_agent.reset(runtime_logger)
    except Exception:
        kimi_agent.reset()

    time.sleep(60)
    obs = env._get_obs()
    done = False
    step_idx = 0

    current_agent = "evocua"
    kimi_steps_remaining = 0

    # Histories used by the stuck detector (reset after each bounce-back)
    step_responses: list = []
    step_actions: list = []

    milestone_tracker = MilestoneTracker()

    if obs and "screenshot" in obs:
        initial_screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
        milestone_tracker.set_initial_screenshot(initial_screenshot_b64)
        logger.info("Initial screenshot captured for milestone tracking")

    stuck_detections: list = []
    milestone_detections: list = []

    switch_events: list = []
    kimi_bursts: list = []
    current_burst: dict | None = None

    step_token_data: list = []
    evocua_token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_request_time": 0.0, "num_requests": 0,
    }
    kimi_token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_request_time": 0.0, "num_requests": 0,
    }

    env.controller.start_recording()
    task_start_time = time.time()

    while not done and step_idx < max_steps:
        info_dict = {}

        if current_agent == "evocua":
            predict_res = evocua_agent.predict(instruction, obs)

            if len(predict_res) == 3:
                response, actions_list, info_dict = predict_res
            else:
                response, actions_list = predict_res

            logger.info(f"[EvoCUA] Step {step_idx + 1} Actions: {actions_list}")

            step_info = getattr(evocua_agent, "last_step_info", {})
            if step_info:
                step_token_data.append({
                    "step": step_idx + 1, "agent": "evocua",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                })
                evocua_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                evocua_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                evocua_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                evocua_token_totals["total_request_time"] += step_info.get("request_time", 0)
                evocua_token_totals["num_requests"] += 1

            if not actions_list or (
                len(actions_list) == 1
                and (actions_list[0] == "" or "error" in str(actions_list[0]).lower())
            ):
                if not (actions_list and actions_list[0] in ["FAIL", "DONE"]):
                    logger.warning("[EvoCUA] No valid actions returned. Breaking loop.")
                    break

            response_str = response if isinstance(response, str) else json.dumps(response)
            action_str = str(actions_list[0]) if actions_list else ""
            step_responses.append(response_str)
            step_actions.append(action_str)
            milestone_tracker.add_step(step_idx + 1, response_str)

        else:
            # Kimi agent
            response, actions_list, info_dict = kimi_agent.predict(instruction, obs)

            logger.info(f"[Kimi] Step {step_idx + 1} Actions: {actions_list}")

            step_info = getattr(kimi_agent, "last_step_info", {})
            if step_info:
                step_token_data.append({
                    "step": step_idx + 1, "agent": "kimi",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                })
                kimi_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                kimi_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                kimi_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                kimi_token_totals["total_request_time"] += step_info.get("request_time", 0)
                kimi_token_totals["num_requests"] += 1

            if (
                not actions_list
                or len(actions_list) == 0
                or actions_list[0] == ""
                or actions_list[0].lower().startswith("error")
            ):
                break

            response_str = str(response) if response else ""
            action_str = str(actions_list[0]) if actions_list else ""
            step_responses.append(response_str)
            step_actions.append(action_str)

        # Execute actions
        for action in actions_list:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info(f"[{current_agent.upper()}] Step {step_idx + 1}: {action}")

            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            screenshot_file = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_file), "wb") as _f:
                _f.write(obs["screenshot"])

            log_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "response": response_str,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": screenshot_file,
                "current_agent": current_agent,
                "kimi_steps_remaining": kimi_steps_remaining if current_agent == "kimi" else None,
            }

            if step_token_data and step_token_data[-1]["step"] == step_idx + 1:
                log_entry["token_info"] = step_token_data[-1]

            if info_dict and current_agent == "evocua":
                log_entry["natural_language_action"] = info_dict.get("action")

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")

            if done:
                logger.info("The episode is done.")
                break

        # ----- Post-step: detect triggers & manage bouncing -----

        if current_agent == "kimi" and not done:
            kimi_steps_remaining -= 1
            if current_burst is not None:
                current_burst["steps_run"] += 1
            logger.info(f"[Kimi] Steps remaining in burst: {kimi_steps_remaining}")

            if kimi_steps_remaining <= 0:
                logger.info("Kimi burst finished. Switching back to EvoCUA.")

                inject_kimi_history_to_evocua(
                    evocua_agent=evocua_agent,
                    kimi_agent=kimi_agent,
                    instruction=instruction,
                    current_obs=obs,
                )

                current_agent = "evocua"

                switch_events.append({
                    "step": step_idx + 1,
                    "direction": "kimi->evocua",
                    "reason": "burst_exhausted",
                })
                if current_burst is not None:
                    current_burst["end_step"] = step_idx + 1
                    kimi_bursts.append(current_burst)
                    current_burst = None

                step_responses.clear()
                step_actions.clear()

        elif current_agent == "evocua" and not done and step_idx >= 1:
            should_switch = False
            switch_to_kimi_reason = None

            is_stuck, stuck_prob, formatted_history = stuck_detector.check_if_stuck(
                step_responses=step_responses,
                step_actions=step_actions,
                current_step=step_idx + 1,
                max_history_steps=6,
            )

            if is_stuck:
                logger.warning(
                    f"STUCK DETECTED at step {step_idx + 1}! Probability: {stuck_prob:.4f}"
                )
                should_switch = True
                switch_to_kimi_reason = "stuck"
                stuck_detections.append({
                    "step": step_idx + 1,
                    "probability": stuck_prob,
                    "history_summary": (
                        formatted_history[:500] + "..."
                        if len(formatted_history) > 500
                        else formatted_history
                    ),
                })

            if not should_switch and milestone_detector is not None:
                is_milestone, milestone_prob, formatted_text = milestone_detector.check_if_milestone(
                    task_description=instruction,
                    step_responses=step_responses,
                    current_step=step_idx + 1,
                )

                if is_milestone:
                    logger.info(
                        f"MILESTONE DETECTED at step {step_idx + 1}! "
                        f"Probability: {milestone_prob:.4f}"
                    )

                    current_screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
                    previous_milestone_screenshot_b64 = milestone_tracker.get_previous_milestone_screenshot()
                    reasoning_history = milestone_tracker.get_reasoning_since_last_milestone()

                    if milestone_judge is not None:
                        judgment = milestone_judge.judge_milestone(
                            task_description=instruction,
                            reasoning_history=reasoning_history,
                            current_screenshot_b64=current_screenshot_b64,
                            previous_milestone_screenshot_b64=previous_milestone_screenshot_b64,
                        )

                        logger.info(
                            f"Milestone judgment: success={judgment.get('success')}, "
                            f"milestone={judgment.get('inferred_milestone', 'Unknown')}"
                        )

                        milestone_tracker.record_milestone(
                            step_num=step_idx + 1,
                            milestone_prob=milestone_prob,
                            judgment=judgment,
                            screenshot_file=screenshot_file,
                            screenshot_b64=current_screenshot_b64,
                        )

                        milestone_detections.append({
                            "step": step_idx + 1,
                            "probability": milestone_prob,
                            "inferred_milestone": judgment.get("inferred_milestone", "Unknown"),
                            "success": judgment.get("success", True),
                            "reasoning": judgment.get("reasoning", ""),
                        })

                        if not judgment.get("success", True):
                            logger.warning(f"MILESTONE FAILED at step {step_idx + 1}!")
                            logger.info(f"Reason: {judgment.get('reasoning', 'No reason provided')}")
                            should_switch = True
                            switch_to_kimi_reason = "milestone_failed"
                    else:
                        milestone_tracker.record_milestone(
                            step_num=step_idx + 1,
                            milestone_prob=milestone_prob,
                            judgment={
                                "success": True,
                                "inferred_milestone": "Unknown",
                                "reasoning": "No judge available",
                            },
                            screenshot_file=screenshot_file,
                            screenshot_b64=current_screenshot_b64,
                        )
                        milestone_detections.append({
                            "step": step_idx + 1,
                            "probability": milestone_prob,
                            "inferred_milestone": "Unknown",
                            "success": True,
                            "reasoning": "No milestone judge available",
                        })

            if should_switch:
                kimi_steps_remaining = kimi_steps_per_burst
                current_agent = "kimi"

                logger.info(
                    f"Switching from EvoCUA to Kimi for {kimi_steps_per_burst} steps "
                    f"(reason: {switch_to_kimi_reason})..."
                )

                switch_events.append({
                    "step": step_idx + 1,
                    "direction": "evocua->kimi",
                    "reason": switch_to_kimi_reason,
                })
                current_burst = {
                    "burst_index": len(kimi_bursts),
                    "trigger_step": step_idx + 1,
                    "trigger_reason": switch_to_kimi_reason,
                    "max_steps": kimi_steps_per_burst,
                    "steps_run": 0,
                    "end_step": None,
                }

                if switch_to_kimi_reason == "stuck":
                    handoff_context = stuck_detector.format_context_for_claude(
                        instruction=instruction,
                        step_responses=step_responses,
                        step_actions=step_actions,
                        stuck_reason=f"Stuck probability: {stuck_prob:.4f}",
                    )
                else:
                    last_milestone = milestone_tracker.get_last_milestone()
                    handoff_context = (
                        f"TASK HANDOFF - MILESTONE FAILED\n\n"
                        f"Task: {instruction}\n\n"
                        f"Failed Milestone: "
                        f"{last_milestone.get('inferred_milestone', 'Unknown') if last_milestone else 'Unknown'}\n"
                        f"Reason: "
                        f"{last_milestone.get('judgment_reasoning', 'Unknown') if last_milestone else 'Unknown'}\n\n"
                        f"The previous agent attempted a milestone but it was not successful.\n"
                        f"Please continue from here and complete the task correctly.\n"
                    )

                handoff_num = len(kimi_bursts) + 1
                with open(
                    os.path.join(example_result_dir, f"handoff_{handoff_num}.txt"),
                    "w", encoding="utf-8",
                ) as f:
                    f.write(handoff_context)

                logger.info("Handoff context prepared. Continuing with Kimi agent.")

                try:
                    kimi_agent.reset(runtime_logger)
                except Exception:
                    kimi_agent.reset()

                inject_evocua_history_to_kimi(
                    kimi_agent=kimi_agent,
                    evocua_agent=evocua_agent,
                    instruction=instruction,
                    current_obs=obs,
                    max_history_steps=10,
                )

                logger.info(
                    f"Injected EvoCUA history ({len(evocua_agent.actions)} steps) "
                    f"into Kimi's internal state"
                )

        step_idx += 1

    if current_burst is not None:
        current_burst["end_step"] = step_idx
        kimi_bursts.append(current_burst)

    task_end_time = time.time()
    time.sleep(30)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    hybrid_summary = {
        "total_steps": step_idx,
        "final_agent": current_agent,
        "kimi_steps_per_burst": kimi_steps_per_burst,
        "num_kimi_bursts": len(kimi_bursts),
        "total_kimi_steps": sum(b["steps_run"] for b in kimi_bursts),
        "total_evocua_steps": step_idx - sum(b["steps_run"] for b in kimi_bursts),
        "switch_events": switch_events,
        "kimi_bursts": kimi_bursts,
        "stuck_detections": stuck_detections,
        "milestone_detections": milestone_detections,
        "all_milestones": milestone_tracker.get_all_milestones(),
        "result": result,
        "task_wall_time": task_end_time - task_start_time,
        "token_usage": {
            "evocua": evocua_token_totals,
            "kimi": kimi_token_totals,
            "combined": {
                "total_input_tokens": evocua_token_totals["input_tokens"] + kimi_token_totals["input_tokens"],
                "total_output_tokens": evocua_token_totals["output_tokens"] + kimi_token_totals["output_tokens"],
                "total_tokens": evocua_token_totals["total_tokens"] + kimi_token_totals["total_tokens"],
                "total_request_time": evocua_token_totals["total_request_time"] + kimi_token_totals["total_request_time"],
                "total_requests": evocua_token_totals["num_requests"] + kimi_token_totals["num_requests"],
            },
            "per_step": step_token_data,
        },
    }
    with open(os.path.join(example_result_dir, "hybrid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(hybrid_summary, f, indent=2)

    log_task_completion(example, result, example_result_dir, args)

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


# ---------------------------------------------------------------------------
# Periodic Verification Baseline
# ---------------------------------------------------------------------------

def run_single_example_periodic_verify(
    evocua_agent,
    kimi_agent,
    verifier,
    env,
    example,
    max_steps,
    instruction,
    args,
    example_result_dir,
    scores,
    verify_every_n_steps=5,
):
    """
    Periodic verification baseline: EvoCUA + Kimi K2.5.

    Instead of using BERT stuck/milestone detectors, Claude verifies progress
    every N steps. If verification fails, switches to Kimi for the remainder.
    """
    from mm_agents.milestone_detector import MilestoneTracker

    runtime_logger = setup_logger(example, example_result_dir)

    env.reset(task_config=example)

    try:
        evocua_agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        try:
            evocua_agent.reset(runtime_logger)
        except Exception:
            evocua_agent.reset()

    try:
        kimi_agent.reset(runtime_logger)
    except Exception:
        kimi_agent.reset()

    time.sleep(60)
    obs = env._get_obs()
    done = False
    step_idx = 0

    current_agent = "evocua"
    switched_at_step = None
    switch_reason = None

    step_responses = []
    step_actions = []

    milestone_tracker = MilestoneTracker()

    initial_screenshot_b64 = None
    if obs and "screenshot" in obs:
        initial_screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
        milestone_tracker.set_initial_screenshot(initial_screenshot_b64)
        logger.info("Initial screenshot captured for periodic verification tracking")

    verification_results = []
    last_verified_screenshot_b64 = initial_screenshot_b64

    step_token_data = []
    evocua_token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_request_time": 0.0, "num_requests": 0,
    }
    kimi_token_totals = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "total_request_time": 0.0, "num_requests": 0,
    }

    env.controller.start_recording()

    while not done and step_idx < max_steps:
        info_dict = {}

        if current_agent == "evocua":
            predict_res = evocua_agent.predict(instruction, obs)

            if len(predict_res) == 3:
                response, actions_list, info_dict = predict_res
            else:
                response, actions_list = predict_res

            logger.info(f"[EvoCUA] Step {step_idx + 1} Actions: {actions_list}")

            step_info = getattr(evocua_agent, "last_step_info", {})
            if step_info:
                step_token_data.append({
                    "step": step_idx + 1, "agent": "evocua",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                })
                evocua_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                evocua_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                evocua_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                evocua_token_totals["total_request_time"] += step_info.get("request_time", 0)
                evocua_token_totals["num_requests"] += 1

            if not actions_list or (
                len(actions_list) == 1
                and (actions_list[0] == "" or "error" in str(actions_list[0]).lower())
            ):
                if not (actions_list and actions_list[0] in ["FAIL", "DONE"]):
                    logger.warning("[EvoCUA] No valid actions returned. Breaking loop.")
                    break

            response_str = response if isinstance(response, str) else json.dumps(response)
            action_str = str(actions_list[0]) if actions_list else ""
            step_responses.append(response_str)
            step_actions.append(action_str)
            milestone_tracker.add_step(step_idx + 1, response_str)

        else:
            response, actions_list, info_dict = kimi_agent.predict(instruction, obs)

            logger.info(f"[Kimi] Step {step_idx + 1} Actions: {actions_list}")

            step_info = getattr(kimi_agent, "last_step_info", {})
            if step_info:
                step_token_data.append({
                    "step": step_idx + 1, "agent": "kimi",
                    "request_time": step_info.get("request_time", 0),
                    "input_tokens": step_info.get("input_tokens", 0),
                    "output_tokens": step_info.get("output_tokens", 0),
                    "total_tokens": step_info.get("total_tokens", 0),
                })
                kimi_token_totals["input_tokens"] += step_info.get("input_tokens", 0)
                kimi_token_totals["output_tokens"] += step_info.get("output_tokens", 0)
                kimi_token_totals["total_tokens"] += step_info.get("total_tokens", 0)
                kimi_token_totals["total_request_time"] += step_info.get("request_time", 0)
                kimi_token_totals["num_requests"] += 1

            if (
                not actions_list
                or len(actions_list) == 0
                or actions_list[0] == ""
                or actions_list[0].lower().startswith("error")
            ):
                break

            response_str = str(response) if response else ""
            action_str = str(actions_list[0]) if actions_list else ""
            step_responses.append(response_str)
            step_actions.append(action_str)

        # Execute actions
        for action in actions_list:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info(f"[{current_agent.upper()}] Step {step_idx + 1}: {action}")

            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            screenshot_file = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_file), "wb") as _f:
                _f.write(obs["screenshot"])

            log_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "response": response_str,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": screenshot_file,
                "current_agent": current_agent,
                "switched_at_step": switched_at_step,
                "switch_reason": switch_reason,
            }

            if step_token_data and step_token_data[-1]["step"] == step_idx + 1:
                log_entry["token_info"] = step_token_data[-1]

            if info_dict and current_agent == "evocua":
                log_entry["natural_language_action"] = info_dict.get("action")

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")

            if done:
                logger.info("The episode is done.")
                break

        # ---- Periodic verification (only while using EvoCUA) ----
        if (
            not done
            and current_agent == "evocua"
            and (step_idx + 1) >= verify_every_n_steps
            and (step_idx + 1) % verify_every_n_steps == 0
            and verifier is not None
        ):
            logger.info(
                f"Running periodic verification at step {step_idx + 1} "
                f"(every {verify_every_n_steps} steps)"
            )

            current_screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
            reasoning_history = milestone_tracker.get_reasoning_since_last_milestone()

            judgment = verifier.judge_milestone(
                task_description=instruction,
                reasoning_history=reasoning_history,
                current_screenshot_b64=current_screenshot_b64,
                previous_milestone_screenshot_b64=last_verified_screenshot_b64,
            )

            logger.info(
                f"Periodic verification result: success={judgment.get('success')}, "
                f"reasoning={judgment.get('reasoning', '')[:200]}"
            )

            verification_results.append({
                "step": step_idx + 1,
                "inferred_milestone": judgment.get("inferred_milestone", "Unknown"),
                "success": judgment.get("success", True),
                "reasoning": judgment.get("reasoning", ""),
            })

            if judgment.get("success", True):
                last_verified_screenshot_b64 = current_screenshot_b64
                milestone_tracker.reasoning_since_last_milestone = []
                logger.info(
                    f"Verification passed at step {step_idx + 1}. Continuing with EvoCUA."
                )
            else:
                logger.warning(
                    f"VERIFICATION FAILED at step {step_idx + 1}! Switching to Kimi."
                )

                switched_at_step = step_idx + 1
                switch_reason = "periodic_verification_failed"
                current_agent = "kimi"

                handoff_context = (
                    f"TASK HANDOFF - PERIODIC VERIFICATION FAILED\n\n"
                    f"Task: {instruction}\n\n"
                    f"Verification at step {step_idx + 1} determined the agent is not "
                    f"making adequate progress.\n"
                    f"Reason: {judgment.get('reasoning', 'No reason provided')}\n\n"
                    f"Please continue from the current state and complete the task.\n"
                )

                with open(
                    os.path.join(example_result_dir, "handoff.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(handoff_context)

                logger.info("Handoff context prepared. Continuing with Kimi agent.")

                try:
                    kimi_agent.reset(runtime_logger)
                except Exception:
                    kimi_agent.reset()

                inject_evocua_history_to_kimi(
                    kimi_agent=kimi_agent,
                    evocua_agent=evocua_agent,
                    instruction=instruction,
                    current_obs=obs,
                    max_history_steps=10,
                )

                logger.info(
                    f"Injected EvoCUA history ({len(evocua_agent.actions)} steps) "
                    f"into Kimi's internal state"
                )

        step_idx += 1

    time.sleep(30)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    periodic_verify_summary = {
        "total_steps": step_idx,
        "final_agent": current_agent,
        "switched_at_step": switched_at_step,
        "switch_reason": switch_reason,
        "verify_every_n_steps": verify_every_n_steps,
        "verification_results": verification_results,
        "result": result,
        "token_usage": {
            "evocua": evocua_token_totals,
            "kimi": kimi_token_totals,
            "combined": {
                "total_input_tokens": (
                    evocua_token_totals["input_tokens"] + kimi_token_totals["input_tokens"]
                ),
                "total_output_tokens": (
                    evocua_token_totals["output_tokens"] + kimi_token_totals["output_tokens"]
                ),
                "total_tokens": (
                    evocua_token_totals["total_tokens"] + kimi_token_totals["total_tokens"]
                ),
                "total_request_time": (
                    evocua_token_totals["total_request_time"]
                    + kimi_token_totals["total_request_time"]
                ),
                "total_requests": (
                    evocua_token_totals["num_requests"] + kimi_token_totals["num_requests"]
                ),
            },
            "per_step": step_token_data,
        },
    }
    with open(os.path.join(example_result_dir, "hybrid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(periodic_verify_summary, f, indent=2)

    log_task_completion(example, result, example_result_dir, args)

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
