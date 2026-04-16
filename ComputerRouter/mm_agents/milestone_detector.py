"""
Milestone Detector module using ModernBERT for detecting milestone steps in GUI agent trajectories.

This module provides a MilestoneDetector class that loads a fine-tuned ModernBERT model
to detect whether a step is a milestone step based on task description and recent action history.

The input format follows the BERT training dataset format from build_training_dataset.py:
- Text: "Task: {task_description}\n" + previous 5 steps + current step
- Output: Binary classification (0 = non-milestone, 1 = milestone)
"""

import os
import json
import base64
import logging
from typing import List, Dict, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("desktopenv.milestone_detector")


class MilestoneDetector:
    """
    A milestone detector using ModernBERT for binary classification.
    
    Detects when a GUI agent reaches a milestone step based on
    task description and recent action history.
    """
    
    def __init__(
        self,
        model_path: str = "/gpfs/radev/scratch/cohan/jw3278/modernbert-milestone-detector",
        device: Optional[str] = None,
        max_length: int = 2048,
        milestone_threshold: float = 0.5,
        context_steps: int = 5,
    ):
        """
        Initialize the MilestoneDetector.
        
        Args:
            model_path: Path to the fine-tuned ModernBERT model
            device: Device to use for inference ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            milestone_threshold: Probability threshold above which to classify as "milestone"
            context_steps: Number of previous steps to include as context (default: 5)
        """
        self.model_path = model_path
        self.max_length = max_length
        self.milestone_threshold = milestone_threshold
        self.context_steps = context_steps
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"MilestoneDetector initializing with model: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the fine-tuned ModernBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("MilestoneDetector model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MilestoneDetector model: {e}")
            raise
    
    def format_step(self, step_num: int, response: str) -> str:
        """Format a single step for the training data format."""
        return f"Step {step_num}:\n{response}"
    
    def build_text_for_step(
        self,
        task_description: str,
        step_responses: List[str],
        target_step: int,
    ) -> str:
        """
        Build input text for a specific step with task description and previous steps context.
        
        This matches the format used in build_training_dataset.py:
        - Task: {task_description}
        - Previous N steps (up to context_steps)
        - Current step
        
        Args:
            task_description: The task instruction
            step_responses: List of all step responses so far
            target_step: Current step number (1-indexed)
            
        Returns:
            Formatted text string for the BERT model
        """
        # Start with task description
        parts = [f"Task: {task_description}\n"]
        
        # Add previous steps (up to context_steps)
        start_step = max(1, target_step - self.context_steps)
        
        for step_num in range(start_step, target_step + 1):
            # step_num is 1-indexed, but list is 0-indexed
            list_idx = step_num - 1
            if list_idx < len(step_responses):
                parts.append(self.format_step(step_num, step_responses[list_idx]))
        
        return "\n".join(parts)
    
    def predict(
        self,
        text: str,
    ) -> Tuple[bool, float]:
        """
        Predict whether the step is a milestone based on the formatted input.
        
        Args:
            text: Formatted input text (task + previous steps + current step)
            
        Returns:
            Tuple of (is_milestone: bool, milestone_probability: float)
        """
        if not text.strip():
            return False, 0.0
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            milestone_prob = probs[0, 1].item()  # Probability of class 1 (milestone)
        
        is_milestone = milestone_prob >= self.milestone_threshold
        
        logger.debug(f"MilestoneDetector prediction: is_milestone={is_milestone}, prob={milestone_prob:.4f}")
        
        return is_milestone, milestone_prob
    
    def check_if_milestone(
        self,
        task_description: str,
        step_responses: List[str],
        current_step: int,
    ) -> Tuple[bool, float, str]:
        """
        Check if the current step is a milestone.
        
        This is the main method to use during agent execution.
        
        Args:
            task_description: The task instruction
            step_responses: List of all agent responses so far
            current_step: Current step number (1-indexed)
            
        Returns:
            Tuple of (is_milestone: bool, milestone_probability: float, formatted_text: str)
        """
        # Format the input
        formatted_text = self.build_text_for_step(
            task_description=task_description,
            step_responses=step_responses,
            target_step=current_step,
        )
        
        # Get prediction
        is_milestone, milestone_prob = self.predict(formatted_text)
        
        logger.info(f"MilestoneDetector check at step {current_step}: is_milestone={is_milestone}, prob={milestone_prob:.4f}")
        
        return is_milestone, milestone_prob, formatted_text


class MilestoneJudge:
    """
    Uses Claude (or other big model) to judge whether a detected milestone was successfully completed.
    
    When the MilestoneDetector detects a milestone step, this class sends the context
    to Claude to verify if the milestone was achieved successfully.
    """
    
    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for GUI automation tasks. Your job is to analyze milestone steps in GUI agent trajectories.

A milestone step is a significant checkpoint in completing a task - for example:
- Successfully opening an application
- Completing a form section
- Completing a sub-goal of the overall task

You will be given:
1. The task description
2. The reasoning/actions from the previous milestone (or start) to the current step
3. A screenshot from the previous milestone (or initial state) - labeled as "BEFORE"
4. A screenshot of the current state - labeled as "AFTER"

Your job is to:
1. Infer what milestone the agent was trying to achieve based on the actions taken
2. Based on the AFTER screenshots to determine if the milestone was successfully completed, the BEFORE screenshot may provide more context, but that is not the main source of the information.
3. Look for concrete evidence in the AFTER screenshot that the intended action was achieved

Be objective and thorough in your analysis."""

    # JSON Schema for structured output
    RESPONSE_SCHEMA = {
        "name": "milestone_judgment",
        "description": "Judgment of whether a milestone was successfully completed",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "inferred_milestone": {
                    "type": "string",
                    "description": "A description of what milestone the agent was trying to achieve based on the actions and reasonings"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the milestone was successfully completed (true) or not (false)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why the milestone was or wasn't successfully completed."
                }
            },
            "required": ["inferred_milestone", "success", "reasoning"],
            "additionalProperties": False
        }
    }

    def __init__(
        self,
        anthropic_client=None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 2048,
    ):
        """
        Initialize the MilestoneJudge.
        
        Args:
            anthropic_client: An Anthropic client instance (if None, will create one)
            model: Claude model to use for judgment
            max_tokens: Maximum tokens for response
        """
        self.model = model
        self.max_tokens = max_tokens
        
        if anthropic_client is None:
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except Exception as e:
                logger.warning(f"Failed to create Anthropic client: {e}")
                self.client = None
        else:
            self.client = anthropic_client
    
    def judge_milestone(
        self,
        task_description: str,
        reasoning_history: str,
        current_screenshot_b64: str,
        previous_milestone_screenshot_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Judge whether a milestone was successfully completed.
        
        Args:
            task_description: The overall task description
            reasoning_history: Reasoning/actions from previous milestone to current step
            current_screenshot_b64: Base64 encoded screenshot of current state (AFTER)
            previous_milestone_screenshot_b64: Base64 encoded screenshot from previous milestone (BEFORE)
            
        Returns:
            Dict with keys:
                - inferred_milestone: str (what milestone was attempted)
                - success: bool (whether it was successful)
                - reasoning: str (explanation)
        """
        if self.client is None:
            logger.error("No Anthropic client available for milestone judgment")
            return {
                "inferred_milestone": "Unknown",
                "success": True,  # Default to success if we can't judge
                "reasoning": "Unable to judge milestone - no Anthropic client available",
                "error": True,
            }
        
        # Build the content list with images
        content = []
        
        # Add previous milestone screenshot (BEFORE) if available
        if previous_milestone_screenshot_b64:
            content.append({
                "type": "text",
                "text": "## BEFORE Screenshot (Previous Milestone / Initial State)"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": previous_milestone_screenshot_b64,
                },
            })
        
        # Add current screenshot (AFTER)
        content.append({
            "type": "text",
            "text": "## AFTER Screenshot (Current State)"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": current_screenshot_b64,
            },
        })
        
        # Build the user message
        user_message = f"""## Task Description
{task_description}

## Actions Since Previous Milestone
{reasoning_history}

Please analyze whether the milestone step was successful by comparing the BEFORE and AFTER screenshots and examining the actions taken."""

        content.append({
            "type": "text",
            "text": user_message,
        })

        # Add JSON format instruction to the content
        json_instruction = f"""

Please respond ONLY with a valid JSON object matching this exact schema (no additional text):
```json
{{
  "inferred_milestone": "string - description of what milestone was attempted",
  "success": true/false,
  "reasoning": "string - explanation of why it succeeded or failed"
}}
```"""
        
        content.append({
            "type": "text",
            "text": json_instruction,
        })

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.JUDGE_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            )
            
            # Extract the response text
            response_text = response.content[0].text.strip()
            
            # Handle markdown code blocks if present
            if response_text.startswith("```"):
                # Remove code block markers
                lines = response_text.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()
            
            result = json.loads(response_text)
            
            # Ensure required fields exist
            if "success" not in result:
                result["success"] = True
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"
            if "inferred_milestone" not in result:
                result["inferred_milestone"] = "Unknown milestone"
            
            logger.info(f"Milestone judgment: success={result['success']}, milestone={result['inferred_milestone']}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse milestone judgment response as JSON: {e}")
            # Try to extract key information from the response
            return {
                "inferred_milestone": "Unknown",
                "success": True,  # Default to success if we can't parse
                "reasoning": f"Failed to parse response: {response_text[:500] if 'response_text' in dir() else 'No response'}",
                "parse_error": True,
            }
        except Exception as e:
            logger.error(f"Error during milestone judgment: {e}")
            return {
                "inferred_milestone": "Unknown",
                "success": True,  # Default to success on error
                "reasoning": f"Error during judgment: {str(e)}",
                "error": True,
            }


class MilestoneTracker:
    """
    Tracks milestone steps and maintains history between milestones.
    
    This class keeps track of:
    - Which steps were detected as milestones
    - The reasoning/actions between milestones
    - Whether each milestone was judged as successful
    - Screenshots at each milestone for comparison
    """
    
    def __init__(self):
        """Initialize the milestone tracker."""
        self.milestones = []  # List of milestone records
        self.last_milestone_step = 0  # Step number of the last milestone
        self.last_milestone_screenshot_b64 = None  # Screenshot at last milestone (base64)
        self.initial_screenshot_b64 = None  # Screenshot at task start (base64)
        self.reasoning_since_last_milestone = []  # Responses since last milestone
    
    def reset(self):
        """Reset the tracker for a new task."""
        self.milestones = []
        self.last_milestone_step = 0
        self.last_milestone_screenshot_b64 = None
        self.initial_screenshot_b64 = None
        self.reasoning_since_last_milestone = []
    
    def set_initial_screenshot(self, screenshot_b64: str):
        """
        Set the initial screenshot at task start.
        
        Args:
            screenshot_b64: Base64 encoded screenshot
        """
        self.initial_screenshot_b64 = screenshot_b64
        logger.debug("Initial screenshot set for milestone tracking")
    
    def add_step(self, step_num: int, response: str):
        """
        Add a step's response to the current milestone segment.
        
        Args:
            step_num: The step number (1-indexed)
            response: The agent's response for this step
        """
        self.reasoning_since_last_milestone.append({
            "step_num": step_num,
            "response": response,
        })
    
    def get_reasoning_since_last_milestone(self) -> str:
        """
        Get formatted reasoning from the last milestone to current step.
        
        Returns:
            Formatted string of all steps since last milestone
        """
        parts = []
        for entry in self.reasoning_since_last_milestone:
            parts.append(f"Step {entry['step_num']}:\n{entry['response']}")
        return "\n\n".join(parts)
    
    def get_previous_milestone_screenshot(self) -> Optional[str]:
        """
        Get the screenshot from the previous milestone (or initial state).
        
        Returns:
            Base64 encoded screenshot, or None if not available
        """
        if self.last_milestone_screenshot_b64:
            return self.last_milestone_screenshot_b64
        return self.initial_screenshot_b64
    
    def record_milestone(
        self,
        step_num: int,
        milestone_prob: float,
        judgment: Dict[str, Any],
        screenshot_file: str = None,
        screenshot_b64: str = None,
    ):
        """
        Record a detected milestone and its judgment.
        
        Args:
            step_num: The step number of this milestone
            milestone_prob: Probability from the detector
            judgment: Result from MilestoneJudge
            screenshot_file: Optional path to the screenshot file
            screenshot_b64: Optional base64 encoded screenshot for this milestone
        """
        milestone_record = {
            "step_num": step_num,
            "milestone_prob": milestone_prob,
            "inferred_milestone": judgment.get("inferred_milestone", "Unknown"),
            "success": judgment.get("success", True),
            "judgment_reasoning": judgment.get("reasoning", ""),
            "reasoning_since_previous": self.get_reasoning_since_last_milestone(),
            "screenshot_file": screenshot_file,
            "previous_milestone_step": self.last_milestone_step,
        }
        
        self.milestones.append(milestone_record)
        
        # Update tracking
        self.last_milestone_step = step_num
        if screenshot_b64:
            self.last_milestone_screenshot_b64 = screenshot_b64
        self.reasoning_since_last_milestone = []
        
        logger.info(f"Recorded milestone at step {step_num}: success={judgment.get('success')}")
    
    def get_last_milestone(self) -> Optional[Dict[str, Any]]:
        """Get the most recent milestone record."""
        if self.milestones:
            return self.milestones[-1]
        return None
    
    def get_all_milestones(self) -> List[Dict[str, Any]]:
        """Get all milestone records."""
        return self.milestones
    
    def get_failed_milestones(self) -> List[Dict[str, Any]]:
        """Get all milestones that were judged as failed."""
        return [m for m in self.milestones if not m.get("success", True)]


class DummyMilestoneDetector:
    """
    A dummy milestone detector for testing or when the model is not available.
    Always returns not milestone.
    """
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using DummyMilestoneDetector - no actual milestone detection")
        self.context_steps = kwargs.get('context_steps', 5)
    
    def check_if_milestone(self, *args, **kwargs) -> Tuple[bool, float, str]:
        return False, 0.0, ""


def create_milestone_detector(
    model_path: str = "/gpfs/radev/scratch/cohan/jw3278/modernbert-milestone-detector",
    use_dummy: bool = False,
    **kwargs,
) -> MilestoneDetector:
    """
    Factory function to create a MilestoneDetector.
    
    Args:
        model_path: Path to the fine-tuned model
        use_dummy: If True, return a DummyMilestoneDetector instead
        **kwargs: Additional arguments passed to MilestoneDetector
        
    Returns:
        MilestoneDetector or DummyMilestoneDetector instance
    """
    if use_dummy:
        return DummyMilestoneDetector(**kwargs)
    
    try:
        return MilestoneDetector(model_path=model_path, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to create MilestoneDetector, falling back to dummy: {e}")
        return DummyMilestoneDetector(**kwargs)
