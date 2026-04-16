"""
Stuck Detector module using ModernBERT for detecting when GUI agents get stuck.

This module provides a StuckDetector class that loads a fine-tuned ModernBERT model
to detect whether an agent is stuck in a repetitive loop based on its reasoning
and action history.

The input format follows the BERT training dataset format:
- Text: Concatenated history of steps with reasoning and actions
- Output: Binary classification (0 = not stuck, 1 = stuck)
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("desktopenv.stuck_detector")


class StuckDetector:
    """
    A stuck detector using ModernBERT for binary classification.
    
    Detects when a GUI agent is stuck in a repetitive loop based on
    its reasoning and action history.
    """
    
    def __init__(
        self,
        model_path: str = "/gpfs/radev/scratch/cohan/jw3278/modernbert-stuck-detector",
        device: Optional[str] = None,
        max_length: int = 2048,
        stuck_threshold: float = 0.5,
        min_steps_to_check: int = 2,
    ):
        """
        Initialize the StuckDetector.
        
        Args:
            model_path: Path to the fine-tuned ModernBERT model
            device: Device to use for inference ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            stuck_threshold: Probability threshold above which to classify as "stuck"
            min_steps_to_check: Minimum number of steps before checking for stuck
        """
        self.model_path = model_path
        self.max_length = max_length
        self.stuck_threshold = stuck_threshold
        self.min_steps_to_check = min_steps_to_check
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"StuckDetector initializing with model: {model_path}")
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
            logger.info("StuckDetector model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load StuckDetector model: {e}")
            raise
    
    def format_step_history(
        self,
        step_responses: List[str],
        step_actions: List[str],
        max_history_steps: int = 6,
    ) -> str:
        """
        Format the step history into the text format expected by the BERT model.
        
        The format follows the training data structure:
        Step N:
        Response: <reasoning and action from agent>
        Action: <pyautogui action>
        
        Args:
            step_responses: List of agent responses (reasoning + action text)
            step_actions: List of executed actions (pyautogui commands)
            max_history_steps: Maximum number of recent steps to include
            
        Returns:
            Formatted text string for the BERT model
        """
        if not step_responses:
            return ""
        
        # Take the most recent steps
        num_steps = min(len(step_responses), max_history_steps)
        start_idx = len(step_responses) - num_steps
        
        formatted_steps = []
        for i in range(start_idx, len(step_responses)):
            step_num = i + 1
            response = step_responses[i] if i < len(step_responses) else ""
            action = step_actions[i] if i < len(step_actions) else ""
            
            # Format similar to training data
            step_text = f"Step {step_num}:\nResponse: {response}\nAction: {action}\n"
            formatted_steps.append(step_text)
        
        return "\n".join(formatted_steps)
    
    def predict(
        self,
        text: str,
        return_proba: bool = False,
    ) -> Tuple[bool, float]:
        """
        Predict whether the agent is stuck based on the formatted step history.
        
        Args:
            text: Formatted step history text
            return_proba: If True, also return the probability
            
        Returns:
            Tuple of (is_stuck: bool, stuck_probability: float)
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
            stuck_prob = probs[0, 1].item()  # Probability of class 1 (stuck)
        
        is_stuck = stuck_prob >= self.stuck_threshold
        
        logger.debug(f"StuckDetector prediction: is_stuck={is_stuck}, prob={stuck_prob:.4f}")
        
        return is_stuck, stuck_prob
    
    def check_if_stuck(
        self,
        step_responses: List[str],
        step_actions: List[str],
        current_step: int,
        max_history_steps: int = 6,
    ) -> Tuple[bool, float, str]:
        """
        Check if the agent is stuck based on the current step history.
        
        This is the main method to use during agent execution.
        
        Args:
            step_responses: List of all agent responses so far
            step_actions: List of all executed actions so far
            current_step: Current step number (1-indexed)
            max_history_steps: Maximum number of recent steps to include
            
        Returns:
            Tuple of (is_stuck: bool, stuck_probability: float, formatted_history: str)
        """
        # Don't check if we haven't reached minimum steps
        if current_step < self.min_steps_to_check:
            logger.debug(f"Step {current_step} < min_steps_to_check ({self.min_steps_to_check}), skipping check")
            return False, 0.0, ""
        
        # Format the history
        formatted_text = self.format_step_history(
            step_responses,
            step_actions,
            max_history_steps=max_history_steps,
        )
        
        # Get prediction
        is_stuck, stuck_prob = self.predict(formatted_text)
        
        logger.info(f"StuckDetector check at step {current_step}: is_stuck={is_stuck}, prob={stuck_prob:.4f}")
        
        return is_stuck, stuck_prob, formatted_text
    
    def format_context_for_claude(
        self,
        instruction: str,
        step_responses: List[str],
        step_actions: List[str],
        stuck_reason: str = "",
    ) -> str:
        """
        Format the context for handoff to Claude when stuck is detected.
        
        This creates a summary of the task and previous attempts to help
        Claude understand the situation and take a different approach.
        
        Args:
            instruction: The original task instruction
            step_responses: List of all agent responses
            step_actions: List of all executed actions
            stuck_reason: Optional explanation of why agent was detected as stuck
            
        Returns:
            Formatted context string for Claude
        """
        context_parts = [
            "=" * 50,
            "TASK HANDOFF - PREVIOUS AGENT GOT STUCK",
            "=" * 50,
            "",
            "ORIGINAL INSTRUCTION:",
            instruction,
            "",
            "PREVIOUS ATTEMPTS (the agent got stuck repeating similar actions):",
            "",
        ]
        
        # Include the last few steps to show what was tried
        max_steps_to_show = min(5, len(step_responses))
        start_idx = len(step_responses) - max_steps_to_show
        
        for i in range(start_idx, len(step_responses)):
            step_num = i + 1
            response = step_responses[i] if i < len(step_responses) else ""
            action = step_actions[i] if i < len(step_actions) else ""
            
            context_parts.append(f"Step {step_num}:")
            context_parts.append(f"  Reasoning: {response[:500]}..." if len(response) > 500 else f"  Reasoning: {response}")
            context_parts.append(f"  Action: {action}")
            context_parts.append("")
        
        context_parts.extend([
            "=" * 50,
            "IMPORTANT: The previous agent got stuck in a loop.",
            "Please try a DIFFERENT approach to complete the task.",
            "Consider:",
            "- Using different UI elements or interactions",
            "- Breaking down the task differently",
            "- Looking for alternative paths to achieve the goal",
            "=" * 50,
        ])
        
        if stuck_reason:
            context_parts.extend([
                "",
                f"Stuck detection reason: {stuck_reason}",
            ])
        
        return "\n".join(context_parts)


class DummyStuckDetector:
    """
    A dummy stuck detector for testing or when the model is not available.
    Always returns not stuck.
    """
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using DummyStuckDetector - no actual stuck detection")
        self.min_steps_to_check = kwargs.get('min_steps_to_check', 2)
    
    def check_if_stuck(self, *args, **kwargs) -> Tuple[bool, float, str]:
        return False, 0.0, ""
    
    def format_context_for_claude(
        self,
        instruction: str,
        step_responses: List[str],
        step_actions: List[str],
        stuck_reason: str = "",
    ) -> str:
        logger.info(f"Formatting context for Claude: {instruction}")
        return f"Task: {instruction}\nPrevious steps: {len(step_responses)}"


def create_stuck_detector(
    model_path: str = "/gpfs/radev/scratch/cohan/jw3278/modernbert-stuck-detector",
    use_dummy: bool = False,
    **kwargs,
) -> StuckDetector:
    """
    Factory function to create a StuckDetector.
    
    Args:
        model_path: Path to the fine-tuned model
        use_dummy: If True, return a DummyStuckDetector instead
        **kwargs: Additional arguments passed to StuckDetector
        
    Returns:
        StuckDetector or DummyStuckDetector instance
    """
    if use_dummy:
        return DummyStuckDetector(**kwargs)
    
    try:
        return StuckDetector(model_path=model_path, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to create StuckDetector, falling back to dummy: {e}")
        return DummyStuckDetector(**kwargs)
