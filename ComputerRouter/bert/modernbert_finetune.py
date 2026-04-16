#!/usr/bin/env python3
"""
Fine-tune ModernBERT for GUI agent binary classification tasks.

Supported tasks:
  - stuck: Detect when GUI agents get stuck in repetitive loops
  - milestone: Detect milestone steps in GUI agent trajectories

Typical install (example):
  pip install -U "torch==2.4.1" tensorboard scikit-learn datasets==3.1.0 accelerate==1.2.1 hf-transfer==0.1.8 huggingface_hub
  pip install "git+https://github.com/huggingface/transformers.git@6e0515e99c39444caae39472ee1b2fd76ece32f1"

Note:
- ModernBERT may require a recent Transformers version (often from GitHub).
- flash-attn is optional and platform-dependent.

Usage:
  # For stuck detection:
  python modernbert_finetune.py --task stuck --dataset_path bert_training_dataset.json --output_dir modernbert-stuck-detector
  
  # For milestone detection:
  python modernbert_finetune.py --task milestone --dataset_path milestone_training_dataset.json --output_dir modernbert-milestone-detector

If you want to push to the Hugging Face Hub:
  export HF_TOKEN=...   # or pass --hf_token ...
  python modernbert_finetune.py --push_to_hub --hub_repo_id <your-namespace>/<repo-name>
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder, login
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# Task configurations
TASK_CONFIGS = {
    "stuck": {
        "description": "Stuck step detection",
        "default_dataset": "bert_training_dataset.json",
        "default_output": "modernbert-stuck-detector",
        "positive_label": "Stuck",
        "negative_label": "Not stuck",
    },
    "milestone": {
        "description": "Milestone step detection", 
        "default_dataset": "milestone_training_dataset.json",
        "default_output": "modernbert-milestone-detector",
        "positive_label": "Milestone",
        "negative_label": "Non-milestone",
    }
}


def infer_text_column(dataset: DatasetDict) -> str:
    """
    Determine which column contains the input text.

    The notebook's sample shows columns like: {"id", "prompt", "label"}.
    Some datasets use "text" instead of "prompt".
    """
    cols = set(dataset["train"].column_names)
    if "text" in cols:
        return "text"
    if "prompt" in cols:
        return "prompt"
    # Add more fallbacks if needed
    raise ValueError(
        f"Could not infer a text column. Available columns: {sorted(cols)}. "
        "Expected one of ['text', 'prompt']."
    )


def infer_label_column(dataset: DatasetDict) -> str:
    cols = set(dataset["train"].column_names)
    if "labels" in cols:
        return "labels"
    if "label" in cols:
        return "label"
    raise ValueError(
        f"Could not infer a label column. Available columns: {sorted(cols)}. "
        "Expected one of ['label', 'labels']."
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    # Compute confusion matrix for detailed analysis
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


class WeightedTrainer(Trainer):
    """
    Custom Trainer that supports class weights for imbalanced datasets.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self._weights_on_device = {}  # Cache for weights on different devices
        self._debug_printed = False
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get labels but don't pop - let the model handle them too
        labels = inputs.get("labels")
        
        # Forward pass - model will compute its own loss
        outputs = model(**inputs)
        
        if self.class_weights is not None and labels is not None:
            # Recompute loss with class weights
            logits = outputs.logits
            
            # Ensure labels are 1D and Long type
            labels = labels.view(-1).long()
            
            # Get or create weights on the correct device
            device = logits.device
            if device not in self._weights_on_device:
                self._weights_on_device[device] = self.class_weights.to(device)
            weights = self._weights_on_device[device]
            
            # Compute weighted cross entropy loss
            # logits: (batch_size, num_classes), labels: (batch_size,)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits, labels)
        else:
            # Use the model's computed loss
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: List[int], strategy: str = "balanced") -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        labels: List of integer labels
        strategy: 'balanced' for inverse frequency, 'sqrt' for sqrt of inverse frequency
    
    Returns:
        Tensor of class weights
    """
    from collections import Counter
    label_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(label_counts)
    
    if strategy == "balanced":
        # Inverse frequency weighting
        weights = {cls: n_samples / (n_classes * count) for cls, count in label_counts.items()}
    elif strategy == "sqrt":
        # Square root of inverse frequency (less aggressive)
        weights = {cls: np.sqrt(n_samples / (n_classes * count)) for cls, count in label_counts.items()}
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
    
    # Create weight tensor in class order
    weight_tensor = torch.tensor([weights[i] for i in range(n_classes)], dtype=torch.float32)
    return weight_tensor


def maybe_login(hf_token: Optional[str]) -> None:
    """
    Log in to the Hugging Face Hub if a token is provided (or found in env).
    """
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)
    else:
        # Not an error unless push_to_hub is requested.
        return


def choose_precision_flags(user_bf16: Optional[bool], user_fp16: Optional[bool]) -> Tuple[bool, bool]:
    """
    Decide bf16/fp16 settings. If user specified them, honor that.
    Otherwise, attempt bf16 on supported CUDA; else disable.
    """
    if user_bf16 is not None or user_fp16 is not None:
        return bool(user_bf16), bool(user_fp16)

    try:
        import torch  # local import to avoid hard dependency during arg parsing
        if torch.cuda.is_available():
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if is_bf16_supported:
                return True, False
            return False, True  # common fallback
        return False, False
    except Exception:
        return False, False


def choose_optimizer_name() -> str:
    """
    The notebook uses 'adamw_torch_fused'. That may fail on CPU or older builds.
    We fall back safely when fused optimizer is unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "adamw_torch_fused"
        return "adamw_torch"
    except Exception:
        return "adamw_torch"


def build_label_maps(tokenized_dataset: DatasetDict) -> Tuple[int, Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """
    Build label2id/id2label maps when the dataset provides class names (ClassLabel feature).
    If not available, return None maps and rely on numeric labels.
    """
    feat = tokenized_dataset["train"].features.get("labels")
    if feat is not None and hasattr(feat, "names") and feat.names:
        names = list(feat.names)
        num_labels = len(names)
        label2id = {name: str(i) for i, name in enumerate(names)}
        id2label = {str(i): name for i, name in enumerate(names)}
        return num_labels, label2id, id2label

    # Fallback: infer from unique values in train split
    unique = set(tokenized_dataset["train"]["labels"])
    num_labels = len(unique)
    return num_labels, None, None


def load_local_dataset(dataset_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load dataset from local JSON file and split into train/test.
    
    Args:
        dataset_path: Path to JSON file with list of dicts containing 'text' and 'label'
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducible split
    
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Ensure labels are integers (0 or 1)
    for item in data:
        item['label'] = int(item['label'])
    
    # Create Dataset object
    dataset = Dataset.from_list(data)
    
    # Split into train/test
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    return DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT for GUI agent binary classification tasks.")
    
    # Task selection
    parser.add_argument("--task", type=str, default="stuck", choices=["stuck", "milestone"],
                        help="Task to train: 'stuck' for stuck detection, 'milestone' for milestone detection.")
    
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to local JSON dataset file. If not provided, uses task-specific default.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing.")
    parser.add_argument("--model_id", type=str, default="answerdotai/ModernBERT-base",
                        help="Model checkpoint ID.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save outputs (and default Hub repo name). If not provided, uses task-specific default.")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max sequence length for tokenization.")
    parser.add_argument("--num_train_epochs", type=float, default=5,
                        help="Training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Per-device train batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Per-device eval batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    
    # Class weighting for imbalanced datasets
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use class weights to handle imbalanced datasets (recommended for milestone task).")
    parser.add_argument("--weight_strategy", type=str, default="balanced", choices=["balanced", "sqrt"],
                        help="Class weighting strategy: 'balanced' (inverse frequency) or 'sqrt' (less aggressive).")
    
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether to push checkpoints/model card to the Hugging Face Hub.")
    parser.add_argument("--hub_repo_id", type=str, default=None,
                        help="Hub repo id, e.g., 'username/modernbert-stuck-detector'. Defaults to output_dir.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token. If omitted, uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN env vars.")
    parser.add_argument(
        "--reference_compile",
        action="store_true",
        help=(
            "Enable ModernBERT's `config.reference_compile` path (uses a `@torch.compile`'d MLP inside each encoder "
            "layer). This can crash on some PyTorch/Transformers setups with errors like "
            "`FX symbolically trace a dynamo-optimized function`. Default: disabled."
        ),
    )
    # Precision overrides (optional)
    parser.add_argument("--bf16", dest="bf16", action="store_true", help="Force bf16 training.")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", help="Disable bf16 training.")
    parser.set_defaults(bf16=None)

    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Force fp16 training.")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false", help="Disable fp16 training.")
    parser.set_defaults(fp16=None)

    args = parser.parse_args()
    
    # Get task configuration
    task_config = TASK_CONFIGS[args.task]
    
    # Apply task-specific defaults if not provided
    if args.dataset_path is None:
        args.dataset_path = task_config["default_dataset"]
    if args.output_dir is None:
        args.output_dir = task_config["default_output"]
    
    print("\n" + "="*60)
    print(f"Task: {task_config['description']}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")

    set_seed(args.seed)

    if args.push_to_hub:
        maybe_login(args.hf_token)
        # Ensure there is a token available for pushing
        if not (args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()):
            raise RuntimeError(
                "push_to_hub was requested but no Hugging Face token was found. "
                "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) or pass --hf_token."
            )

    # Load dataset from local JSON file
    raw_dataset: DatasetDict = load_local_dataset(args.dataset_path, args.test_size, args.seed)
    print(f"Train dataset size: {len(raw_dataset['train'])}")
    print(f"Test dataset size: {len(raw_dataset['test'])}")
    
    # Print label distribution with task-specific labels
    train_labels = raw_dataset['train']['label']
    test_labels = raw_dataset['test']['label']
    neg_label = task_config["negative_label"]
    pos_label = task_config["positive_label"]
    
    train_neg = train_labels.count(0)
    train_pos = train_labels.count(1)
    test_neg = test_labels.count(0)
    test_pos = test_labels.count(1)
    
    print(f"\nTrain label distribution:")
    print(f"  {neg_label} (0): {train_neg}")
    print(f"  {pos_label} (1): {train_pos}")
    print(f"  Imbalance ratio: {train_neg/train_pos:.2f}:1" if train_pos > 0 else "")
    print(f"Test label distribution:")
    print(f"  {neg_label} (0): {test_neg}")
    print(f"  {pos_label} (1): {test_pos}")
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, args.weight_strategy)
        print(f"\nClass weights ({args.weight_strategy}): {class_weights.tolist()}")
        print(f"  {neg_label} weight: {class_weights[0]:.4f}")
        print(f"  {pos_label} weight: {class_weights[1]:.4f}")

    text_col = "text"
    label_col = "label"

    # Normalize columns to what Trainer expects
    if label_col != "labels":
        raw_dataset = raw_dataset.rename_column(label_col, "labels")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = args.max_length

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    # Tokenize and drop raw text/id columns
    # Remove all non-model inputs except labels.
    remove_cols = [c for c in raw_dataset["train"].column_names if c not in ("labels",)]
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    # Set torch format (Trainer will convert automatically, but this is explicit and convenient)
    tokenized_dataset.set_format(type="torch")

    # Label maps (optional)
    num_labels, label2id, id2label = build_label_maps(tokenized_dataset)
    
    # For binary classification, ensure num_labels is 2
    if args.task in ["stuck", "milestone"]:
        num_labels = 2
        print(f"\nUsing num_labels={num_labels} for binary classification")

    # Model
    model_kwargs = {"num_labels": num_labels}
    if label2id is not None and id2label is not None:
        model_kwargs.update({"label2id": label2id, "id2label": id2label})

    # ModernBERT + transformers 4.48 + no flash-attn produces NaN logits under bf16
    # autocast (Trainer then logs NaN loss as 0.0 and grad_norm as nan). Force eager
    # attention unless flash-attn is installed, which is the only path that's numerically
    # stable in this combo.
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except Exception:
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **model_kwargs)
    # ModernBERT implements an optional `reference_compile` path that calls a `@torch.compile`'d MLP per layer.
    # On some stacks this can trip Dynamo/FX tracing interactions during training. Disable by default.
    if hasattr(model, "config") and hasattr(model.config, "reference_compile"):
        model.config.reference_compile = bool(args.reference_compile)

    bf16, fp16 = choose_precision_flags(args.bf16, args.fp16)
    optim_name = choose_optimizer_name()

    if bf16 and fp16:
        warnings.warn("Both bf16 and fp16 are enabled; bf16 will generally take precedence on supported hardware.")

    hub_repo_id = args.hub_repo_id or args.output_dir

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=bf16,
        fp16=fp16,
        optim=optim_name,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
        push_to_hub=args.push_to_hub,
        hub_strategy="every_save" if args.push_to_hub else "end",
        hub_token=args.hf_token or HfFolder.get_token(),
        hub_model_id=hub_repo_id if args.push_to_hub else None,
    )

    # Choose trainer based on whether class weights are used
    if class_weights is not None:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        print("\nUsing WeightedTrainer for class-imbalanced training")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()

    print("\n" + "="*50)
    print("Training complete! Evaluating on test set...")
    print("="*50 + "\n")
    
    # Final evaluation
    eval_results = trainer.evaluate()
    print("\nFinal Test Set Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save evaluation results
    eval_output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(eval_output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_output_path}")

    # Save locally
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)
    trainer.create_model_card()
    
    # Save training config
    config_output_path = os.path.join(args.output_dir, "training_config.json")
    config_dict = {
        "task": args.task,
        "task_description": task_config["description"],
        "dataset_path": args.dataset_path,
        "model_id": args.model_id,
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "test_size": args.test_size,
        "use_class_weights": args.use_class_weights,
        "weight_strategy": args.weight_strategy if args.use_class_weights else None,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "train_samples": len(tokenized_dataset["train"]),
        "test_samples": len(tokenized_dataset["test"]),
        "label_mapping": {
            "0": task_config["negative_label"],
            "1": task_config["positive_label"]
        }
    }
    with open(config_output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Training config saved to: {config_output_path}")

    if args.push_to_hub:
        trainer.push_to_hub()

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
