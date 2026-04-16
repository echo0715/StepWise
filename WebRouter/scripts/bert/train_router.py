#!/usr/bin/env python3
"""Fine-tune ModernBERT for binary model routing classification.

Trains a classifier to route tasks between two models:
  - gpt-oss-20b (label=0)
  - gpt-5-mini  (label=1)

Adapted from 3rd_party/bert/modernbert_finetune.py, simplified for the
model-routing use case.

Usage:
  python train_router.py
  python train_router.py --dataset /path/to/routing_dataset.json --num-epochs 10
  python train_router.py --no-class-weights --weight-strategy sqrt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ModernBERT uses @torch.compile at import time. PyTorch in this Python 3.12
# environment raises on torch.compile, so fall back to a no-op decorator.
if sys.version_info >= (3, 12):
    def _noop_compile(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f

    torch.compile = _noop_compile

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import *  # noqa: F401, F403

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = str(Path(__file__).resolve().parent / "output" / "routing_dataset.json")
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "output" / "modernbert-router")
DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_WEIGHT_STRATEGY = "balanced"
DEFAULT_LABEL_MAP = {0: "gpt-oss-20b", 1: "gpt-5-mini"}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
class EncodedTextDataset(torch.utils.data.Dataset):
    """Torch dataset holding tokenized text classification examples."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        texts = [row["text"] for row in rows]
        labels = [int(row["label"]) for row in rows]
        encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.features = []
        for idx in range(len(rows)):
            item = {k: v[idx] for k, v in encodings.items()}
            item["labels"] = labels[idx]
            self.features.append(item)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]


def load_routing_dataset(path: str, seed: int = 42) -> dict[str, list[dict]]:
    """Load routing dataset from JSON and split into train/test.

    If entries contain a ``split`` field with values ``"train"`` / ``"test"``,
    that field is used directly. Otherwise a random 80/20 split is performed.

    Args:
        path: Path to the JSON dataset file.
        seed: Random seed for the fallback random split.

    Returns:
        Dict with ``train`` and ``test`` splits, each a list of row dicts.
    """
    logger.info("Loading dataset from %s", path)
    with open(path, "r") as f:
        data = json.load(f)

    logger.info("Loaded %d examples", len(data))

    # Ensure labels are ints
    for item in data:
        item["label"] = int(item["label"])

    # Check whether the data already carries explicit splits
    has_splits = all("split" in item for item in data)

    if has_splits:
        train_data = [item for item in data if item["split"] == "train"]
        test_data = [item for item in data if item["split"] == "test"]
        if len(train_data) == 0 or len(test_data) == 0:
            logger.warning(
                "Split field present but one split is empty (train=%d, test=%d). "
                "Falling back to random 80/20 split.",
                len(train_data),
                len(test_data),
            )
            has_splits = False

    if has_splits:
        logger.info(
            "Using explicit split field: %d train, %d test", len(train_data), len(test_data)
        )
        return {"train": train_data, "test": test_data}

    # Fallback: random 80/20 split
    logger.info("No split field found; performing random 80/20 split (seed=%d)", seed)
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * 0.2))
    test_data = shuffled[:n_test]
    train_data = shuffled[n_test:]
    return {"train": train_data, "test": test_data}


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------
def compute_class_weights(
    labels: list[int], strategy: str = "balanced"
) -> dict[int, float]:
    """Compute per-class weights for imbalanced data.

    Args:
        labels: List of integer class labels.
        strategy: ``"balanced"`` for inverse frequency or ``"sqrt"`` for the
            square-root dampened variant.

    Returns:
        Dict mapping class index to weight.
    """
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(counter)
    weights: dict[int, float] = {}
    for cls, count in counter.items():
        if strategy == "balanced":
            weights[cls] = total / (n_classes * count)
        elif strategy == "sqrt":
            weights[cls] = math.sqrt(total / (n_classes * count))
        else:
            raise ValueError(f"Unknown weight strategy: {strategy!r}")
    return weights


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, and recall for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a dataloader and return loss + binary metrics."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            losses.append(loss.item())
            all_logits.append(outputs.logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics((logits, labels))
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def resolve_device(requested: str) -> torch.device:
    """Choose a safe training device."""
    if requested == "cpu":
        return torch.device("cpu")

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        try:
            a = torch.randn((2, 2), device="cuda")
            b = torch.randn((2, 2), device="cuda")
            _ = a @ b
            return torch.device("cuda")
        except Exception as exc:
            if requested == "cuda":
                raise RuntimeError(f"CUDA requested but unusable: {exc}") from exc
            logger.warning(
                "CUDA detected but unusable for training, falling back to CPU: %s",
                exc,
            )

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ModernBERT for binary model routing (gpt-oss-20b vs gpt-5-mini)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to routing_dataset.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the trained model and artefacts.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id for the base model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum tokenizer sequence length.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        dest="use_class_weights",
        help="Use class weights for imbalanced data (default: enabled).",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        help="Disable class weighting.",
    )
    parser.add_argument(
        "--weight-strategy",
        type=str,
        default=DEFAULT_WEIGHT_STRATEGY,
        choices=["balanced", "sqrt"],
        help="Class weighting strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--label-0-name",
        type=str,
        default=DEFAULT_LABEL_MAP[0],
        help="Display name for label 0 in logs/config output.",
    )
    parser.add_argument(
        "--label-1-name",
        type=str,
        default=DEFAULT_LABEL_MAP[1],
        help="Display name for label 1 in logs/config output.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device selection (default: %(default)s)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()
    set_seed(args.seed)
    label_map = {0: args.label_0_name, 1: args.label_1_name}

    logger.info("=" * 60)
    logger.info("Model routing fine-tuning: %s -> binary classifier", args.model_name)
    logger.info("Dataset : %s", args.dataset)
    logger.info("Output  : %s", args.output_dir)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    raw_dataset = load_routing_dataset(args.dataset, seed=args.seed)
    logger.info("Train samples: %d", len(raw_dataset["train"]))
    logger.info("Test  samples: %d", len(raw_dataset["test"]))

    train_labels = [int(item["label"]) for item in raw_dataset["train"]]
    test_labels = [int(item["label"]) for item in raw_dataset["test"]]
    for split_name, labels in [("Train", train_labels), ("Test", test_labels)]:
        counts = Counter(labels)
        logger.info(
            "%s label distribution: %s (0=%s, 1=%s)",
            split_name,
            dict(counts),
            label_map[0],
            label_map[1],
        )

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length

    train_dataset = EncodedTextDataset(raw_dataset["train"], tokenizer, args.max_length)
    test_dataset = EncodedTextDataset(raw_dataset["test"], tokenizer, args.max_length)

    # ------------------------------------------------------------------
    # 3. Class weights
    # ------------------------------------------------------------------
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_labels, strategy=args.weight_strategy)
        logger.info(
            "Class weights (%s): %s",
            args.weight_strategy,
            {label_map[k]: round(v, 4) for k, v in class_weights.items()},
        )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    # Disable reference_compile to avoid Dynamo/FX issues
    if hasattr(model, "config") and hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    device = resolve_device(args.device)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(train_dataloader) * args.num_epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(
            [class_weights[i] for i in range(len(class_weights))],
            dtype=torch.float32,
            device=device,
        )
        logger.info("Using weighted loss on device %s", device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    logger.info("Starting training on %s...", device)
    global_step = 0
    best_eval_results = None
    best_f1 = float("-inf")

    for epoch_idx in range(args.num_epochs):
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_dataloader, start=1):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % 50 == 0:
                logger.info(
                    "Epoch %d/%d step %d/%d - train_loss=%.4f",
                    epoch_idx + 1,
                    args.num_epochs,
                    batch_idx,
                    len(train_dataloader),
                    float(sum(epoch_losses) / len(epoch_losses)),
                )

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        eval_results = evaluate_model(model, eval_dataloader, device)
        logger.info(
            "Epoch %d/%d done - train_loss=%.4f eval_loss=%.4f eval_f1=%.4f eval_acc=%.4f",
            epoch_idx + 1,
            args.num_epochs,
            train_loss,
            eval_results["loss"],
            eval_results["f1"],
            eval_results["accuracy"],
        )

        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            best_eval_results = {"epoch": epoch_idx + 1, **eval_results}
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saved new best checkpoint to %s", output_dir)

    if best_eval_results is None:
        best_eval_results = evaluate_model(model, eval_dataloader, device)
        best_eval_results["epoch"] = args.num_epochs
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    logger.info("Best test results:")
    for key, value in sorted(best_eval_results.items()):
        if isinstance(value, (int, float)):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    # Evaluation results
    eval_path = Path(output_dir) / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(best_eval_results, f, indent=2)
    logger.info("Evaluation results saved to %s", eval_path)

    # Training config for reproducibility
    config_dict = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "output_dir": output_dir,
        "max_length": args.max_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "use_class_weights": args.use_class_weights,
        "weight_strategy": args.weight_strategy if args.use_class_weights else None,
        "class_weights": (
            {str(k): round(v, 6) for k, v in class_weights.items()}
            if class_weights is not None
            else None
        ),
        "seed": args.seed,
        "requested_device": args.device,
        "device": str(device),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "label_map": label_map,
    }
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Training config saved to %s", config_path)

    logger.info("Done. All artefacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
