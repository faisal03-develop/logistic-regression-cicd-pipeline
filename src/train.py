"""
Training pipeline: orchestrates data loading, preprocessing,
model training, evaluation, and artifact persistence.
"""

import os
import json
import argparse
import numpy as np

from src.model import LogisticRegression
from src.preprocess import preprocess_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.txt")


# ---------------------------------------------------------------------------
# Metric helpers (implemented from scratch)
# ---------------------------------------------------------------------------

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Return (TP, FP, TN, FN)."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics from scratch.

    Returns
    -------
    dict with accuracy, precision, recall, f1
    """
    tp, fp, tn, fn = _confusion_matrix(y_true, y_pred)

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def print_metrics(metrics: dict, split: str = "Test") -> None:
    """Pretty-print metrics to stdout."""
    print(f"\n{'─'*45}")
    print(f"  {split} Set Metrics")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  TP={metrics['true_positives']}  TN={metrics['true_negatives']}"
          f"  FP={metrics['false_positives']}  FN={metrics['false_negatives']}")
    print(f"{'─'*45}\n")


def save_metrics(metrics: dict, path: str) -> None:
    """Persist metrics to a plain-text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "=== Credit Risk Classifier — Model Metrics ===\n",
        f"Accuracy  : {metrics['accuracy']:.6f}\n",
        f"Precision : {metrics['precision']:.6f}\n",
        f"Recall    : {metrics['recall']:.6f}\n",
        f"F1 Score  : {metrics['f1_score']:.6f}\n",
        f"\nConfusion Matrix:\n",
        f"  TP={metrics['true_positives']}  FP={metrics['false_positives']}\n",
        f"  FN={metrics['false_negatives']}  TN={metrics['true_negatives']}\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[Train] Metrics saved to {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    metrics_path: str = METRICS_PATH,
    learning_rate: float = 0.05,
    n_iterations: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Full training pipeline.

    1. Load and preprocess data
    2. Train logistic regression model
    3. Evaluate on validation and test sets
    4. Save model, scaler, and metrics

    Returns
    -------
    dict of test metrics
    """
    print("\n" + "=" * 55)
    print("  Credit Risk Classifier — Training Pipeline")
    print("=" * 55)

    # Step 1: Preprocess
    data = preprocess_pipeline(
        csv_path=data_path,
        scaler_save_path=scaler_path,
    )

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print(f"\n[Train] Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {int(cls)}: {cnt:,} ({cnt/len(y_train)*100:.1f}%)")

    # Step 2: Train
    model = LogisticRegression(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        verbose=verbose,
        verbose_interval=100,
    )
    model.fit(X_train, y_train)

    # Step 3: Evaluate
    val_preds = model.predict(X_val)
    val_metrics = compute_metrics(y_val, val_preds)
    print_metrics(val_metrics, split="Validation")

    test_preds = model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)
    print_metrics(test_metrics, split="Test")

    # Step 4: Persist artifacts
    model.save(model_path)
    save_metrics(test_metrics, metrics_path)

    print("[Train] All artifacts saved successfully.\n")
    return test_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Credit Risk Classifier")
    parser.add_argument("--data", default=DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logs")
    args = parser.parse_args()

    metrics = train(
        data_path=args.data,
        learning_rate=args.lr,
        n_iterations=args.iters,
        verbose=not args.quiet,
    )
    print(json.dumps(metrics, indent=2))
