"""
Prediction module: loads trained artifacts and exposes a clean
prediction interface consumed by the Flask API.
"""

import os
import json
from typing import Any

import numpy as np

from src.model import LogisticRegression
from src.preprocess import StandardScaler, preprocess_single_input

# ---------------------------------------------------------------------------
# Default artifact paths
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.txt")

# Risk labels
RISK_LABEL = {0: "Low Risk", 1: "High Risk"}
RISK_DESCRIPTION = {
    0: "The applicant is unlikely to default on the loan.",
    1: "The applicant has a high probability of defaulting on the loan.",
}


# ---------------------------------------------------------------------------
# Artifact manager (singleton-like loader)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Lazy-loads and caches trained model and scaler."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        metrics_path: str = METRICS_PATH,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metrics_path = metrics_path

        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._metrics: dict | None = None

    # ---- loaders ----------------------------------------------------------

    @property
    def model(self) -> LogisticRegression:
        if self._model is None:
            self._model = LogisticRegression.load(self.model_path)
        return self._model

    @property
    def scaler(self) -> StandardScaler:
        if self._scaler is None:
            self._scaler = StandardScaler.load(self.scaler_path)
        return self._scaler

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self._metrics = self._load_metrics()
        return self._metrics

    def _load_metrics(self) -> dict:
        if not os.path.exists(self.metrics_path):
            return {}
        metrics: dict[str, Any] = {}
        with open(self.metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    val = val.strip()
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        metrics[key] = val
        return metrics

    def is_ready(self) -> bool:
        """Check whether all artifact files exist on disk."""
        return all(
            os.path.exists(p)
            for p in [self.model_path, self.scaler_path, self.metrics_path]
        )

    def reload(self) -> None:
        """Force re-load artifacts from disk (useful after retraining)."""
        self._model = None
        self._scaler = None
        self._metrics = None


# ---------------------------------------------------------------------------
# Global registry instance (used by app.py)
# ---------------------------------------------------------------------------

registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict(raw_input: dict) -> dict:
    """
    Run inference for a single credit application.

    Parameters
    ----------
    raw_input : dict
        Feature key-value pairs from the API request body.

    Returns
    -------
    dict with keys:
        prediction       (int)  — 0 or 1
        probability      (float) — P(default)
        risk_label       (str)  — "Low Risk" / "High Risk"
        risk_description (str)  — human-readable explanation
    """
    if not registry.is_ready():
        raise RuntimeError(
            "Model artifacts not found. Run `make train` to train the model first."
        )

    # Preprocess
    X = preprocess_single_input(raw_input, registry.scaler)

    # Inference
    proba = float(registry.model.predict_proba(X)[0])
    label = int(registry.model.predict(X)[0])

    return {
        "prediction": label,
        "probability": round(proba, 6),
        "risk_label": RISK_LABEL[label],
        "risk_description": RISK_DESCRIPTION[label],
    }


def get_metrics() -> dict:
    """Return model performance metrics loaded from disk."""
    if not registry.is_ready():
        raise RuntimeError("Model artifacts not found.")
    return registry.metrics
