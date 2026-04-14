"""
Shared pytest fixtures for unit and integration tests.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LogisticRegression
from src.preprocess import StandardScaler, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Small 2-feature dataset fixture (for fast unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_dataset():
    rng = np.random.default_rng(0)
    n = 200
    X0 = rng.normal(loc=[1, 1], scale=0.5, size=(n // 2, 2))
    y0 = np.zeros(n // 2)
    X1 = rng.normal(loc=[3, 3], scale=0.5, size=(n // 2, 2))
    y1 = np.ones(n // 2)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    split = int(n * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture(scope="session")
def trained_model(tiny_dataset):
    X_train, _, y_train, _ = tiny_dataset
    model = LogisticRegression(learning_rate=0.1, n_iterations=300, verbose=False)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def fitted_scaler(tiny_dataset):
    X_train, _, _, _ = tiny_dataset
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ---------------------------------------------------------------------------
# Full 10-feature dataset matching production FEATURE_COLUMNS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def full_trained_model():
    """10-feature model + scaler for API integration tests."""
    rng = np.random.default_rng(42)
    n = 600

    age                  = rng.integers(22, 71, size=n).astype(float)
    income               = rng.uniform(20_000, 150_000, size=n)
    credit_score         = rng.uniform(300, 850, size=n)
    loan_amount          = rng.uniform(5_000, 100_000, size=n)
    loan_tenure_months   = rng.choice([12, 24, 36, 48, 60, 72, 84], size=n).astype(float)
    debt_to_income_ratio = rng.uniform(0.0, 0.9, size=n)
    num_open_accounts    = rng.integers(1, 16, size=n).astype(float)
    num_credit_inquiries = rng.integers(0, 11, size=n).astype(float)
    months_employed      = rng.uniform(0, 360, size=n)
    has_mortgage         = rng.binomial(1, 0.35, size=n).astype(float)

    X = np.column_stack([
        age, income, credit_score, loan_amount, loan_tenure_months,
        debt_to_income_ratio, num_open_accounts, num_credit_inquiries,
        months_employed, has_mortgage,
    ])

    log_odds = (
        -0.008 * credit_score
        + 3.5 * debt_to_income_ratio
        + 0.25 * num_credit_inquiries
        + 2.0
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = rng.binomial(1, prob).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(learning_rate=0.1, n_iterations=500, verbose=False)
    model.fit(X_scaled, y)
    return model, scaler


# ---------------------------------------------------------------------------
# Flask test client — wired with the 10-feature model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flask_client(tmp_path_factory, full_trained_model):
    model, scaler = full_trained_model

    artifact_dir = tmp_path_factory.mktemp("models")
    model_path   = str(artifact_dir / "model.pkl")
    scaler_path  = str(artifact_dir / "scaler.pkl")
    metrics_path = str(artifact_dir / "metrics.txt")

    model.save(model_path)
    scaler.save(scaler_path)

    with open(metrics_path, "w") as f:
        f.write("accuracy : 0.95\nprecision : 0.93\nrecall : 0.91\nf1_score : 0.92\n")

    from src import predict as predict_module
    predict_module.registry.model_path   = model_path
    predict_module.registry.scaler_path  = scaler_path
    predict_module.registry.metrics_path = metrics_path
    predict_module.registry.reload()

    from app import create_app
    application = create_app()
    application.config["TESTING"] = True
    with application.test_client() as client:
        yield client
