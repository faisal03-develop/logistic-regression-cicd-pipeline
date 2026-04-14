"""
Data preprocessing utilities: loading, cleaning, scaling, and splitting.
Implements StandardScaler from scratch using only NumPy.
"""


import numpy as np
import pandas as pd
import pickle
import os



# ---------------------------------------------------------------------------
# Standard Scaler (from scratch)
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    z = (x - μ) / σ

    Implemented from scratch with NumPy — no sklearn.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute per-feature mean and std from training data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Replace zero std with 1 to avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization using fitted mean and std."""
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Reverse the standardization."""
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted. Call fit() first.")
        return X_scaled * self.std_ + self.mean_

    def save(self, path: str) -> None:
        """Persist scaler to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Scaler] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StandardScaler":
        """Load a previously saved scaler from disk."""
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        if not isinstance(scaler, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance.")
        print(f"[Scaler] Loaded from {path}")
        return scaler


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "loan_tenure_months",
    "debt_to_income_ratio",
    "num_open_accounts",
    "num_credit_inquiries",
    "months_employed",
    "has_mortgage",
]

TARGET_COLUMN = "default"


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV dataset and validate required columns exist."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    print(f"[Preprocess] Loaded {len(df):,} rows from {csv_path}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - Drop rows with any null values
      - Ensure target is binary int {0, 1}
      - Clip extreme outliers using IQR for numeric features
    """
    before = len(df)
    df = df.dropna().copy()
    after_null = len(df)

    # Clip outliers beyond 3 IQRs for numeric feature columns
    numeric_features = [
        c for c in FEATURE_COLUMNS if c not in ("has_mortgage",)
    ]
    for col in numeric_features:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q99)

    # Ensure target is integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    print(
        f"[Preprocess] Cleaned dataset: {before} → {after_null} rows "
        f"(dropped {before - after_null} nulls)"
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train / validation / test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(df))

    n_test = int(len(df) * test_size)
    n_val = int(len(df) * val_size)
    n_train = len(df) - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.float64)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(
        f"[Preprocess] Split → train: {len(X_train):,} | "
        f"val: {len(X_val):,} | test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaler_save_path: str | None = None,
) -> dict:
    """
    Full preprocessing pipeline: load → clean → split → scale.

    Returns a dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
    """
    df = load_dataset(csv_path)
    df = clean_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df, test_size=test_size, val_size=val_size
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if scaler_save_path:
        scaler.save(scaler_save_path)

    print("[Preprocess] Scaling complete.")

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }


def preprocess_single_input(raw: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Preprocess a single prediction request (from the API).

    Parameters
    ----------
    raw    : dict mapping feature names → raw values
    scaler : fitted StandardScaler

    Returns
    -------
    np.ndarray, shape (1, n_features)
    """
    missing = [f for f in FEATURE_COLUMNS if f not in raw]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = np.array([[float(raw[f]) for f in FEATURE_COLUMNS]])
    return scaler.transform(row)
