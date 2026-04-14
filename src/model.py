"""
Logistic Regression implemented from scratch using only NumPy.
Used for Credit Risk Classification (default vs non-default).
"""

import numpy as np
import pickle
import os


class LogisticRegression:
    """
    Binary Logistic Regression trained via batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of full passes over the training data.
    verbose : bool
        Whether to print training progress.
    verbose_interval : int
        How often (in iterations) to print loss.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        verbose: bool = True,
        verbose_interval: int = 100,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights to small random values and bias to zero."""
        np.random.seed(42)
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.
        σ(z) = 1 / (1 + e^{-z})
        """
        # Clip to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy (log loss).
        L = -(1/m) * Σ [y * log(ŷ) + (1 - y) * log(1 - ŷ)]
        """
        m = len(y_true)
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = -(1 / m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return float(loss)

    def _compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients of log loss with respect to weights and bias.
        dL/dw = (1/m) * Xᵀ · (ŷ - y)
        dL/db = (1/m) * Σ (ŷ - y)
        """
        m = len(y_true)
        error = y_pred - y_true                     # (m,)
        dw = (1 / m) * X.T.dot(error)              # (n_features,)
        db = (1 / m) * np.sum(error)               # scalar
        return dw, db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
        y : np.ndarray, shape (m,) — binary labels {0, 1}

        Returns
        -------
        self
        """
        m, n_features = X.shape
        self._initialize_weights(n_features)
        self.loss_history = []

        if self.verbose:
            print(f"\n{'='*55}")
            print(f"  Training Logistic Regression (from scratch)")
            print(f"  Samples: {m} | Features: {n_features}")
            print(f"  LR: {self.learning_rate} | Iterations: {self.n_iterations}")
            print(f"{'='*55}")

        for i in range(1, self.n_iterations + 1):
            # Forward pass
            z = X.dot(self.weights) + self.bias      # linear combination
            y_pred = self._sigmoid(z)                 # probabilities

            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Backward pass — compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)

            # Parameter update (gradient descent)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (i % self.verbose_interval == 0 or i == 1):
                print(f"  Iteration {i:>5}/{self.n_iterations}  |  Loss: {loss:.6f}")

        if self.verbose:
            print(f"{'='*55}")
            print(f"  Training complete. Final loss: {self.loss_history[-1]:.6f}")
            print(f"{'='*55}\n")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities P(y=1 | X).

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)

        Returns
        -------
        np.ndarray, shape (m,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        z = X.dot(self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary class predictions.

        Parameters
        ----------
        X         : np.ndarray, shape (m, n_features)
        threshold : float — decision boundary (default 0.5)

        Returns
        -------
        np.ndarray of int, shape (m,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted model to disk using pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Model] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LogisticRegression":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance.")
        print(f"[Model] Loaded from {path}")
        return model

    def __repr__(self) -> str:
        return (
            f"LogisticRegression("
            f"learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, "
            f"fitted={self.is_fitted})"
        )
