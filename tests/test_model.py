"""
Unit tests for LogisticRegression and StandardScaler.
"""

import os
import pickle
import numpy as np
import pytest

from src.model import LogisticRegression
from src.preprocess import StandardScaler


# ---------------------------------------------------------------------------
# LogisticRegression unit tests
# ---------------------------------------------------------------------------

class TestLogisticRegressionInit:
    def test_default_hyperparameters(self):
        model = LogisticRegression()
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.is_fitted is False
        assert model.weights is None

    def test_custom_hyperparameters(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        assert model.learning_rate == 0.1
        assert model.n_iterations == 200

    def test_repr(self):
        model = LogisticRegression()
        assert "LogisticRegression" in repr(model)
        assert "fitted=False" in repr(model)


class TestSigmoid:
    """Test the sigmoid activation function in isolation."""

    def test_sigmoid_at_zero(self):
        model = LogisticRegression()
        z = np.array([0.0])
        result = model._sigmoid(z)
        assert abs(result[0] - 0.5) < 1e-9

    def test_sigmoid_large_positive(self):
        model = LogisticRegression()
        result = model._sigmoid(np.array([100.0]))
        assert abs(result[0] - 1.0) < 1e-6

    def test_sigmoid_large_negative(self):
        model = LogisticRegression()
        result = model._sigmoid(np.array([-100.0]))
        assert abs(result[0] - 0.0) < 1e-6

    def test_sigmoid_output_range(self):
        model = LogisticRegression()
        z = np.linspace(-10, 10, 100)
        out = model._sigmoid(z)
        assert (out >= 0).all() and (out <= 1).all()

    def test_sigmoid_monotonic(self):
        model = LogisticRegression()
        z = np.linspace(-5, 5, 50)
        out = model._sigmoid(z)
        assert (np.diff(out) >= 0).all()


class TestLoss:
    def test_perfect_predictions_low_loss(self):
        model = LogisticRegression()
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.999, 0.001, 0.999, 0.001])
        loss = model._compute_loss(y_true, y_pred)
        assert loss < 0.01

    def test_bad_predictions_high_loss(self):
        model = LogisticRegression()
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.001, 0.999, 0.001, 0.999])
        loss = model._compute_loss(y_true, y_pred)
        assert loss > 5.0

    def test_loss_non_negative(self):
        rng = np.random.default_rng(1)
        model = LogisticRegression()
        y_true = rng.integers(0, 2, size=50).astype(float)
        y_pred = rng.uniform(0.01, 0.99, size=50)
        assert model._compute_loss(y_true, y_pred) >= 0


class TestFitPredict:
    def test_fit_returns_self(self, tiny_dataset):
        X_train, _, y_train, _ = tiny_dataset
        model = LogisticRegression(learning_rate=0.1, n_iterations=50, verbose=False)
        result = model.fit(X_train, y_train)
        assert result is model

    def test_weights_shape_after_fit(self, tiny_dataset):
        X_train, _, y_train, _ = tiny_dataset
        model = LogisticRegression(learning_rate=0.1, n_iterations=50, verbose=False)
        model.fit(X_train, y_train)
        assert model.weights.shape == (X_train.shape[1],)
        assert isinstance(model.bias, float)

    def test_is_fitted_flag(self, tiny_dataset):
        X_train, _, y_train, _ = tiny_dataset
        model = LogisticRegression(verbose=False)
        assert not model.is_fitted
        model.fit(X_train, y_train)
        assert model.is_fitted

    def test_loss_decreases(self, tiny_dataset):
        X_train, _, y_train, _ = tiny_dataset
        model = LogisticRegression(learning_rate=0.1, n_iterations=200, verbose=False)
        model.fit(X_train, y_train)
        assert model.loss_history[0] > model.loss_history[-1]

    def test_predict_returns_binary(self, trained_model, tiny_dataset):
        _, X_test, _, _ = tiny_dataset
        preds = trained_model.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_in_range(self, trained_model, tiny_dataset):
        _, X_test, _, _ = tiny_dataset
        probas = trained_model.predict_proba(X_test)
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_high_accuracy_on_separable_data(self, trained_model, tiny_dataset):
        _, X_test, _, y_test = tiny_dataset
        preds = trained_model.predict(X_test)
        accuracy = np.mean(preds == y_test)
        assert accuracy >= 0.90, f"Expected ≥90% accuracy, got {accuracy:.2f}"

    def test_predict_raises_if_not_fitted(self, tiny_dataset):
        _, X_test, _, _ = tiny_dataset
        model = LogisticRegression(verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X_test)

    def test_predict_proba_raises_if_not_fitted(self, tiny_dataset):
        _, X_test, _, _ = tiny_dataset
        model = LogisticRegression(verbose=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X_test)


class TestCustomThreshold:
    def test_threshold_zero_predicts_all_ones(self, trained_model, tiny_dataset):
        _, X_test, _, _ = tiny_dataset
        preds = trained_model.predict(X_test, threshold=0.0)
        assert (preds == 1).all()

    def test_threshold_one_predicts_all_zeros(self, trained_model, tiny_dataset):
        _, X_test, _, _ = tiny_dataset
        preds = trained_model.predict(X_test, threshold=1.0)
        assert (preds == 0).all()


class TestModelSerialization:
    def test_save_and_load(self, trained_model, tmp_path):
        path = str(tmp_path / "test_model.pkl")
        trained_model.save(path)
        assert os.path.exists(path)
        loaded = LogisticRegression.load(path)
        assert loaded.is_fitted
        np.testing.assert_array_almost_equal(
            loaded.weights, trained_model.weights
        )

    def test_unfitted_model_raises_on_save(self, tmp_path):
        model = LogisticRegression(verbose=False)
        with pytest.raises(RuntimeError, match="Cannot save"):
            model.save(str(tmp_path / "unfitted.pkl"))


# ---------------------------------------------------------------------------
# StandardScaler unit tests
# ---------------------------------------------------------------------------

class TestStandardScaler:
    def test_fit_stores_mean_std(self, tiny_dataset):
        X_train, _, _, _ = tiny_dataset
        scaler = StandardScaler()
        scaler.fit(X_train)
        assert scaler.mean_ is not None
        assert scaler.std_ is not None
        assert scaler.mean_.shape == (X_train.shape[1],)

    def test_transform_zero_mean(self, tiny_dataset):
        X_train, _, _, _ = tiny_dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        np.testing.assert_array_almost_equal(
            X_scaled.mean(axis=0), np.zeros(X_train.shape[1]), decimal=6
        )

    def test_transform_unit_variance(self, tiny_dataset):
        X_train, _, _, _ = tiny_dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        np.testing.assert_array_almost_equal(
            X_scaled.std(axis=0), np.ones(X_train.shape[1]), decimal=5
        )

    def test_transform_raises_if_not_fitted(self, tiny_dataset):
        X, _, _, _ = tiny_dataset
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(X)

    def test_inverse_transform_roundtrip(self, tiny_dataset):
        X_train, _, _, _ = tiny_dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_recovered, X_train, decimal=10)

    def test_save_and_load(self, fitted_scaler, tiny_dataset, tmp_path):
        X_train, _, _, _ = tiny_dataset
        path = str(tmp_path / "scaler.pkl")
        fitted_scaler.save(path)
        loaded = StandardScaler.load(path)
        np.testing.assert_array_equal(loaded.mean_, fitted_scaler.mean_)
        np.testing.assert_array_equal(loaded.std_, fitted_scaler.std_)

    def test_zero_variance_feature_handled(self):
        """A constant feature should not cause division by zero."""
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.isfinite(X_scaled).all()
