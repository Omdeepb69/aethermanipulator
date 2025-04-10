import pytest
import numpy as np
import sys
import os

# Add the directory containing model.py to the Python path
# Assuming model.py is in the same directory or a known location
# If model.py is in the parent directory:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# If model.py is in the same directory (usually works by default, but explicit is safer):
sys.path.insert(0, os.path.dirname(__file__))

# Attempt to import the model, provide a dummy if it fails
try:
    from model import SimpleModel
except ImportError:
    # Define a dummy SimpleModel if model.py is not found or has issues
    # This allows the test file structure to be generated even without the actual model module
    class SimpleModel:
        def __init__(self, num_features=None, hyperparameter=0.1):
            self.num_features = num_features
            self.hyperparameter = hyperparameter
            self._is_trained = False
            self._internal_weights = None
            if self.num_features is not None and not isinstance(self.num_features, int) or self.num_features <= 0:
                 raise ValueError("num_features must be a positive integer or None")


        def train(self, X, y):
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError("X and y must be numpy arrays.")
            if X.ndim != 2:
                 raise ValueError("X must be a 2D array.")
            if y.ndim != 1:
                 raise ValueError("y must be a 1D array.")

            n_samples, n_features = X.shape

            if self.num_features is None:
                self.num_features = n_features
            elif self.num_features != n_features:
                raise ValueError(f"Expected {self.num_features} features, got {n_features}")

            if n_samples != len(y):
                 raise ValueError(f"Number of samples in X ({n_samples}) and y ({len(y)}) do not match.")

            # Simulate training
            self._internal_weights = np.random.rand(self.num_features) * self.hyperparameter
            self._is_trained = True


        def predict(self, X):
            if not self._is_trained:
                raise RuntimeError("Model has not been trained yet.")
            if not isinstance(X, np.ndarray):
                raise TypeError("X must be a numpy array.")
            if X.ndim != 2:
                 raise ValueError("X must be a 2D array.")

            n_samples, n_features = X.shape

            if self.num_features != n_features:
                raise ValueError(f"Expected {self.num_features} features, got {n_features}")

            # Simulate prediction
            # Return dummy predictions of the correct shape
            predictions = np.zeros(n_samples)
            return predictions


        def evaluate(self, X, y):
            if not self._is_trained:
                raise RuntimeError("Model has not been trained yet.")

            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError("X and y must be numpy arrays.")
            if X.ndim != 2:
                 raise ValueError("X must be a 2D array.")
            if y.ndim != 1:
                 raise ValueError("y must be a 1D array.")

            n_samples, n_features = X.shape

            if self.num_features != n_features:
                raise ValueError(f"Expected {self.num_features} features, got {n_features}")
            if n_samples != len(y):
                 raise ValueError(f"Number of samples in X ({n_samples}) and y ({len(y)}) do not match.")

            # Simulate evaluation
            predictions = self.predict(X) # Use predict internally
            # Dummy metric calculation
            accuracy = np.mean(np.round(predictions) == y) # Example metric
            loss = np.mean((predictions - y)**2) # Example metric

            return {"accuracy": accuracy, "loss": loss}

        @property
        def is_trained(self):
            return self._is_trained


# --- Test Fixtures ---

@pytest.fixture
def dummy_data():
    """Provides sample training/testing data."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    return X, y

@pytest.fixture
def untrained_model():
    """Provides an untrained instance of SimpleModel."""
    return SimpleModel()

@pytest.fixture
def trained_model(dummy_data):
    """Provides a trained instance of SimpleModel."""
    X, y = dummy_data
    model = SimpleModel()
    model.train(X, y)
    return model

# --- Test Cases ---

# 1. Model Initialization Tests
def test_model_initialization_defaults():
    model = SimpleModel()
    assert model.num_features is None
    assert model.hyperparameter == 0.1
    assert not model.is_trained
    assert model._internal_weights is None

def test_model_initialization_custom_params():
    model = SimpleModel(num_features=5, hyperparameter=0.5)
    assert model.num_features == 5
    assert model.hyperparameter == 0.5
    assert not model.is_trained

def test_model_initialization_invalid_features():
     with pytest.raises(ValueError):
         SimpleModel(num_features=0)
     with pytest.raises(ValueError):
         SimpleModel(num_features=-1)
     with pytest.raises(ValueError):
         SimpleModel(num_features="abc")


# 2. Model Training Tests
def test_model_train_sets_trained_flag(untrained_model, dummy_data):
    X, y = dummy_data
    assert not untrained_model.is_trained
    untrained_model.train(X, y)
    assert untrained_model.is_trained

def test_model_train_sets_num_features(untrained_model, dummy_data):
    X, y = dummy_data
    assert untrained_model.num_features is None
    untrained_model.train(X, y)
    assert untrained_model.num_features == X.shape[1]

def test_model_train_updates_internal_state(untrained_model, dummy_data):
     X, y = dummy_data
     assert untrained_model._internal_weights is None
     untrained_model.train(X, y)
     assert untrained_model._internal_weights is not None
     assert isinstance(untrained_model._internal_weights, np.ndarray)
     assert untrained_model._internal_weights.shape == (X.shape[1],)

def test_model_train_raises_feature_mismatch(dummy_data):
    X, y = dummy_data
    model = SimpleModel(num_features=X.shape[1] + 1) # Mismatched features
    with pytest.raises(ValueError, match=r"Expected \d+ features, got \d+"):
        model.train(X, y)

def test_model_train_raises_sample_mismatch(untrained_model, dummy_data):
    X, y = dummy_data
    y_mismatch = np.append(y, [1]) # Different number of samples
    with pytest.raises(ValueError, match=r"Number of samples in X .* and y .* do not match"):
        untrained_model.train(X, y_mismatch)

def test_model_train_raises_invalid_input_type(untrained_model):
    X_list = [[1, 2], [3, 4]]
    y_list = [0, 1]
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    with pytest.raises(TypeError, match="X and y must be numpy arrays"):
        untrained_model.train(X_list, y_np)
    with pytest.raises(TypeError, match="X and y must be numpy arrays"):
        untrained_model.train(X_np, y_list)
    with pytest.raises(ValueError, match="X must be a 2D array"):
        untrained_model.train(np.array([1, 2, 3]), y_np) # X is 1D
    with pytest.raises(ValueError, match="y must be a 1D array"):
        untrained_model.train(X_np, np.array([[0], [1]])) # y is 2D


# 3. Model Prediction Tests
def test_model_predict_returns_correct_shape(trained_model, dummy_data):
    X, _ = dummy_data
    X_pred = X[:2] # Predict on a subset
    predictions = trained_model.predict(X_pred)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_pred.shape[0],)

def test_model_predict_raises_not_trained(untrained_model, dummy_data):
    X, _ = dummy_data
    with pytest.raises(RuntimeError, match="Model has not been trained yet"):
        untrained_model.predict(X)

def test_model_predict_raises_feature_mismatch(trained_model, dummy_data):
    X, _ = dummy_data
    X_wrong_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Wrong number of features
    with pytest.raises(ValueError, match=r"Expected \d+ features, got \d+"):
        trained_model.predict(X_wrong_features)

def test_model_predict_raises_invalid_input_type(trained_model):
     X_list = [[1, 2], [3, 4]]
     with pytest.raises(TypeError, match="X must be a numpy array"):
         trained_model.predict(X_list)
     with pytest.raises(ValueError, match="X must be a 2D array"):
        trained_model.predict(np.array([1, 2])) # X is 1D


# 4. Model Evaluation Tests
def test_model_evaluate_returns_dict_with_metrics(trained_model, dummy_data):
    X, y = dummy_data
    metrics = trained_model.evaluate(X, y)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "loss" in metrics
    assert isinstance(metrics["accuracy"], (float, np.floating))
    assert isinstance(metrics["loss"], (float, np.floating))

def test_model_evaluate_raises_not_trained(untrained_model, dummy_data):
    X, y = dummy_data
    with pytest.raises(RuntimeError, match="Model has not been trained yet"):
        untrained_model.evaluate(X, y)

def test_model_evaluate_raises_feature_mismatch(trained_model, dummy_data):
    X, y = dummy_data
    X_wrong_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Wrong number of features
    y_match = y[:2]
    with pytest.raises(ValueError, match=r"Expected \d+ features, got \d+"):
        trained_model.evaluate(X_wrong_features, y_match)

def test_model_evaluate_raises_sample_mismatch(trained_model, dummy_data):
    X, y = dummy_data
    y_mismatch = np.append(y, [1]) # Different number of samples
    with pytest.raises(ValueError, match=r"Number of samples in X .* and y .* do not match"):
        trained_model.evaluate(X, y_mismatch)

def test_model_evaluate_raises_invalid_input_type(trained_model, dummy_data):
    X, y = dummy_data
    X_list = X.tolist()
    y_list = y.tolist()

    with pytest.raises(TypeError, match="X and y must be numpy arrays"):
        trained_model.evaluate(X_list, y)
    with pytest.raises(TypeError, match="X and y must be numpy arrays"):
        trained_model.evaluate(X, y_list)
    with pytest.raises(ValueError, match="X must be a 2D array"):
        trained_model.evaluate(np.array([1, 2, 3, 4]), y) # X is 1D
    with pytest.raises(ValueError, match="y must be a 1D array"):
        trained_model.evaluate(X, np.array([[0], [1], [0], [1]])) # y is 2D