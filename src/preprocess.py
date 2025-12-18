# preprocess.py
import joblib
import numpy as np
import os

BASE = os.path.dirname(__file__)
_SCALER_PATH = os.path.join(BASE, "models", "robust_scaler.pkl")

# Load scaler once (module import)
scaler = joblib.load(_SCALER_PATH)

def preprocess(feature_vector):
    """
    Accepts:
      - feature_vector: 1D numpy array or list of length n_features (same as training)
    Returns:
      - scaled: numpy array of shape (1, n_features) ready for model.predict
    Notes:
      - This avoids DataFrame/column-name checks by passing a numpy array to scaler.transform.
      - Ensures the number of features matches scaler.n_features_in_ and raises a clear error otherwise.
    """
    arr = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)

    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and arr.shape[1] != expected:
        raise ValueError(
            f"Feature length mismatch in preprocess(): got {arr.shape[1]} features, "
            f"but scaler expects {expected}. Ensure extract_features() produces the same "
            "number and order of features used in training."
        )

    scaled = scaler.transform(arr)
    return scaled
