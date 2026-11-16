"""Checks that guard the claims the project makes. Run with: python -m pytest"""
import json
import math

import numpy as np

from src.data import FEATURES, ROOT, ZERO_MEANS_MISSING, load_data, load_raw
from src.evaluate import confusion_at, export_model, pick_threshold
from src.modeling import logistic


def test_zeros_become_missing():
    X, y = load_data()
    raw = load_raw()
    assert X.shape == (768, 8) and int(y.sum()) == 268
    for col in ZERO_MEANS_MISSING:
        assert X[col].isna().sum() == (raw[col] == 0).sum()
        assert (X[col].dropna() > 0).all()
    assert X["Pregnancies"].isna().sum() == 0          # zero pregnancies is a real value


def test_exported_model_matches_scikit_learn():
    """The web page rebuilds predictions from model.json; that maths must equal the pipeline's."""
    X, y = load_data()
    m = export_model(X, y)
    pipe = logistic().fit(X, y)
    expected = pipe.predict_proba(X)[:, 1]
    filled = X.fillna(dict(zip(FEATURES, m["median"]))).to_numpy()
    z = m["intercept"] + ((filled - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"])
    assert np.allclose(1 / (1 + np.exp(-z)), expected, atol=1e-9)


def test_confusion_counts_add_up():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.5])
    c = confusion_at(y, p, 0.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (2, 1, 1, 1)
    assert math.isclose(c["sensitivity"], 2 / 3)


def test_saved_threshold_meets_the_screening_target():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        return
    m = json.loads(path.read_text())
    assert m["threshold_screening"]["sensitivity"] >= m["target_sensitivity"]
    assert m["threshold_default"]["tp"] + m["threshold_default"]["fn"] == m["cases"]
