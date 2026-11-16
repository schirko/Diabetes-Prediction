"""Checks that guard the claims the project makes. Run with: python -m pytest"""
import json
import math

import numpy as np

from src.data import FEATURES, ROOT, ZERO_MEANS_MISSING, load_data, load_raw
from src.evaluate import confusion_at, export_model, pick_threshold
from src.modeling import LEFT_OUT, current_features, logistic


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


def test_current_model_leaves_out_insulin_and_skinfold():
    X, y = load_data()
    cols = current_features(FEATURES)
    assert len(cols) == 6 and not set(LEFT_OUT) & set(cols)
    m = export_model(X[cols], y)
    assert m["features"] == cols
    expected = logistic().fit(X[cols], y).predict_proba(X[cols])[:, 1]
    filled = X[cols].fillna(dict(zip(cols, m["median"]))).to_numpy()
    z = m["intercept"] + ((filled - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"])
    assert np.allclose(1 / (1 + np.exp(-z)), expected, atol=1e-9)


def test_curves_export_matches_scikit_learn():
    """The page runs the curves model from lookup tables; that must equal scikit-learn's answer."""
    from src import nonlinear as nl
    X, y = load_data()
    Xc = X[current_features(FEATURES)]
    m = nl.export_curves_model(Xc, y)
    expected = nl.curves_model().fit(Xc, y).predict_proba(Xc)[:, 1]
    got = nl.predict_from_export(m, Xc.fillna(Xc.median()).to_numpy())
    assert np.abs(got - expected).max() < 1e-4
    assert m["features"] == list(Xc.columns)


def test_every_missing_data_strategy_runs_and_never_sees_a_gap():
    from src.missing import strategies
    X, y = load_data()
    for name, model in strategies().items():
        p = model.fit(X.iloc[:300], y.iloc[:300]).predict_proba(X.iloc[300:400])[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), name


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
    assert m["current"]["threshold_screening"]["sensitivity"] >= m["target_sensitivity"]
    published = json.loads((ROOT / "reports" / "model.json").read_text())
    assert published["features"] == m["current"]["features"]
    assert m["threshold_default"]["tp"] + m["threshold_default"]["fn"] == m["cases"]
