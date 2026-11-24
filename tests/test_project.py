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


def test_corrected_ttest():
    """With no correction term it must equal an ordinary paired t-test; the correction can only widen it."""
    from scipy import stats
    from src.contest import corrected_ttest
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.85, 0.02, 15), rng.normal(0.84, 0.02, 15)
    plain = corrected_ttest(a, b, n_train=1, n_test=0)
    assert math.isclose(plain["p"], stats.ttest_rel(a, b).pvalue, rel_tol=1e-9)
    assert corrected_ttest(a, b, n_train=614, n_test=154)["p"] > plain["p"]
    assert corrected_ttest(a, a, 614, 154)["diff"] == 0


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


def test_cost_based_cutoff_formula():
    from src import decision
    assert decision.cutoff_for_ratio(5) == 0.17
    assert decision.cutoff_for_ratio(2) == 0.33
    assert decision.cutoff_for_ratio(10) == 0.09
    assert abs(decision.implied_ratio(0.28) - 18 / 7) < 1e-9


def test_published_per_100_counts_add_up():
    m5 = json.loads((ROOT / "reports" / "metrics.json").read_text())["m5"]
    for v in m5["costs"]["ratios"].values():
        assert sum(v["per_100"][k] for k in ("tp", "fn", "fp", "tn")) == 100
    p = m5["published"]
    assert sum(p["per_100"][k] for k in ("tp", "fn", "fp", "tn")) == 100


def test_cost_based_cutoff_is_near_the_cheapest():
    m5 = json.loads((ROOT / "reports" / "metrics.json").read_text())["m5"]
    for v in m5["costs"]["ratios"].values():
        low, high = v["empirical_best"]["low"], v["empirical_best"]["high"]
        assert low - 1e-9 <= v["cutoff"] <= high + 1e-9


def test_exact_contributions_add_up_to_the_prediction():
    from src import explain
    from src import nonlinear as nl
    from src.modeling import current_features
    X, y = load_data()
    Xc = X[current_features(FEATURES)]
    m = nl.export_curves_model(Xc, y)
    contrib = explain.contributions_exact(m, Xc.head(20))
    z = m["base_logit"] + contrib.sum(axis=1).to_numpy()
    p = nl.predict_from_export(m, Xc.head(20).fillna(Xc.median()).to_numpy())
    assert np.allclose(1 / (1 + np.exp(-z)), p, atol=1e-9)


def test_counterfactual_is_the_smallest_change_that_clears_her():
    from src import explain
    from src import nonlinear as nl
    from src.modeling import current_features
    X, y = load_data()
    Xc = X[current_features(FEATURES)]
    m = nl.export_curves_model(Xc, y)
    woman = {"Pregnancies": 2, "Glucose": 122, "BloodPressure": 72.0, "BMI": 31.0, "DiabetesPedigreeFunction": 0.45, "Age": 30}
    vals = [woman[f] for f in m["features"]]
    cut = 0.33
    out = explain.counterfactual(m, vals, cut, "Glucose", 20.0)
    assert out["needed"] == 10.0 and out["risk"] < cut
    one_less = vals.copy()
    one_less[m["features"].index("Glucose")] -= out["needed"] - 1
    assert explain._z(m, one_less) >= np.log(cut / (1 - cut))      # one step earlier she is still flagged
    assert explain.counterfactual(m, vals, 0.17, "Glucose", 20.0)["needed"] is None


def test_shap_matches_the_exact_contributions():
    m6 = json.loads((ROOT / "reports" / "metrics.json").read_text())["m6"]
    assert m6["shap"]["max_abs_difference"] < 0.01
    assert m6["boosted"]["spearman"] > 0.8
