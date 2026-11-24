"""Milestone 6: why did the model give this woman this number?

    E17  Which inputs matter (permutation importance), what shape each curve has (partial dependence, ICE),
         whether SHAP agrees with the model's exact contributions, and whether a boosted model tells the
         same story.
    E18  Counterfactuals: the smallest change in one measurement that would move a flagged woman below the cut-off.

The published model is additive in log-odds (a starting value plus one contribution per input), so every
explanation below can also be computed exactly, and the exact answer is what SHAP and the counterfactual
search are checked against.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from scipy import stats

from . import nonlinear as nl
from .contest import corrected_ttest
from .evaluate import cv_splitter
from .modeling import SEED

PERM_REPEATS = 5
N_BOOT = 300
BMI_POINTS = (24.0, 30.0, 42.0)        # where the BMI curve is read: low, the start of the flat stretch, high
# "Realistic" limits for a counterfactual. They are deliberately modest: a glucose reading 20 mg/dL lower, or a
# BMI 10% lower (about 3 points at a BMI of 30). Blood pressure is not searched: the model barely uses it.
LIMITS = {"Glucose": 20.0, "BMI_share": 0.10}
STEP = {"Glucose": 1.0, "BMI": 0.1}


# ------------------------------------------------------------------ helpers
def fit_current(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clone(nl.curves_model()).fit(X, y)


def _logit(p):
    return float(np.log(p / (1 - p)))


def _medians(X):
    return X.median()


def contributions_exact(m: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Each woman's contribution per input, from the exported lookup tables (log-odds, relative to the median woman)."""
    filled = X.fillna(pd.Series(dict(zip(m["features"], m["median"]))))
    out = {}
    for f in m["features"]:
        out[f] = np.interp(filled[f].to_numpy(), m["grid"][f], m["contribution"][f])
    return pd.DataFrame(out, index=X.index)


# ------------------------------------------------------------------ E17a: permutation importance
def permutation_importance(X, y) -> dict:
    """How much does the AUC fall on held-out women when one input is shuffled?

    The model is fitted on each training set, then one input of the held-out women is shuffled (5 times) and the
    drop in AUC recorded. Over the 50 splits the drops give a mean and a spread. A bigger drop means the model
    leaned on that input more.
    """
    rng = np.random.default_rng(SEED)
    drops = {f: [] for f in X.columns}
    for train, test in cv_splitter().split(X, y):
        model = fit_current(X.iloc[train], y.iloc[train])
        Xt, yt = X.iloc[test], y.iloc[test].to_numpy()
        base = roc_auc_score(yt, model.predict_proba(Xt)[:, 1])
        for f in X.columns:
            d = []
            for _ in range(PERM_REPEATS):
                Xp = Xt.copy()
                Xp[f] = rng.permutation(Xp[f].to_numpy())
                d.append(base - roc_auc_score(yt, model.predict_proba(Xp)[:, 1]))
            drops[f].append(float(np.mean(d)))
    res = {}
    for f, v in drops.items():
        v = np.array(v)
        res[f] = {"mean": float(v.mean()), "low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5)),
                  "splits_with_a_drop": float((v > 0).mean())}
    return res


# ------------------------------------------------------------------ E17b: partial dependence and ICE
def partial_dependence(model, X, feature: str, grid=None, n_ice: int = 40):
    """Average predicted risk, and risk curves for individual women, as one input moves across its range.

    Everyone else keeps their own values (so this is the average over real women, not over a typical woman).
    Returns (grid, average risk at each grid point, ICE array of shape (n_ice, len(grid))).
    """
    if grid is None:
        lo, hi = X[feature].quantile(0.01), X[feature].quantile(0.99)
        grid = np.linspace(lo, hi, 40)
    rows = X.sample(n_ice, random_state=SEED)
    avg, ice = [], np.zeros((n_ice, len(grid)))
    for j, g in enumerate(grid):
        Xg = X.copy()
        Xg[feature] = g
        avg.append(float(model.predict_proba(Xg)[:, 1].mean()))
        Xi = rows.copy()
        Xi[feature] = g
        ice[:, j] = model.predict_proba(Xi)[:, 1]
    return np.asarray(grid), np.array(avg), ice


# ------------------------------------------------------------------ E17c: is the BMI plateau real?
def _bmi_contrast(model, X, points=BMI_POINTS) -> dict:
    """The BMI curve at three BMI values, with every other input at its median (log-odds)."""
    med = X.median()
    rows = pd.DataFrame([med.values] * len(points), columns=X.columns)
    rows["BMI"] = list(points)
    z = model.decision_function(rows)
    return {"low_to_start": float(z[1] - z[0]), "plateau": float(z[2] - z[1])}


def bmi_plateau(X, y) -> dict:
    """Is the flat stretch of the BMI curve (about 30 to 42) real, or noise?

    Two tests.
    1. Bootstrap: refit on 300 resamples of the women and read the rise in log-odds from BMI 24 to 30 and from
       30 to 42. If the plateau were real, the second rise would stay near zero; if it were noise, it would
       swing both ways and its interval would include a steady rise.
    2. Cross-validation: a model whose BMI is a straight line (curves only for glucose and age) scored against
       the published model on the same 50 splits, with the corrected t-test.
    """
    rng = np.random.default_rng(SEED)
    full = _bmi_contrast(fit_current(X, y), X)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X))
        Xb, yb = X.iloc[idx], y.iloc[idx]
        boot.append(_bmi_contrast(fit_current(Xb, yb), X))
    b = pd.DataFrame(boot)
    iv = {k: [float(np.percentile(b[k], 2.5)), float(np.percentile(b[k], 97.5))] for k in b}
    # a steady continuation of the 24-to-30 slope over the 12 BMI points from 30 to 42
    steady = full["low_to_start"] * (12 / 6)
    straight = nl.lr_pipeline(*nl._splines(["Glucose", "Age"]), c=0.3)
    cv = cv_splitter()
    auc_pub, auc_str = [], []
    for train, test in cv.split(X, y):
        yt = y.iloc[test].to_numpy()
        auc_pub.append(roc_auc_score(yt, fit_current(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])[:, 1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = clone(straight).fit(X.iloc[train], y.iloc[train])
        auc_str.append(roc_auc_score(yt, m.predict_proba(X.iloc[test])[:, 1]))
    n_train = int(len(X) * 0.8)
    test = corrected_ttest(auc_pub, auc_str, n_train, len(X) - n_train)
    return {"points": list(BMI_POINTS), "estimate": full, "interval": iv, "steady_rise_if_not_flat": steady,
            "odds": {"24_to_30": float(np.exp(full["low_to_start"])), "30_to_42": float(np.exp(full["plateau"]))},
            "curve_vs_straight_bmi": {"auc_curve": float(np.mean(auc_pub)), "auc_straight": float(np.mean(auc_str)), **test},
            "n_boot": N_BOOT}


# ------------------------------------------------------------------ E17d: SHAP against the exact answer
def shap_check(X, y) -> dict:
    """SHAP values for the published model, compared with its exact contributions.

    Because the model is additive in log-odds, the exact SHAP value of input j for a woman is her contribution
    from j minus the average contribution over the background women. SHAP (exact algorithm, 100 background
    women) should reproduce that to rounding error. This is a test of the explanation method, not of the model.
    """
    import shap  # optional dependency, imported on use
    model = fit_current(X, y)
    med = X.median()
    filled = X.fillna(med)
    background = filled.sample(100, random_state=SEED)

    def logit(data):
        return model.decision_function(pd.DataFrame(data, columns=X.columns))

    explainer = shap.Explainer(logit, background, algorithm="exact")
    values = explainer(filled, silent=True).values
    m = nl.export_curves_model(X, y)
    contrib = contributions_exact(m, X)
    expected = contrib.to_numpy() - contributions_exact(m, background).mean().to_numpy()
    diff = np.abs(values - expected)
    mean_abs = np.abs(values).mean(axis=0)
    return {"max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
            "mean_abs_shap": dict(zip(X.columns, map(float, mean_abs))),
            "share": dict(zip(X.columns, map(float, mean_abs / mean_abs.sum()))),
            "n_women": int(len(X)), "n_background": 100}


# ------------------------------------------------------------------ E17e: does a different kind of model tell the same story?
def boosted_story(X, y) -> dict:
    """SHAP importance for a boosted-tree model (XGBoost), next to the published model's.

    A tree model has no exact per-input contributions, so SHAP is the only way to read it. If it ranks the inputs
    the way the published model does, the explanation is about the data and not about the model type.
    """
    import shap
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=SEED, n_jobs=1).fit(X, y)
    values = shap.TreeExplainer(model).shap_values(X)
    boosted = np.abs(values).mean(axis=0)
    pub = shap_check_cached_share(X, y)
    pub_share = np.array([pub[f] for f in X.columns])
    boosted_share = boosted / boosted.sum()
    rho = float(stats.spearmanr(pub_share, boosted_share).statistic)
    return {"boosted_share": dict(zip(X.columns, map(float, boosted_share))),
            "published_share": dict(zip(X.columns, map(float, pub_share))), "spearman": rho,
            "top_boosted": str(X.columns[int(np.argmax(boosted))]), "top_published": str(X.columns[int(np.argmax(pub_share))])}


_SHARE_CACHE: dict = {}


def shap_check_cached_share(X, y) -> dict:
    """The published model's mean |SHAP| share per input, computed once per process (the exact algorithm is the slow part)."""
    key = (len(X), tuple(X.columns))
    if key not in _SHARE_CACHE:
        _SHARE_CACHE[key] = shap_check(X, y)["share"]
    return _SHARE_CACHE[key]


# ------------------------------------------------------------------ E18: counterfactuals
def _z(m: dict, values) -> float:
    """Log-odds for one woman from the lookup tables (the same arithmetic the web page does)."""
    z = m["base_logit"]
    for f, v in zip(m["features"], values):
        z += float(np.interp(v, m["grid"][f], m["contribution"][f]))
    return z


def counterfactual(m: dict, values, cutoff: float, feature: str, limit=None) -> dict:
    """The smallest drop in one measurement that moves a woman's risk below the cut-off.

    Searches downward from her value in steps (1 mg/dL of glucose, 0.1 of BMI), changing only that input and
    keeping the model's own clipping range. Returns {"needed": drop} when it works within `limit`,
    {"needed": None} when no drop within the limit is enough, and {"needed": 0.0} when she is already below.
    """
    values = list(values)
    i = m["features"].index(feature)
    target = _logit(cutoff)
    if _z(m, values) < target:
        return {"needed": 0.0, "new_value": values[i], "risk": 1 / (1 + np.exp(-_z(m, values)))}
    lo = m["range"][feature][0]
    step = STEP[feature]
    k = 1
    while True:
        drop = round(k * step, 4)
        new = values[i] - drop
        if new < lo or (limit is not None and drop > limit + 1e-9):
            return {"needed": None}
        trial = values.copy()
        trial[i] = new
        if _z(m, trial) < target:
            return {"needed": drop, "new_value": round(new, 4), "risk": 1 / (1 + np.exp(-_z(m, trial)))}
        k += 1


def counterfactual_study(X, y, cutoff: float) -> dict:
    """For every woman the model flags, can a realistic change in glucose or BMI clear her?

    Realistic means: glucose down by at most 20 mg/dL, or BMI down by at most 10%. Each input is changed on its own.
    The model's risks are associations in this group of women. A lower glucose reading is not a treatment, so these
    are explanations of the model, not advice.
    """
    m = nl.export_curves_model(X, y)
    filled = X.fillna(pd.Series(dict(zip(m["features"], m["median"]))))
    rows = filled[m["features"]].to_numpy()
    risk = np.array([1 / (1 + np.exp(-_z(m, r))) for r in rows])
    flagged = np.where(risk >= cutoff)[0]
    g_need, b_need, g_ok, b_ok, either, neither_rows = [], [], [], [], [], []
    for i in flagged:
        r = list(rows[i])
        g = counterfactual(m, r, cutoff, "Glucose", LIMITS["Glucose"])
        b = counterfactual(m, r, cutoff, "BMI", LIMITS["BMI_share"] * r[m["features"].index("BMI")])
        g_ok.append(g["needed"] is not None)
        b_ok.append(b["needed"] is not None)
        either.append(g_ok[-1] or b_ok[-1])
        if g_ok[-1]:
            g_need.append(g["needed"])
        if b_ok[-1]:
            b_need.append(b["needed"])
    n = len(flagged)
    # How far above the cut-off the flagged women sit, in risk: a woman at 90% will not be cleared by a small change.
    return {"cutoff": cutoff, "flagged": int(n), "share_flagged": float(n / len(rows)),
            "glucose_alone": float(np.mean(g_ok)), "bmi_alone": float(np.mean(b_ok)), "either": float(np.mean(either)),
            "median_glucose_drop": float(np.median(g_need)) if g_need else None,
            "median_bmi_drop": float(np.median(b_need)) if b_need else None,
            "limits": LIMITS,
            "flagged_risk_median": float(np.median(risk[flagged])),
            "cleared_by_either_when_risk_below_30": float(np.mean([e for e, i in zip(either, flagged) if risk[i] < 0.30]))
            if any(risk[i] < 0.30 for i in flagged) else None}


# ------------------------------------------------------------------ the whole milestone
def run(X, y, cutoffs: dict, default_ratio) -> dict:
    """Everything Milestone 6 reports, for reports/metrics.json.

    `cutoffs` maps each cost ratio (2, 5, 10) to its cut-off; the counterfactual study runs at each.
    """
    shap_res = shap_check(X, y)
    _SHARE_CACHE[(len(X), tuple(X.columns))] = shap_res["share"]
    return {
        "permutation": permutation_importance(X, y),
        "bmi_plateau": bmi_plateau(X, y),
        "shap": shap_res,
        "boosted": boosted_story(X, y),
        "counterfactual": {str(r): counterfactual_study(X, y, c) for r, c in cutoffs.items()},
        "default_ratio": default_ratio,
    }
