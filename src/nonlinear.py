"""Milestone 3: curved effects, interactions and models built for them.

Every candidate uses the six inputs of the current model (D09), sits inside a
pipeline, and is scored on the same 50 splits as everything else in the project.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

from .evaluate import confusion_at, cv_splitter
from .modeling import SEED

BASELINE = "Straight lines (Milestone 2)"
CURVES = "Curves for glucose, BMI and age"
AGE_BANDS = [("21 to 29", 21, 29), ("30 to 44", 30, 44), ("45 and over", 45, 200)]
CURVED_INPUTS = ["Glucose", "BMI", "Age"]


# ----------------------------------------------------------------- building blocks
def _imputer():
    return SimpleImputer(strategy="median").set_output(transform="pandas")


def _add_age_x_glucose(X: pd.DataFrame) -> pd.DataFrame:
    """Age and glucose multiplied (after centring), so the effect of one can depend on the other."""
    X = X.copy()
    X["Age_x_Glucose"] = (X["Age"] - 33) * (X["Glucose"] - 120) / 100
    return X


def _log_skewed(X: pd.DataFrame) -> pd.DataFrame:
    """Log scale for the three right-skewed inputs (a few very large values)."""
    X = X.copy()
    for col in ["Pregnancies", "DiabetesPedigreeFunction", "Age"]:
        X[col] = np.log1p(X[col])
    return X


class ClipToRange(BaseEstimator, TransformerMixin):
    """Hold extreme values at the 1st and 99th percentiles of the training rows.

    Without this, a curve fitted to a handful of women at the edges (a BMI over 50,
    a glucose under 70) can swing wildly: before clipping, the BMI curve multiplied
    the odds by 82 at the top of its range. Clipping keeps each curve flat where
    there is almost no data.
    """

    def __init__(self, columns, low: float = 0.01, high: float = 0.99):
        self.columns, self.low, self.high = columns, low, high

    def fit(self, X, y=None):
        self.bounds_ = {c: (X[c].quantile(self.low), X[c].quantile(self.high)) for c in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X


def _splines(columns, n_knots: int = 4):
    """Curves (cubic splines) for the named columns, straight lines for the rest.

    The columns are clipped to their 1st to 99th percentile range first.
    """
    return [ClipToRange(columns),
            ColumnTransformer(
                [("curve", SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear"), columns)],
                remainder="passthrough")]


def lr_pipeline(*steps, c: float = 1.0):
    """Median fill, optional reshaping steps, rescale, logistic regression."""
    return make_pipeline(_imputer(), *steps, StandardScaler(), LogisticRegression(max_iter=2000, C=c))


def curves_model():
    """The Milestone 3 winner: curves for glucose, BMI and age, straight lines for the rest.

    C=0.3 adds a little shrinkage so the curves can't chase noise; it was set once, not tuned.
    Extreme values are held at the 1st and 99th percentiles before the curves (ClipToRange).
    """
    return lr_pipeline(*_splines(CURVED_INPUTS), c=0.3)


def _ebm(interactions: int):
    from interpret.glassbox import ExplainableBoostingClassifier  # optional dependency, imported on use
    # outer_bags=4 (default 14) keeps 50 fits to a few minutes
    return ExplainableBoostingClassifier(interactions=interactions, outer_bags=4, random_state=SEED, n_jobs=1)


def candidates(columns) -> dict:
    """Every Milestone 3 model for the given input columns, keyed by the name shown in tables."""
    rising = {"Glucose", "BMI", "DiabetesPedigreeFunction"}
    return {
        BASELINE: lr_pipeline(),
        "Log scale for skewed inputs": lr_pipeline(FunctionTransformer(_log_skewed)),
        "Curve for age only": lr_pipeline(*_splines(["Age"])),
        CURVES: curves_model(),
        "Age x glucose interaction": lr_pipeline(FunctionTransformer(_add_age_x_glucose)),
        "Curves + interaction": lr_pipeline(FunctionTransformer(_add_age_x_glucose), *_splines(CURVED_INPUTS), c=0.3),
        "Explainable Boosting Machine": _ebm(interactions=0),
        "EBM with pairwise interactions": _ebm(interactions=5),
        # Risk may only rise with glucose, BMI and family history. Age and pregnancies are left
        # free because the data shows risk falling again after 55.
        "Boosting, risk can only rise": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED,
            monotonic_cst=[1 if c in rising else 0 for c in columns]),
    }


# ----------------------------------------------------------------- scoring
def age_band_masks(X):
    return {lab: ((X["Age"] >= lo) & (X["Age"] <= hi)).to_numpy() for lab, lo, hi in AGE_BANDS}


def band_specificity(y, p, mask, target: float = 0.85) -> float:
    """Specificity inside one group when its own cut-off catches `target` of its cases."""
    yb, pb = np.asarray(y)[mask], np.asarray(p)[mask]
    for t in np.arange(0.99, 0.0, -0.01):
        c = confusion_at(yb, pb, round(t, 2))
        if c["sensitivity"] >= target:
            return c["specificity"]
    return 0.0


def _fit_split(model, X, y, train, test):
    with warnings.catch_warnings():          # runs in a worker process, so the filter is set here
        warnings.simplefilter("ignore")
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
    return test, fitted.predict_proba(X.iloc[test])[:, 1]


def run_splits(model, X, y, cv=None):
    """Fit once per split. Returns (AUC per split, Brier per split, held-out probability per woman).

    The held-out probability is the average of each woman's predictions, one per repeat.
    """
    splits = list((cv or cv_splitter()).split(X, y))
    results = Parallel(n_jobs=-1)(delayed(_fit_split)(model, X, y, tr, te) for tr, te in splits)
    yv = y.to_numpy()
    aucs = np.array([roc_auc_score(yv[te], p) for te, p in results])
    briers = np.array([brier_score_loss(yv[te], p) for te, p in results])
    total, count = np.zeros(len(y)), np.zeros(len(y))
    for te, p in results:
        total[te] += p
        count[te] += 1
    return aucs, briers, total / count


def compare(X, y, names=None, cv=None) -> tuple:
    """Score candidates on the same splits. Returns (summary table, held-out probabilities per model).

    Columns:
      auc, auc_sd, brier            the usual scores over 50 splits
      auc_vs_baseline, wins         paired with the Milestone 2 model, split by split
      auc_21 / auc_30 / auc_45plus  AUC inside each age band, from held-out predictions
      spec_45plus                   healthy women 45+ cleared when that band's own
                                    cut-off catches 85% of its cases
    """
    warnings.filterwarnings("ignore")
    models = candidates(list(X.columns))
    names = names or list(models)
    bands = age_band_masks(X)
    base_auc, rows, probs = None, [], {}
    for name in names:
        auc, brier, p = run_splits(models[name], X, y, cv)
        probs[name] = p
        base_auc = auc if base_auc is None else base_auc
        row = {"model": name, "auc": auc.mean(), "auc_sd": auc.std(), "brier": brier.mean(),
               "auc_vs_baseline": (auc - base_auc).mean(), "wins": float((auc > base_auc).mean())}
        for (lab, m), short in zip(bands.items(), ["21", "30", "45plus"]):
            row[f"auc_{short}"] = roc_auc_score(y[m], p[m])
        row["spec_45plus"] = band_specificity(y, p, bands["45 and over"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model"), probs


def calibration_by_age(X, y, p) -> pd.DataFrame:
    """Average predicted risk against the real rate, in five-year age groups."""
    groups = pd.cut(X["Age"], [20, 24, 29, 34, 39, 44, 49, 54, 59, 81],
                    labels=["21-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"])
    return pd.DataFrame({"women": y.groupby(groups, observed=False).size(),
                         "actual": y.groupby(groups, observed=False).mean(),
                         "predicted": pd.Series(p, index=y.index).groupby(groups, observed=False).mean()})


def bootstrap_band_gain(y, p_new, p_old, mask, n: int = 1000) -> list:
    """95% interval for the change in band specificity, resampling the women in that band."""
    rng = np.random.default_rng(SEED)
    idx_all, yv = np.where(mask)[0], np.asarray(y)
    out = []
    for _ in range(n):
        idx = rng.choice(idx_all, len(idx_all))
        if yv[idx].min() == yv[idx].max():
            continue
        # band_specificity takes a mask; build the resample explicitly to keep duplicates
        yy, a, b = yv[idx], np.asarray(p_new)[idx], np.asarray(p_old)[idx]
        ones = np.ones(len(idx), bool)
        out.append(band_specificity(yy, a, ones) - band_specificity(yy, b, ones))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


# ----------------------------------------------------------------- E29: insulin's second hearing
def insulin_retest(X_all, y) -> pd.DataFrame:
    """On the 394 women with a real insulin reading only: does insulin add anything?

    Compares the six-input model with three ways of adding insulin back, all on the
    same 10 x 5-fold splits of that subset.
    """
    warnings.filterwarnings("ignore")
    keep = X_all["Insulin"].notna()
    Xs, ys = X_all[keep].copy(), y[keep]
    six = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
    Xs["log_insulin"] = np.log(Xs["Insulin"])
    Xs["glucose_x_insulin"] = np.log(Xs["Glucose"] * Xs["Insulin"])  # rough insulin-resistance score
    variants = {
        "Six inputs (current)": six,
        "+ insulin": six + ["Insulin"],
        "+ log insulin": six + ["log_insulin"],
        "+ log(glucose x insulin)": six + ["glucose_x_insulin"],
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    base, rows = None, []
    for name, cols in variants.items():
        auc, _, _ = run_splits(lr_pipeline(), Xs[cols], ys, cv)
        base = auc if base is None else base
        rows.append({"variant": name, "women": int(len(ys)), "auc": auc.mean(), "auc_sd": auc.std(),
                     "auc_vs_six": (auc - base).mean(), "wins": float((auc > base).mean())})
    return pd.DataFrame(rows).set_index("variant")


# ----------------------------------------------------------------- export for the web page
GRID_STEP = {"Pregnancies": 1, "Glucose": 1, "BloodPressure": 1, "BMI": 0.1,
             "DiabetesPedigreeFunction": 0.01, "Age": 1}


def export_curves_model(X, y) -> dict:
    """The published model (curves for glucose, BMI and age), as lookup tables for the web page."""
    return export_additive(curves_model(), X, y)


def export_additive(model, X, y) -> dict:
    """Fit an additive model on every row and flatten it into lookup tables.

    Additive means its log-odds are a starting value plus one contribution per
    input, with no input changing another's effect. Each input's contribution is
    tabulated on a fine grid (relative to a typical woman, who has the median of
    every measurement), and the web page interpolates between grid points.
    A missing value takes the median first, so it contributes nothing.
    """
    features = list(X.columns)
    model = clone(model).fit(X, y)
    medians = X.median()
    grids, contribs = {}, {}
    # Contribution of one input = change in log-odds when only that input moves away from its median.
    base_row = pd.DataFrame([medians.values], columns=features)
    base_logit = float(model.decision_function(base_row)[0])
    for f in features:
        lo, hi = float(X[f].min()), float(X[f].max())
        grid = np.round(np.arange(lo, hi + GRID_STEP[f] / 2, GRID_STEP[f]), 4)
        rows = pd.DataFrame(np.repeat(base_row.values, len(grid), axis=0), columns=features)
        rows[f] = grid
        contribs[f] = [round(float(v), 6) for v in model.decision_function(rows) - base_logit]
        grids[f] = [float(v) for v in grid]
    filled = X.fillna(medians)
    pick = [0, 1, 4]
    return {
        "kind": "additive_lookup",
        "features": features,
        "median": [float(medians[f]) for f in features],
        "mean": [float(X[f].mean()) for f in features],
        "base_logit": base_logit,
        "grid": grids,
        "contribution": contribs,
        "range": {f: [float(X[f].min()), float(X[f].max())] for f in features},
        "check": {"inputs": filled.iloc[pick].to_numpy().tolist(),
                  "probabilities": model.predict_proba(X.iloc[pick])[:, 1].tolist()},
    }


def predict_from_export(m: dict, rows) -> np.ndarray:
    """The web page's arithmetic, in Python, so tests can compare it with scikit-learn."""
    out = []
    for row in np.atleast_2d(rows):
        z = m["base_logit"]
        for f, v in zip(m["features"], row):
            g, c = np.array(m["grid"][f]), np.array(m["contribution"][f])
            z += float(np.interp(v, g, c))
        out.append(1 / (1 + np.exp(-z)))
    return np.array(out)
