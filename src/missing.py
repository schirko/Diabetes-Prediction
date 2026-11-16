"""Milestone 2: different ways of handling the missing readings.

Every strategy is a pipeline, so whatever it learns about filling gaps is
learned from the training fold only. All of them are scored on the same
50 cross-validation splits used everywhere else in the project.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (switches IterativeImputer on)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import FEATURES
from .evaluate import confusion_at, cv_splitter
from .modeling import SEED

WORST_TWO = ["Insulin", "SkinThickness"]      # 49% and 30% missing
BASELINE = "Median fill (Milestone 1)"


def _drop_worst(X):
    return X.drop(columns=WORST_TWO)


def _lr():
    return LogisticRegression(max_iter=1000)


def strategies() -> dict:
    """Seven ways to deal with the gaps. Six end in the same logistic regression."""
    return {
        BASELINE: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _lr()),
        "Median fill + 'was missing' flags": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), _lr()),
        "Nearest neighbours (10 most similar women)": make_pipeline(
            StandardScaler(), KNNImputer(n_neighbors=10), _lr()),
        "Iterative (predict each gap from the other columns)": make_pipeline(
            StandardScaler(), IterativeImputer(max_iter=15, random_state=SEED), _lr()),
        "Iterative + 'was missing' flags": make_pipeline(
            StandardScaler(), IterativeImputer(max_iter=15, random_state=SEED, add_indicator=True), _lr()),
        "Drop insulin and skinfold": make_pipeline(
            FunctionTransformer(_drop_worst), SimpleImputer(strategy="median"), StandardScaler(), _lr()),
        "Gradient boosting (uses gaps directly)": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED),
    }


def compare_strategies(X, y) -> pd.DataFrame:
    """Score each strategy on the 50 splits and compare it, split by split, with the baseline."""
    scores = {}
    for name, model in strategies().items():
        res = cross_validate(model, X, y, cv=cv_splitter(), scoring={"auc": "roc_auc", "brier": "neg_brier_score"},
                             n_jobs=-1)
        scores[name] = {"auc": res["test_auc"], "brier": -res["test_brier"]}
    base = scores[BASELINE]["auc"]
    rows = []
    for name, s in scores.items():
        diff = s["auc"] - base
        rows.append({
            "strategy": name,
            "auc": s["auc"].mean(), "auc_sd": s["auc"].std(), "brier": s["brier"].mean(),
            "auc_vs_baseline": diff.mean(),
            "splits_better_than_baseline": float((diff > 0).mean()),
        })
    return pd.DataFrame(rows).set_index("strategy")


def recovery_test(X, column: str = "Insulin", hide: float = 0.25, repeats: int = 20) -> pd.DataFrame:
    """Hide readings we DO have, ask each method to guess them, and measure the miss.

    This is the only way to check a fill method: we know the true answers.
    R2 of 0 means "no better than guessing the median"; 1 would be perfect.
    """
    known = X.index[X[column].notna()]
    rng = np.random.default_rng(SEED)
    scale = StandardScaler().fit(X)
    methods = {
        "Median": lambda: SimpleImputer(strategy="median"),
        "Nearest neighbours": lambda: KNNImputer(n_neighbors=10),
        "Iterative": lambda: IterativeImputer(max_iter=15, random_state=SEED),
    }
    col = FEATURES.index(column)
    out = {m: {"r2": [], "mae": []} for m in methods}
    for _ in range(repeats):
        hidden = rng.choice(known, int(len(known) * hide), replace=False)
        Xm = X.copy()
        Xm.loc[hidden, column] = np.nan
        truth = X.loc[hidden, column].to_numpy()
        for name, make in methods.items():
            filled = scale.inverse_transform(make().fit_transform(scale.transform(Xm)))
            guess = pd.Series(filled[:, col], index=X.index).loc[hidden].to_numpy()
            out[name]["r2"].append(r2_score(truth, guess))
            out[name]["mae"].append(np.abs(truth - guess).mean())
    return pd.DataFrame({m: {"r2": np.mean(v["r2"]), "typical_miss": np.mean(v["mae"])} for m, v in out.items()}).T


def multiple_imputation(X, y, draws: int = 30) -> pd.DataFrame:
    """Fill the gaps 30 different plausible ways and predict each woman every time.

    Each draw samples the missing values from their likely range instead of
    using one best guess. The folds stay fixed, so the only thing that changes
    between draws is what was filled in. Returns one column of held-out
    probabilities per draw.
    """
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    preds = np.zeros((len(y), draws))
    for d in range(draws):
        model = make_pipeline(
            StandardScaler(),
            IterativeImputer(max_iter=15, sample_posterior=True, random_state=d),
            _lr(),
        )
        for train, test in folds:
            preds[test, d] = model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])[:, 1]
    return pd.DataFrame(preds, index=X.index, columns=[f"draw_{d + 1}" for d in range(draws)])


def spread_summary(X, y, draws_df: pd.DataFrame, threshold: float) -> dict:
    """How far each woman's estimate moves across the draws, and who crosses the cut-off."""
    p = draws_df.to_numpy()
    lo, hi = np.percentile(p, 5, axis=1), np.percentile(p, 95, axis=1)
    width = (hi - lo) * 100                                   # in percentage points
    flagged_share = (p >= threshold).mean(axis=1)
    flips = (flagged_share > 0.1) & (flagged_share < 0.9)     # flagged in some draws, cleared in others
    groups = {
        "all readings present": X.notna().all(axis=1).to_numpy(),
        "insulin missing": X["Insulin"].isna().to_numpy(),
        "insulin and skinfold missing": (X["Insulin"].isna() & X["SkinThickness"].isna()).to_numpy(),
    }
    by_group = {
        g: {"women": int(m.sum()), "median_width": float(np.median(width[m])),
            "p90_width": float(np.percentile(width[m], 90)), "flip_share": float(flips[m].mean()),
            "flips": int(flips[m].sum())}
        for g, m in groups.items()
    }
    pooled = p.mean(axis=1)
    return {
        "draws": int(p.shape[1]), "threshold": float(threshold),
        "by_group": by_group, "flips_total": int(flips.sum()),
        "width": [round(float(w), 2) for w in width],
        "insulin_missing": [bool(v) for v in X["Insulin"].isna()],
        "pooled": confusion_at(y, pooled, threshold),
    }
