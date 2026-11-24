"""Milestone 4: a fair, tuned contest between model types.

Nested cross-validation:
  * OUTER splits (5 folds x 3 repeats = 15) score each model on women it never saw.
  * INSIDE each outer training set, Optuna tunes the model's settings using a
    second, INNER cross-validation (3 folds). The outer test women are never
    used for tuning, so tuning can't flatter the score.

Every model sees exactly the same outer splits, so differences can be tested
split by split with the corrected resampled t-test (Nadeau and Bengio, 2003).

The full contest takes several minutes, so results are saved to
reports/m4_contest.json and reloaded unless rerun=True.
"""
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import ROOT
from .modeling import SEED
from . import nonlinear as nl

CACHE = ROOT / "reports" / "m4_contest.json"
OUTER_FOLDS, OUTER_REPEATS, INNER_FOLDS, N_TRIALS = 5, 3, 3, 30
PUBLISHED = "Curves (M3, as published)"


def outer_splitter():
    return RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=SEED)


# ----------------------------------------------------------------- the contestants
def _scaled(model):
    """Median fill and rescale, then the model. Most model types need inputs on one scale."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def _curves(c=0.3, n_knots=4):
    return nl.lr_pipeline(*nl._splines(nl.CURVED_INPUTS, n_knots=n_knots), c=c)


def contestants() -> dict:
    """name -> (builder(params) -> unfitted model, search space(trial) -> params, or None if not tuned)."""
    return {
        PUBLISHED: (lambda p: _curves(), None),
        "Curves, tuned": (
            lambda p: _curves(p["C"], p["n_knots"]),
            lambda t: {"C": t.suggest_float("C", 0.01, 10, log=True), "n_knots": t.suggest_int("n_knots", 3, 7)}),
        "Straight lines, tuned": (
            lambda p: nl.lr_pipeline(c=p["C"]),
            lambda t: {"C": t.suggest_float("C", 1e-3, 100, log=True)}),
        "k-nearest neighbours": (
            lambda p: _scaled(KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"])),
            lambda t: {"k": t.suggest_int("k", 5, 75), "weights": t.suggest_categorical("weights", ["uniform", "distance"])}),
        "Support vector machine": (
            lambda p: _scaled(SVC(C=p["C"], gamma=p["gamma"], probability=True, random_state=SEED)),
            lambda t: {"C": t.suggest_float("C", 0.01, 100, log=True), "gamma": t.suggest_float("gamma", 1e-3, 1, log=True)}),
        "Random forest": (
            lambda p: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
                n_estimators=300, max_depth=p["max_depth"], min_samples_leaf=p["min_leaf"],
                max_features=p["max_features"], random_state=SEED, n_jobs=1)),
            lambda t: {"max_depth": t.suggest_int("max_depth", 2, 12), "min_leaf": t.suggest_int("min_leaf", 1, 30),
                       "max_features": t.suggest_float("max_features", 0.3, 1.0)}),
        "Gradient boosting": (
            lambda p: HistGradientBoostingClassifier(
                learning_rate=p["lr"], max_depth=p["depth"], max_iter=p["iters"], l2_regularization=p["l2"],
                min_samples_leaf=p["min_leaf"], random_state=SEED),
            lambda t: {"lr": t.suggest_float("lr", 0.01, 0.3, log=True), "depth": t.suggest_int("depth", 2, 6),
                       "iters": t.suggest_int("iters", 50, 400), "l2": t.suggest_float("l2", 0.0, 10.0),
                       "min_leaf": t.suggest_int("min_leaf", 5, 60)}),
        "XGBoost": (
            lambda p: _xgb(p),
            lambda t: {"n": t.suggest_int("n", 50, 400), "depth": t.suggest_int("depth", 2, 6),
                       "lr": t.suggest_float("lr", 0.01, 0.3, log=True), "subsample": t.suggest_float("subsample", 0.6, 1.0),
                       "colsample": t.suggest_float("colsample", 0.5, 1.0),
                       "min_child": t.suggest_float("min_child", 1, 20, log=True),
                       "reg_lambda": t.suggest_float("reg_lambda", 0.1, 10, log=True)}),
        "Neural network": (
            lambda p: _scaled(MLPClassifier(hidden_layer_sizes=(p["units"],), alpha=p["alpha"],
                                            learning_rate_init=p["lr"], max_iter=600, early_stopping=True,
                                            n_iter_no_change=20, random_state=SEED)),
            lambda t: {"units": t.suggest_int("units", 4, 64, log=True), "alpha": t.suggest_float("alpha", 1e-5, 1, log=True),
                       "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True)}),
        "Stacked ensemble": (lambda p: stacked_ensemble(), None),
    }


def _xgb(p):
    from xgboost import XGBClassifier  # optional dependency, imported on use
    # XGBoost accepts missing values directly, so it gets no fill step.
    return XGBClassifier(n_estimators=p["n"], max_depth=p["depth"], learning_rate=p["lr"], subsample=p["subsample"],
                         colsample_bytree=p["colsample"], min_child_weight=p["min_child"], reg_lambda=p["reg_lambda"],
                         eval_metric="logloss", random_state=SEED, n_jobs=1)


def stacked_ensemble():
    """Four different model types combined by a logistic regression that learns how much to trust each (E14).

    The members use fixed, sensible settings chosen before the contest, so the
    ensemble is not tuned on the test women either.
    """
    members = [
        ("curves", _curves()),
        ("svm", _scaled(SVC(C=1.0, probability=True, random_state=SEED))),
        ("forest", make_pipeline(SimpleImputer(strategy="median"),
                                 RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=1))),
        ("boosting", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)),
    ]
    return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000), cv=5,
                              stack_method="predict_proba")


# ----------------------------------------------------------------- one outer split
def _tune(build, space, X, y, seed):
    """Optuna search on the training women only, scored by inner cross-validation AUC."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cross_val_score(build(space(trial)), X, y, cv=inner, scoring="roc_auc").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def _one_split(name, X, y, train, test, split_id):
    build, space = contestants()[name]
    Xtr, ytr = X.iloc[train], y.iloc[train]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = _tune(build, space, Xtr, ytr, SEED + split_id) if space else {}
        model = build(params).fit(Xtr, ytr)
        p = model.predict_proba(X.iloc[test])[:, 1]
    return split_id, test, p, params


def run_model(name, X, y):
    """Nested cross-validation for one contestant."""
    splits = list(outer_splitter().split(X, y))
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(_one_split)(name, X, y, tr, te, i) for i, (tr, te) in enumerate(splits))
    results.sort(key=lambda r: r[0])
    yv = y.to_numpy()
    total, count = np.zeros(len(y)), np.zeros(len(y))
    for _, te, p, _ in results:
        total[te] += p
        count[te] += 1
    return {
        "auc": [float(roc_auc_score(yv[te], p)) for _, te, p, _ in results],
        "brier": [float(brier_score_loss(yv[te], p)) for _, te, p, _ in results],
        "oof": [round(float(v), 6) for v in total / count],
        "params": [params for *_, params in results],
        "seconds": round(time.time() - start, 1),
    }


def run_contest(X, y, rerun: bool = False, names=None) -> dict:
    """Run (or reload) the whole contest. Returns {model name: results}."""
    names = names or list(contestants())
    if CACHE.exists() and not rerun:
        saved = json.loads(CACHE.read_text())
        if saved.get("config") == _config() and all(n in saved["models"] for n in names):
            return {n: saved["models"][n] for n in names}
    out = {}
    for name in names:
        out[name] = run_model(name, X, y)
        print(f"{name:28s} AUC {np.mean(out[name]['auc']):.4f}  ({out[name]['seconds']:.0f}s)")
    CACHE.write_text(json.dumps({"config": _config(), "models": out}))
    return out


def _config():
    return {"outer": [OUTER_FOLDS, OUTER_REPEATS], "inner": INNER_FOLDS, "trials": N_TRIALS, "seed": SEED}


# ----------------------------------------------------------------- comparing fairly
def corrected_ttest(a, b, n_train: int, n_test: int) -> dict:
    """Is model a better than model b? Nadeau-Bengio corrected resampled t-test on paired split scores.

    An ordinary t-test treats the 15 splits as independent, but they reuse the
    same women, so it is far too confident. The correction inflates the variance
    by n_test / n_train to allow for that overlap.
    """
    d = np.asarray(a) - np.asarray(b)
    j = len(d)
    var = (1 / j + n_test / n_train) * d.var(ddof=1)
    se = np.sqrt(var) if var > 0 else 1e-12
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    half = stats.t.ppf(0.975, df=j - 1) * se
    return {"diff": float(d.mean()), "low": float(d.mean() - half), "high": float(d.mean() + half),
            "p": float(p), "wins": float((d > 0).mean())}


def league_table(results: dict, X, y) -> pd.DataFrame:
    """One row per model: score, interval, and the formal comparison with the published model."""
    n = len(y)
    n_test = n / OUTER_FOLDS
    n_train = n - n_test
    base = results[PUBLISHED]["auc"]
    older = (X["Age"] >= 45).to_numpy()
    yv = y.to_numpy()
    rows = []
    for name, r in results.items():
        auc = np.asarray(r["auc"])
        own = corrected_ttest(auc, np.zeros_like(auc), n_train, n_test)   # interval for the model's own AUC
        cmp = corrected_ttest(auc, base, n_train, n_test)
        oof = np.asarray(r["oof"])
        rows.append({
            "model": name, "auc": auc.mean(), "auc_low": own["low"], "auc_high": own["high"],
            "brier": float(np.mean(r["brier"])), "auc_45plus": float(roc_auc_score(yv[older], oof[older])),
            "vs_published": cmp["diff"], "vs_low": cmp["low"], "vs_high": cmp["high"], "p_value": cmp["p"],
            "wins": cmp["wins"], "seconds": r["seconds"],
        })
    return pd.DataFrame(rows).set_index("model").sort_values("auc", ascending=False)


def tuned_settings(results: dict, name: str) -> pd.DataFrame:
    """The settings Optuna chose in each outer split: are they stable, or all over the place?"""
    return pd.DataFrame(results[name]["params"]).round(4)
