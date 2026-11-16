"""Run the whole evaluation and write the results the notebooks and site read.

    python -m src.evaluate

Writes:
    reports/metrics.json   every number quoted on the project page
    reports/oof.csv        one out-of-fold probability per woman
    reports/model.json     the current model in a form JavaScript can run
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

from .data import FEATURES, ROOT, load_data, load_raw
from .modeling import SEED, candidates, columns_for, current_features, logistic

REPORTS = ROOT / "reports"
N_SPLITS, N_REPEATS = 5, 10
TARGET_SENSITIVITY = 0.85  # screening goal: catch at least 85% of true cases
SCORING = {
    "auc": "roc_auc",
    "avg_precision": "average_precision",
    "brier": "neg_brier_score",
    "accuracy": "accuracy",
    "sensitivity": "recall",
}


def cv_splitter():
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


def compare_models(X, y) -> pd.DataFrame:
    """Score every candidate on the same 50 train/test splits."""
    rows = []
    for name, model in candidates().items():
        res = cross_validate(model, X[columns_for(name, FEATURES)], y, cv=cv_splitter(), scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for key in SCORING:
            vals = res[f"test_{key}"] * (-1 if key == "brier" else 1)
            row[key] = vals.mean()
            row[f"{key}_sd"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def out_of_fold(model, X, y) -> np.ndarray:
    """One honest probability per row: the average of its 10 held-out predictions."""
    total = np.zeros(len(y))
    for train, test in cv_splitter().split(X, y):
        fitted = model.fit(X.iloc[train], y.iloc[train])
        total[test] += fitted.predict_proba(X.iloc[test])[:, 1]
    return total / N_REPEATS


def lucky_split_demo(n_seeds: int = 200) -> dict:
    """Repeat the original notebook's recipe with 200 different random splits.

    Original recipe: fill zeros using the whole table, keep four columns,
    one 80/20 split (random_state=0), plain logistic regression, report accuracy.
    """
    d = load_raw().astype(float)
    d["Glucose"] = d["Glucose"].replace(0, d["Glucose"].mean())
    d["BMI"] = d["BMI"].replace(0, d["BMI"].median())
    X4, y = d[["Pregnancies", "Glucose", "BMI", "Age"]], d["Outcome"].astype(int)
    accs = []
    for seed in range(n_seeds):
        a, b, c, e = train_test_split(X4, y, test_size=0.2, random_state=seed)
        accs.append(LogisticRegression(max_iter=1000).fit(a, c).score(b, e))
    accs = np.array(accs)
    return {
        "original_seed_accuracy": float(accs[0]),
        "mean": float(accs.mean()),
        "min": float(accs.min()),
        "max": float(accs.max()),
        "share_below_original": float((accs < accs[0]).mean()),
        "all": [round(float(a), 4) for a in accs],
    }


def confusion_at(y, p, threshold: float) -> dict:
    """Counts and rates if everyone at or above the threshold is flagged."""
    flag = p >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else None,
        "npv": tn / (tn + fn) if tn + fn else None,
        "flagged_share": (tp + fp) / len(y),
        "accuracy": (tp + tn) / len(y),
    }


def pick_threshold(y, p, target: float = TARGET_SENSITIVITY) -> float:
    """Highest cut-off (to 2 decimals) that still catches the target share of cases."""
    for t in np.arange(0.99, 0.0, -0.01):
        if confusion_at(y, p, round(t, 2))["sensitivity"] >= target:
            return round(float(t), 2)
    return 0.01


def bootstrap_auc(y, p, n: int = 2000) -> list:
    """95% interval for the AUC, from resampling the women with replacement."""
    rng = np.random.default_rng(SEED)
    yv, idx, out = y.to_numpy(), np.arange(len(y)), []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        if yv[s].min() != yv[s].max():
            out.append(roc_auc_score(yv[s], p[s]))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def calibration(y, p) -> dict:
    """Do predicted probabilities match what actually happened?"""
    frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    refit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)  # slope 1, intercept 0 is ideal
    return {
        "bins": [{"predicted": float(a), "observed": float(b)} for a, b in zip(mean_pred, frac)],
        "slope": float(refit.coef_[0][0]),
        "intercept": float(refit.intercept_[0]),
        "brier": float(brier_score_loss(y, p)),
    }


def by_age(X, y, p, threshold: float) -> list:
    """Same cut-off, three age bands: does the model serve each one equally?"""
    out = []
    for label, lo, hi in [("21 to 29", 21, 29), ("30 to 44", 30, 44), ("45 and over", 45, 200)]:
        m = ((X["Age"] >= lo) & (X["Age"] <= hi)).to_numpy()
        c = confusion_at(y[m], p[m], threshold)
        out.append({
            "band": label, "n": int(m.sum()), "cases": int(y[m].sum()),
            "prevalence": float(y[m].mean()), "auc": float(roc_auc_score(y[m], p[m])),
            "sensitivity": c["sensitivity"], "specificity": c["specificity"], "missed": c["fn"],
        })
    return out


def export_model(X, y) -> dict:
    """Fit a logistic model on every row of X and flatten it to plain numbers.

    X decides which inputs the model has. The web page rebuilds the prediction from these:
        z = intercept + sum(coef * (value - mean) / scale);  p = 1 / (1 + exp(-z))
    A measurement marked "not measured" is replaced by its median first.
    """
    features = list(X.columns)
    pipe = logistic().fit(X, y)
    imputer, scaler, lr = pipe.named_steps.values()
    filled = pd.DataFrame(imputer.transform(X), columns=features)
    return {
        "features": features,
        "median": [float(v) for v in imputer.statistics_],
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "odds_ratio_per_sd": [float(np.exp(v)) for v in lr.coef_[0]],
        "range": {f: [float(X[f].min()), float(X[f].max())] for f in features},
        "quartiles": {f: [float(v) for v in X[f].quantile([0.25, 0.5, 0.75])] for f in features},
        "check": {  # three worked examples the tests and the page compare against
            "inputs": filled.iloc[[0, 1, 4]].to_numpy().tolist(),
            "probabilities": pipe.predict_proba(X.iloc[[0, 1, 4]])[:, 1].tolist(),
        },
    }


def summarise(X, y, p, threshold=None) -> dict:
    """The standard set of results for one model's held-out probabilities."""
    threshold = pick_threshold(y, p) if threshold is None else threshold
    return {
        "features": list(X.columns),
        "oof_auc": float(roc_auc_score(y, p)),
        "oof_auc_ci": bootstrap_auc(y, p),
        "calibration": calibration(y, p),
        "threshold_default": confusion_at(y, p, 0.5),
        "threshold_screening": confusion_at(y, p, threshold),
        "by_age": by_age(X, y, p, threshold),
    }


def milestone2(X, y, threshold_m1: float) -> dict:
    """Everything notebook 04 and the page quote about missing data."""
    from . import missing                      # imported here so Milestone 1 code has no dependency on it
    table = missing.compare_strategies(X, y)
    draws = missing.multiple_imputation(X, y)
    return {
        "strategies": json.loads(table.to_json(orient="index")),
        "recovery": {c: json.loads(missing.recovery_test(X, c).to_json(orient="index"))
                     for c in ["Insulin", "SkinThickness"]},
        "spread": missing.spread_summary(X, y, draws, threshold_m1),
    }


def main():
    warnings.filterwarnings("ignore")
    REPORTS.mkdir(exist_ok=True)
    X, y = load_data()

    table = compare_models(X, y)
    p = out_of_fold(logistic(), X, y)
    p_glucose = out_of_fold(logistic(), X[["Glucose"]], y)
    threshold = pick_threshold(y, p)

    # Simple clinical-style rule for comparison: flag a 2-hour glucose of 140 or more.
    # Rows with no glucose reading (5) are treated as not flagged.
    rule = confusion_at(y, X["Glucose"].fillna(0).to_numpy(), 140)

    Xc = X[current_features(FEATURES)]          # the current model's inputs (D09)
    p_current = out_of_fold(logistic(), Xc, y)

    metrics = {
        "n": int(len(y)), "cases": int(y.sum()), "prevalence": float(y.mean()),
        "missing": {k: int(v) for k, v in X.isna().sum().items()},
        "cv": {"splits": N_SPLITS, "repeats": N_REPEATS},
        "comparison": json.loads(table.to_json(orient="index")),
        "lucky_split": lucky_split_demo(),
        # --- Milestone 1 baseline: all eight inputs, median fill (notebooks 02 and 03)
        "oof_auc": float(roc_auc_score(y, p)),
        "oof_auc_ci": bootstrap_auc(y, p),
        "oof_auc_glucose_only": float(roc_auc_score(y, p_glucose)),
        "calibration": calibration(y, p),
        "threshold_default": confusion_at(y, p, 0.5),
        "threshold_screening": confusion_at(y, p, threshold),
        "target_sensitivity": TARGET_SENSITIVITY,
        "glucose_140_rule": rule,
        "by_age": by_age(X, y, p, threshold),
        # --- Milestone 2: missing-data study (notebook 04)
        "m2": milestone2(X, y, threshold),
        # --- The current model: what the web page runs and reports
        "current": summarise(Xc, y, p_current),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=1))
    pd.DataFrame({"outcome": y, "p_model": p.round(6), "p_glucose_only": p_glucose.round(6),
                  "p_current": p_current.round(6)}).to_csv(REPORTS / "oof.csv", index=False)
    (REPORTS / "model.json").write_text(json.dumps(export_model(Xc, y), indent=1))

    print(table[["auc", "auc_sd", "avg_precision", "brier", "accuracy", "sensitivity"]].round(3).to_string())
    print(f"\nOut-of-fold AUC {metrics['oof_auc']:.3f}, 95% interval "
          f"{metrics['oof_auc_ci'][0]:.3f} to {metrics['oof_auc_ci'][1]:.3f}")
    print(f"Screening threshold {threshold:.2f}: {json.dumps(metrics['threshold_screening'])}")
    print(f"Default 0.50:             {json.dumps(metrics['threshold_default'])}")
    print(f"Glucose >= 140 rule:      {json.dumps(rule)}")
    print("By age:", json.dumps(metrics["by_age"], indent=1))
    print("Calibration slope/intercept:", metrics["calibration"]["slope"], metrics["calibration"]["intercept"])
    cur = metrics["current"]
    print(f"\nCurrent model ({len(cur['features'])} inputs): AUC {cur['oof_auc']:.3f}, "
          f"screening cut-off {cur['threshold_screening']['threshold']:.2f}")
    print(pd.DataFrame(metrics["m2"]["strategies"]).T.round(4).to_string())


if __name__ == "__main__":
    main()
