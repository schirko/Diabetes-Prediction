"""Milestone 8: does the frozen model work on women who are not in the Pima data?

The outside data is NHANES 2013-2014 (US national health survey): women aged 21 and over who had the 2-hour glucose test.
Raw SAS files go in data/nhanes/ (download list in README); `build_table` turns them into one small table, saved as
data/nhanes_women.csv so the tests and the site build do not need the raw files.

What is different from the Pima data, and what we did about it
  - Outcome. Pima: diabetes diagnosed within five years. NHANES: a one-time snapshot. We call a woman positive when her
    HbA1c is 6.5% or higher, or she reports a doctor's diagnosis. HbA1c is a different blood test from the glucose input,
    so the outcome does not simply restate an input.
  - Family history. The model wants the Pima pedigree score; NHANES has only yes or no for a relative with diabetes. The
    input is left blank, and the frozen model fills it with the Pima median (the same thing the page does for "not measured").
  - Women on diabetes medicine are not given the glucose test, so the sample leaves out most women already treated.
The frozen model is fitted on all 768 Pima women once and never refitted on NHANES.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from . import decision, nonlinear as nl
from .data import ROOT, load_data
from .modeling import current_features

RAW = ROOT / "data" / "nhanes"
TABLE = ROOT / "data" / "nhanes_women.csv"
SEED = 42
BOOT = 1000


def build_table() -> pd.DataFrame:
    """Women 21+ with a 2-hour glucose reading, one row each, in the Pima data's column names."""
    r = lambda name: pd.read_sas(RAW / f"{name}_H.xpt", format="xport")
    m = r("DEMO")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]]
    for name in ["OGTT", "BMX", "BPX", "RHQ", "GHB", "DIQ"]:
        m = m.merge(r(name), on="SEQN", how="left")
    w = m[(m.RIAGENDR == 2) & (m.RIDAGEYR >= 21) & m.LBXGLT.notna()].copy()
    diastolic = w[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].replace(0, np.nan).mean(axis=1)   # 0 = not read
    ever = w.RHQ131
    pregnancies = np.where(ever == 2, 0, np.where(w.RHQ160.between(1, 40), w.RHQ160, np.nan))
    out = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": w.LBXGLT.to_numpy(), "BloodPressure": diastolic.to_numpy(),
        "BMI": w.BMXBMI.to_numpy(), "Age": w.RIDAGEYR.to_numpy(), "Group": w.RIDRETH3.to_numpy(),
        "HbA1c": w.LBXGH.to_numpy(), "Diagnosed": (w.DIQ010 == 1).astype(int).to_numpy()})
    out = out[out.HbA1c.notna()].reset_index(drop=True)
    out["Outcome"] = ((out.HbA1c >= 6.5) | (out.Diagnosed == 1)).astype(int)
    return out


def load_table() -> pd.DataFrame:
    if not TABLE.exists():
        build_table().to_csv(TABLE, index=False)
    return pd.read_csv(TABLE)


def frozen_model():
    """The published model, fitted once on all Pima women."""
    X, y = load_data()
    cols = current_features(list(X.columns))
    return nl.curves_model().fit(X[cols], y), cols


def predict(model, cols, table: pd.DataFrame) -> np.ndarray:
    frame = table.reindex(columns=cols)           # DiabetesPedigreeFunction is absent -> NaN -> Pima median
    return model.predict_proba(frame)[:, 1]


def _auc_ci(y, p, rng):
    n = len(y)
    vals = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def calibration_fit(y, p):
    """Slope and intercept of a logistic recalibration of log-odds, and the calibration-in-the-large gap."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    m = LogisticRegression(C=1e6).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def at_cutoff(y, p, cut):
    flag = p >= cut
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    n = len(y)
    return {"cutoff": cut, "flagged": float(flag.mean()), "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "precision": tp / max(tp + fp, 1) if tp + fp else None,
            "missed_per_100": 100 * fn / n, "false_alarms_per_100": 100 * fp / n, "caught_per_100": 100 * tp / n}


def shift_table(pima: pd.DataFrame, nh: pd.DataFrame) -> dict:
    out = {}
    for c in ["Age", "Glucose", "BMI", "BloodPressure", "Pregnancies"]:
        out[c] = {"pima_median": float(pima[c].median()), "nhanes_median": float(nh[c].median()),
                  "pima_p10": float(pima[c].quantile(.1)), "nhanes_p10": float(nh[c].quantile(.1)),
                  "pima_p90": float(pima[c].quantile(.9)), "nhanes_p90": float(nh[c].quantile(.9))}
    return out


def domain_auc(pima: pd.DataFrame, nh: pd.DataFrame, cols) -> float:
    """How well can a simple model tell a Pima woman from an NHANES woman using the same inputs? 0.5 = identical populations."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    a = pd.concat([pima[cols], nh[cols]], ignore_index=True)
    lab = np.r_[np.zeros(len(pima)), np.ones(len(nh))]
    a = a.fillna(a.median())
    p = cross_val_predict(GradientBoostingClassifier(random_state=SEED), a, lab,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return float(roc_auc_score(lab, p))


def run(cutoffs: dict) -> dict:
    nh = load_table()
    X, y_pima = load_data()
    model, cols = frozen_model()
    p = predict(model, cols, nh)
    y = nh["Outcome"].to_numpy()
    rng = np.random.default_rng(SEED)
    shared = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
    pima_p = model.predict_proba(X[cols])[:, 1]

    res = {"n": int(len(nh)), "positives": int(y.sum()), "prevalence": float(y.mean()),
           "pima_prevalence": float(y_pima.mean()),
           "outcome_parts": {"a1c_6_5": int((nh.HbA1c >= 6.5).sum()), "diagnosed": int(nh.Diagnosed.sum()),
                             "both": int(((nh.HbA1c >= 6.5) & (nh.Diagnosed == 1)).sum())}}
    lo, hi = _auc_ci(y, p, rng)
    res["auc"] = {"value": float(roc_auc_score(y, p)), "low": lo, "high": hi,
                  "pima_in_sample": float(roc_auc_score(y_pima, pima_p))}
    # the glucose reading alone, as a yardstick: does the model add anything beyond it here?
    g = nh["Glucose"].to_numpy()
    glo, ghi = _auc_ci(y, g, rng)
    res["glucose_only"] = {"auc": float(roc_auc_score(y, g)), "low": glo, "high": ghi}
    # sensitivity check: the narrower outcome (HbA1c only)
    y2 = (nh.HbA1c >= 6.5).astype(int).to_numpy()
    res["a1c_only"] = {"positives": int(y2.sum()), "auc": float(roc_auc_score(y2, p))}

    slope, intercept = calibration_fit(y, p)
    res["calibration"] = {"mean_predicted": float(p.mean()), "observed": float(y.mean()), "slope": slope,
                          "intercept": intercept, "brier": float(brier_score_loss(y, p)),
                          "brier_prevalence_only": float(brier_score_loss(y, np.full(len(y), y.mean())))}
    # risk bands: predicted vs observed
    bands = pd.cut(p, [0, .1, .3, .6, 1.0], include_lowest=True, labels=["under 10%", "10% to 30%", "30% to 60%", "60% and over"])
    res["bands"] = {str(b): {"n": int((bands == b).sum()), "mean_predicted": float(p[bands == b].mean()) if (bands == b).any() else None,
                             "observed": float(y[bands == b].mean()) if (bands == b).any() else None}
                    for b in bands.categories}
    # intercept-only recalibration, honestly scored: shift fitted on training folds, scored on held-out folds
    from sklearn.model_selection import StratifiedKFold
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    shifted = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(z, y):
        lo_, hi_ = -8.0, 8.0
        for _ in range(60):                            # find the shift that makes the training mean match the observed rate
            mid = (lo_ + hi_) / 2
            if (1 / (1 + np.exp(-(z[tr] + mid)))).mean() > y[tr].mean():
                hi_ = mid
            else:
                lo_ = mid
        shifted[te] = 1 / (1 + np.exp(-(z[te] + (lo_ + hi_) / 2)))
    res["recalibrated"] = {"brier": float(brier_score_loss(y, shifted)), "mean_predicted": float(shifted.mean()),
                           "auc": float(roc_auc_score(y, shifted))}
    res["cutoffs"] = {str(r): at_cutoff(y, p, c) for r, c in cutoffs.items()}
    # the same cut-offs on the Pima women, for the side-by-side table
    res["cutoffs_pima"] = {str(r): at_cutoff(y_pima.to_numpy(), pima_p, c) for r, c in cutoffs.items()}
    # age groups
    age = nh["Age"].to_numpy()
    res["by_age"] = {}
    for name, mask in [("21 to 44", age < 45), ("45 to 64", (age >= 45) & (age < 65)), ("65 and over", age >= 65)]:
        ok = mask & ~np.isnan(p)
        res["by_age"][name] = {"n": int(ok.sum()), "positives": int(y[ok].sum()),
                               "auc": float(roc_auc_score(y[ok], p[ok])) if 0 < y[ok].sum() < ok.sum() else None,
                               "mean_predicted": float(p[ok].mean()), "observed": float(y[ok].mean())}
    labels = {1: "Mexican American", 2: "Other Hispanic", 3: "White, not Hispanic", 4: "Black, not Hispanic",
              6: "Asian, not Hispanic", 7: "Other or mixed"}
    res["by_group"] = {}
    for code, name in labels.items():
        ok = (nh["Group"].to_numpy() == code) & ~np.isnan(p)
        res["by_group"][name] = {"n": int(ok.sum()), "positives": int(y[ok].sum()),
                                 "auc": float(roc_auc_score(y[ok], p[ok])) if 0 < y[ok].sum() < ok.sum() else None,
                                 "mean_predicted": float(p[ok].mean()), "observed": float(y[ok].mean())}
    res["shift"] = shift_table(X, nh)
    res["domain_auc"] = domain_auc(X, nh, shared)
    res["missing"] = {c: int(nh[c].isna().sum()) for c in shared}
    return res


def predictions(cutoffs=None) -> pd.DataFrame:
    nh = load_table()
    model, cols = frozen_model()
    nh = nh.copy()
    nh["p"] = predict(model, cols, nh)
    return nh
