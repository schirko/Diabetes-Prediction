"""Milestone 5: what to do about the 35% / 65% split, and where the cut-off should go.

    E15  Does rebalancing the training data (class weights, SMOTE) beat simply moving the cut-off?
    E16  What cut-off does each stated cost of a missed case imply? (decision curve, cost per 100 women)
    E31  Does the cut-off need to differ by age band? (fix for B12)

All of it works on the current model's out-of-fold risks (reports/oof.csv) except E15, which
refits the current model on the same 50 train/test splits used everywhere else.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from . import nonlinear as nl
from .evaluate import calibration, confusion_at
from .modeling import SEED

N_SPLITS, N_REPEATS = 5, 10
COST_RATIOS = (2, 5, 10)      # how many times worse a missed case is than a false alarm
DEFAULT_RATIO = 5
OLD_TARGET = 0.85             # the Milestone 1 rule: catch at least 85% of cases
GRID = np.round(np.arange(0.02, 0.90, 0.01), 2)
BANDS = [("21 to 29", 21, 29), ("30 to 44", 30, 44), ("45 and over", 45, 200)]
METHODS = ["Published model", "Class weights", "SMOTE"]


def cutoff_for_ratio(ratio: float) -> float:
    """The cost-based cut-off: flag a woman when her risk is at least 1 / (1 + ratio).

    Flagging her costs one false-alarm unit if she is healthy (chance 1 - p); not flagging her
    costs `ratio` units if she has diabetes (chance p). Flagging is cheaper once p >= 1 / (1 + ratio).
    """
    return round(1.0 / (1.0 + ratio), 2)      # two decimals, like the page's slider


def implied_ratio(cutoff: float) -> float:
    """The cost ratio a cut-off quietly assumes: the reverse of cutoff_for_ratio."""
    return (1.0 - cutoff) / cutoff


# ---------------------------------------------------------------- E15: rebalancing
def _fit_predict(method: str, Xtr, ytr, Xte, seed: int):
    if method == "SMOTE":
        # Fill gaps and add synthetic cases using the training rows only, then fit the usual model.
        fill = Xtr.median()
        Xf = Xtr.fillna(fill)
        Xs, ys = SMOTE(random_state=seed).fit_resample(Xf, ytr)
        model = nl.curves_model().fit(pd.DataFrame(Xs, columns=Xtr.columns), ys)
    else:
        model = nl.curves_model()
        if method == "Class weights":
            model.set_params(logisticregression__class_weight="balanced")
        model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def compare_rebalancing(X, y) -> dict:
    """Published model, class weights and SMOTE on the same 50 splits (SMOTE inside each training fold)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof = {m: np.zeros(len(y)) for m in METHODS}
    per_split = {m: {"auc": [], "brier": []} for m in METHODS}
    for i, (tr, te) in enumerate(cv.split(X, y)):
        for m in METHODS:
            p = _fit_predict(m, X.iloc[tr], y.iloc[tr], X.iloc[te], SEED + i)
            oof[m][te] += p
            per_split[m]["auc"].append(roc_auc_score(y.iloc[te], p))
            per_split[m]["brier"].append(brier_score_loss(y.iloc[te], p))
    rows = {}
    for m in METHODS:
        p = oof[m] / N_REPEATS
        fpr, tpr, _ = roc_curve(y, p)
        fa = int(round(fpr[np.searchsorted(tpr, OLD_TARGET)] * (y == 0).sum()))   # false alarms to catch 85%
        at_half = confusion_at(y, p, 0.5)
        rows[m] = {
            "auc": float(np.mean(per_split[m]["auc"])), "auc_sd": float(np.std(per_split[m]["auc"])),
            "brier": float(np.mean(per_split[m]["brier"])),
            "calibration_slope": calibration(y, np.clip(p, 1e-6, 1 - 1e-6))["slope"],
            "mean_risk": float(p.mean()),
            "sensitivity_at_half": at_half["sensitivity"], "specificity_at_half": at_half["specificity"],
            "false_alarms_for_85": fa,
        }
    return {"table": rows, "oof": {m: (oof[m] / N_REPEATS).round(6).tolist() for m in METHODS}}


# ---------------------------------------------------------------- E16: costs
def cost_per_100(counts: dict, ratio: float) -> float:
    """Expected cost of a policy per 100 women, in false-alarm units."""
    n = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    return (ratio * counts["fn"] + counts["fp"]) / n * 100


def _cost_curve(y, p, ratio):
    yv = np.asarray(y)
    flag = p[:, None] >= GRID[None, :]
    fn = ((~flag) & (yv[:, None] == 1)).sum(0)
    fp = (flag & (yv[:, None] == 0)).sum(0)
    return (ratio * fn + fp) / len(yv) * 100


def best_cutoff_interval(y, p, ratio, n: int = 1000) -> dict:
    """Where the cheapest cut-off sits, and how much it moves when the women are resampled."""
    yv, rng = np.asarray(y), np.random.default_rng(SEED)
    best = []
    for _ in range(n):
        s = rng.integers(0, len(yv), len(yv))
        best.append(GRID[int(np.argmin(_cost_curve(yv[s], p[s], ratio)))])
    point = GRID[int(np.argmin(_cost_curve(yv, p, ratio)))]
    return {"best": float(point), "low": float(np.percentile(best, 2.5)), "high": float(np.percentile(best, 97.5))}


def net_benefit(y, p, pt: float) -> float:
    c = confusion_at(y, p, pt)
    n = len(y)
    return c["tp"] / n - c["fp"] / n * pt / (1 - pt)


def cost_study(y, p, rule_counts: dict, old_cutoff: float) -> dict:
    """What each stated cost ratio implies, and how it compares with the other ways of choosing."""
    n = len(y)
    out = {"ratios": {}, "old_cutoff": old_cutoff, "old_implied_ratio": implied_ratio(old_cutoff)}
    everyone = {"tp": int((y == 1).sum()), "fp": int((y == 0).sum()), "fn": 0, "tn": 0}
    nobody = {"tp": 0, "fp": 0, "fn": int((y == 1).sum()), "tn": int((y == 0).sum())}
    for r in COST_RATIOS:
        pt = cutoff_for_ratio(r)
        c = confusion_at(y, p, pt)
        policies = {
            "Test nobody": cost_per_100(nobody, r),
            "Test everybody": cost_per_100(everyone, r),
            "Glucose of 140 or more": cost_per_100(rule_counts, r),
            "Old rule (catch 85%)": cost_per_100(confusion_at(y, p, old_cutoff), r),
            "Cut-off 0.50": cost_per_100(confusion_at(y, p, 0.5), r),
            "Cost-based cut-off": cost_per_100(c, r),
        }
        out["ratios"][str(r)] = {
            "ratio": r, "cutoff": pt, "counts": c,
            "per_100": per_100_counts(c, n),
            "cost_per_100": policies, "empirical_best": best_cutoff_interval(y, p, r),
            "net_benefit": net_benefit(y, p, pt),
            "net_benefit_all": float((y == 1).mean() - (y == 0).mean() * pt / (1 - pt)),
        }
    pts = np.round(np.arange(0.02, 0.81, 0.01), 2)
    out["decision_curve"] = {
        "pt": pts.tolist(),
        "model": [net_benefit(y, p, t) for t in pts],
        "all": [float((y == 1).mean() - (y == 0).mean() * t / (1 - t)) for t in pts],
    }
    out["cost_curves"] = {str(r): _cost_curve(y, p, r).tolist() for r in COST_RATIOS} | {"grid": GRID.tolist()}
    return out


def per_100_counts(c: dict, n: int) -> dict:
    """Largest-remainder rounding to 100 women, the same way the project page does it."""
    keys = ["tp", "fn", "fp", "tn"]
    raw = np.array([c[k] / n * 100 for k in keys])
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: 100 - base.sum()]:
        base[i] += 1
    return dict(zip(keys, base.tolist()))


# ---------------------------------------------------------------- E31: age bands
def _band_labels(X):
    age = X["Age"].to_numpy()
    return np.select([age <= 29, age <= 44], [0, 1], 2)


def calibration_by_band(X, y, p) -> list:
    band = _band_labels(X)
    rows = []
    for k, (label, _, _) in enumerate(BANDS):
        m = band == k
        logit = np.log(p[m] / (1 - p[m])).reshape(-1, 1)
        slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(logit, y[m]).coef_[0][0])
        rows.append({"band": label, "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean()), "slope": slope})
    return rows


def _choose(policy, yb, pb, ratio):
    """A cut-off for one age band, chosen from that band's training rows only."""
    flag = pb[:, None] >= GRID[None, :]
    tp = (flag & (yb[:, None] == 1)).sum(0)
    fn = ((~flag) & (yb[:, None] == 1)).sum(0)
    fp = (flag & (yb[:, None] == 0)).sum(0)
    if policy == "Cheapest per band":
        return GRID[int(np.argmin(ratio * fn + fp))]
    sens = tp / max(tp[0] + fn[0], 1)                         # catch 85% inside this band
    ok = np.where(sens >= OLD_TARGET)[0]
    return GRID[ok.max()] if len(ok) else GRID[0]


def band_policies(X, y, p, ratio: float) -> dict:
    """Single cost-based cut-off vs a cut-off per band, with the per-band ones chosen on training rows only.

    Each woman is scored by the cut-off that was set without her (repeated 5-fold over the 768 women).
    Counts are totals over the whole data (average of the 10 repeats).
    """
    yv, band = np.asarray(y), _band_labels(X)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    policies = ["Single cost-based cut-off", "Cheapest per band", "Catch 85% in every band"]
    tot = {pol: np.zeros((3, 4)) for pol in policies}          # band x (tp, fp, fn, tn)
    cuts = {pol: [[] for _ in range(3)] for pol in policies}
    for tr, te in cv.split(p, band * 2 + yv):
        for pol in policies:
            for k in range(3):
                trk, tek = tr[band[tr] == k], te[band[te] == k]
                if pol == "Single cost-based cut-off":
                    t = cutoff_for_ratio(ratio)
                else:
                    t = _choose(pol, yv[trk], p[trk], ratio)
                cuts[pol][k].append(float(t))
                flag, yt = p[tek] >= t, yv[tek]
                tot[pol][k] += [(flag & (yt == 1)).sum(), (flag & (yt == 0)).sum(),
                                ((~flag) & (yt == 1)).sum(), ((~flag) & (yt == 0)).sum()]
    out = {}
    for pol in policies:
        arr = tot[pol] / N_REPEATS
        allc = arr.sum(0)
        out[pol] = {
            "cost_per_100": float((ratio * allc[2] + allc[1]) / allc.sum() * 100),
            "bands": [{"band": BANDS[k][0], "cutoff": float(np.median(cuts[pol][k])),
                       "sensitivity": float(arr[k][0] / (arr[k][0] + arr[k][2])),
                       "specificity": float(arr[k][3] / (arr[k][3] + arr[k][1])),
                       "missed": float(arr[k][2]), "false_alarms": float(arr[k][1])} for k in range(3)],
        }
    return out


def at_cutoff_by_band(X, y, p, cutoff: float) -> list:
    band, yv = _band_labels(X), np.asarray(y)
    rows = []
    for k, (label, _, _) in enumerate(BANDS):
        m = band == k
        c = confusion_at(yv[m], p[m], cutoff)
        rows.append({"band": label, "cases": int(yv[m].sum()), "sensitivity": c["sensitivity"],
                     "specificity": c["specificity"], "missed": c["fn"], "false_alarms": c["fp"]})
    return rows


def run(X, y, p, rule_counts: dict, old_cutoff: float) -> dict:
    """Everything Milestone 5 reports. `X` is the current model's inputs, `p` its out-of-fold risks."""
    p = np.asarray(p)
    reb = compare_rebalancing(X, y)
    costs = cost_study(y, p, rule_counts, old_cutoff)
    bands = {"calibration": calibration_by_band(X, y, p),
             "old_cutoff": at_cutoff_by_band(X, y, p, old_cutoff),
             "by_ratio": {}}
    for r in COST_RATIOS:
        bands["by_ratio"][str(r)] = {"at_cost_cutoff": at_cutoff_by_band(X, y, p, cutoff_for_ratio(r)),
                                     "policies": band_policies(X, y, p, r)}
    return {"cost_ratios": list(COST_RATIOS), "default_ratio": DEFAULT_RATIO,
            "rebalancing": reb["table"], "costs": costs, "bands": bands}, reb["oof"]
