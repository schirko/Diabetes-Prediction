"""Milestone 7: how sure is the model about one woman's number?

    E19  A bootstrap range around each woman's estimated risk: refit the model on many resamples of the 768 women
         and see how far her risk moves.
    E20  Conformal prediction: a rule with a stated guarantee that turns the risk into "no", "yes" or
         "too uncertain to call", with the guarantee checked on women the rule has not seen.

Two different questions are answered here and they are easy to confuse.
    The bootstrap range says how much the MODEL'S ESTIMATE would change if it had been trained on a different set of
    768 women like these. It is about the model, not about the woman: she either develops diabetes or she doesn't,
    so no range around a risk can be "right" about her outcome the way a range around a height can.
    The conformal call is about HER OUTCOME: of the women the rule labels "no" (or "yes"), at least a stated share
    really are.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from . import nonlinear as nl
from .explain import fit_current
from .modeling import SEED

B = 200                       # bootstrap refits
GRID_POINTS = 40              # points per input in the exported tables (Pregnancies, a count, keeps every integer)
LEVEL = (2.5, 97.5)           # a 95% range
COVER_RUNS = 60               # simulated studies for the bootstrap coverage check
COVER_B = 100                 # refits per simulated study
CONF_RUNS = 300               # random splits for the conformal check
ALPHAS = (0.10, 0.20)         # conformal error levels: at least 90% / 80% of each class labelled correctly


# ------------------------------------------------------------------ E19: bootstrap tables
def _coarse_grid(m: dict, f: str) -> np.ndarray:
    lo, hi = m["range"][f]
    if f == "Pregnancies":
        return np.arange(lo, hi + 1.0, 1.0)
    return np.round(np.linspace(lo, hi, GRID_POINTS), 4)


def _tables(model, features, medians, grids):
    """One refitted model as lookup tables: a base log-odds plus one contribution table per input."""
    base_row = pd.DataFrame([medians.values], columns=features)
    base = float(model.decision_function(base_row)[0])
    out = {}
    for f in features:
        rows = pd.DataFrame(np.repeat(base_row.values, len(grids[f]), axis=0), columns=features)
        rows[f] = grids[f]
        out[f] = model.decision_function(rows) - base
    return base, out


def bootstrap_tables(X, y, n_boot: int = B, seed: int = SEED) -> dict:
    """Refit the published model on `n_boot` resamples of the women; keep each fit as lookup tables.

    The tables use the same arithmetic as the page: log-odds = base + sum of one contribution per input, read off a
    grid by straight-line interpolation. Every refit measures contributions from the same reference woman (the median
    of every input in the full data), so the tables line up across refits.
    """
    full = nl.export_curves_model(X, y)
    features = full["features"]
    medians = X.median()
    grids = {f: _coarse_grid(full, f) for f in features}
    rng = np.random.default_rng(seed)
    bases, contribs = [], {f: [] for f in features}
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        model = fit_current(X.iloc[idx], y.iloc[idx])
        base, tabs = _tables(model, features, medians, grids)
        bases.append(base)
        for f in features:
            contribs[f].append(tabs[f])
    return {"features": features, "median": [float(medians[f]) for f in features],
            "grid": {f: [float(v) for v in grids[f]] for f in features},
            "base": np.array(bases), "contribution": {f: np.array(contribs[f]) for f in features}}


def risks_from_tables(t: dict, rows) -> np.ndarray:
    """Predicted risk for each row under every bootstrap refit: shape (n_rows, n_boot). Same arithmetic as the page."""
    rows = np.atleast_2d(rows)
    z = np.tile(t["base"], (len(rows), 1))
    for j, f in enumerate(t["features"]):
        g = np.array(t["grid"][f])
        c = t["contribution"][f]                       # (n_boot, n_grid)
        x = np.clip(rows[:, j], g[0], g[-1])
        i = np.clip(np.searchsorted(g, x, side="right") - 1, 0, len(g) - 2)
        w = ((x - g[i]) / (g[i + 1] - g[i]))[:, None]
        z += (c[:, i].T * (1 - w) + c[:, i + 1].T * w)
    return 1 / (1 + np.exp(-z))


def interval_summary(X, y, t: dict, point: np.ndarray, cutoffs: dict) -> dict:
    """How wide the 95% ranges are, where they are widest, and how often the cut-off falls inside one."""
    filled = X.fillna(pd.Series(dict(zip(t["features"], t["median"]))))[t["features"]].to_numpy()
    p = risks_from_tables(t, filled)
    lo, hi = np.percentile(p, LEVEL[0], axis=1), np.percentile(p, LEVEL[1], axis=1)
    width = hi - lo
    centre = p.mean(axis=1)
    out = {"median_width": float(np.median(width)), "mean_width": float(width.mean()),
           "p10_width": float(np.percentile(width, 10)), "p90_width": float(np.percentile(width, 90)),
           "max_width": float(width.max()), "point_inside": float(((point >= lo) & (point <= hi)).mean()),
           "mean_gap_point_vs_boot_mean": float(np.abs(point - centre).mean())}
    by_risk = {}
    for name, a, b in [("under 10%", 0, 0.10), ("10% to 30%", 0.10, 0.30), ("30% to 60%", 0.30, 0.60), ("60% and over", 0.60, 1.01)]:
        mask = (point >= a) & (point < b)
        by_risk[name] = {"n": int(mask.sum()), "median_width": float(np.median(width[mask])) if mask.any() else None}
    out["by_risk"] = by_risk
    out["cutoff_inside"] = {}
    for r, c in cutoffs.items():
        inside = (lo <= c) & (c <= hi)
        out["cutoff_inside"][str(r)] = {"cutoff": c, "share": float(inside.mean()),
                                        "flag_changes_in_share_of_refits": float(np.mean(np.minimum((p >= c).mean(axis=1), 1 - (p >= c).mean(axis=1))))}
    # which women have the widest ranges?
    order = np.argsort(-width)[:20]
    ex = X.iloc[order]
    out["widest_traits"] = {"median_glucose": float(ex["Glucose"].median()), "median_bmi": float(ex["BMI"].median()),
                            "median_age": float(ex["Age"].median()),
                            "all_glucose": float(X["Glucose"].median()), "all_bmi": float(X["BMI"].median()),
                            "all_age": float(X["Age"].median())}
    return out, lo, hi


def spread_check(X, y, t: dict) -> dict:
    """Does the bootstrap spread agree with how much the estimate moves between different training sets?

    Each woman is held out of 10 of the 50 cross-validation fits, so she has 10 estimates from models that never saw
    her, each trained on 80% of the women. A subsample of 80% moves less than a full-size resample, by a known factor:
    the variance of an estimate from m of n women, around the full-data estimate, is about (n/m - 1) times the variance
    of an estimate from a full-size sample, so the standard deviations should differ by about sqrt(0.8 / 0.2) = 2.
    """
    from .evaluate import cv_splitter
    n = len(X)
    store = [[] for _ in range(n)]
    filled = X.fillna(X.median())
    for train, test in cv_splitter().split(X, y):
        model = fit_current(X.iloc[train], y.iloc[train])
        for i, p in zip(test, model.predict_proba(X.iloc[test])[:, 1]):
            store[i].append(p)
    cv_sd = np.array([np.std(s, ddof=1) for s in store])
    p = risks_from_tables(t, filled[t["features"]].to_numpy())
    boot_sd = p.std(axis=1, ddof=1)
    factor = np.sqrt(0.8 / 0.2)
    ratio = boot_sd / (cv_sd * factor)
    return {"median_ratio": float(np.median(ratio)), "p10_ratio": float(np.percentile(ratio, 10)),
            "p90_ratio": float(np.percentile(ratio, 90)), "corr_log_sd": float(np.corrcoef(np.log(boot_sd), np.log(cv_sd))[0, 1]),
            "expected_factor": float(factor), "median_cv_sd": float(np.median(cv_sd)), "median_boot_sd": float(np.median(boot_sd))}


def coverage_check(X, y, n_runs: int = COVER_RUNS, n_boot: int = COVER_B, seed: int = SEED) -> dict:
    """If the bootstrap says 95%, does the true risk fall inside about 95% of the time?

    The true risk is unknowable for real women, so this is a simulation. The full-data model is taken as the truth. New
    outcomes are drawn for the same 768 women from those risks, a fresh model is fitted to each simulated study, its 95% range
    is built by the same bootstrap, and we count how many women's true risks fall inside their range. It tests the
    method, assuming the model's shape is right.
    """
    rng = np.random.default_rng(seed + 1)
    truth_model = fit_current(X, y)
    true_p = truth_model.predict_proba(X)[:, 1]
    full = nl.export_curves_model(X, y)
    features = full["features"]
    medians = X.median()
    grids = {f: _coarse_grid(full, f) for f in features}
    filled = X.fillna(medians)[features].to_numpy()
    shares, widths, bias = [], [], []
    for _ in range(n_runs):
        ysim = pd.Series((rng.random(len(X)) < true_p).astype(int), index=y.index)
        bases, contribs = [], {f: [] for f in features}
        for _ in range(n_boot):
            idx = rng.integers(0, len(X), len(X))
            m = fit_current(X.iloc[idx], ysim.iloc[idx])
            base, tabs = _tables(m, features, medians, grids)
            bases.append(base)
            for f in features:
                contribs[f].append(tabs[f])
        t = {"features": features, "grid": {f: list(grids[f]) for f in features}, "base": np.array(bases),
             "contribution": {f: np.array(contribs[f]) for f in features}}
        p = risks_from_tables(t, filled)
        lo, hi = np.percentile(p, LEVEL[0], axis=1), np.percentile(p, LEVEL[1], axis=1)
        shares.append(float(((true_p >= lo) & (true_p <= hi)).mean()))
        widths.append(float(np.median(hi - lo)))
        bias.append(float(np.mean(p.mean(axis=1) - true_p)))
    return {"runs": n_runs, "boot_per_run": n_boot, "coverage_mean": float(np.mean(shares)),
            "coverage_low": float(np.percentile(shares, 2.5)), "coverage_high": float(np.percentile(shares, 97.5)),
            "median_width": float(np.mean(widths)), "mean_bias": float(np.mean(bias))}


# ------------------------------------------------------------------ E20: conformal prediction
def class_thresholds(p_cal, y_cal, alpha: float) -> dict:
    """Class-by-class (Mondrian) split conformal. Score = 1 - the model's probability for the true class.

    For each class k, q_k is the (1 - alpha) quantile of the scores of calibration women truly in class k, with the usual
    finite-sample correction ceil((n_k + 1)(1 - alpha)) / n_k. A new woman's prediction set contains class k when her score
    for k is at most q_k. Because the score is monotone in the risk, the whole rule is two cut-offs on the risk:
    risk below `low` -> {no}, risk above `high` -> {yes}, in between -> {no, yes}, "too uncertain to call".
    """
    p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal)
    out = {}
    for k in (0, 1):
        # score of a woman whose true class is k = 1 - (the model's probability of k): 1 - p for yes, p for no
        s = np.sort(1 - p_cal[y_cal == 1] if k == 1 else p_cal[y_cal == 0])
        n = len(s)
        rank = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        out[k] = float(s[rank - 1])
    # class 1 is in the set when 1 - p <= q1  <=>  p >= 1 - q1 ; class 0 is in the set when p <= q0 (score for class 0 is p)
    return {"q0": out[0], "q1": out[1], "low": 1 - out[1], "high": out[0]}


def conformal_calls(p, th: dict):
    """0 = confident no, 1 = confident yes, 2 = too uncertain to call, 3 = empty set (neither class fits)."""
    p = np.asarray(p)
    in0 = p <= th["high"]
    in1 = p >= th["low"]
    out = np.full(len(p), 3)
    out[in0 & ~in1] = 0
    out[in1 & ~in0] = 1
    out[in0 & in1] = 2
    return out


def conformal_check(X, y, alphas=ALPHAS, n_runs: int = CONF_RUNS, seed: int = SEED) -> dict:
    """Verify the guarantee on women the rule has not seen.

    Each run splits the 768 women into training (60%), calibration (20%) and test (20%), stratified. The model is fitted on
    the training women, the two risk cut-offs are set on the calibration women, and the rule is scored on the test women.
    Over many runs, the share of test women whose true class is in their set should be at least 1 - alpha for each class.
    """
    res = {a: {"cov0": [], "cov1": [], "cov": [], "no": [], "yes": [], "uncertain": [], "empty": [], "acc_confident": [],
               "uncertain_cases": [], "no_wrong": [], "yes_wrong": []} for a in alphas}
    yv = y.to_numpy()
    for r in range(n_runs):
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + r)
        rest, test = next(outer.split(X, yv))
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000 + r)
        tr_i, cal_i = next(inner.split(X.iloc[rest], yv[rest]))
        train, cal = rest[tr_i], rest[cal_i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = clone(nl.curves_model()).fit(X.iloc[train], y.iloc[train])
        p_cal = model.predict_proba(X.iloc[cal])[:, 1]
        p_te = model.predict_proba(X.iloc[test])[:, 1]
        for a in alphas:
            th = class_thresholds(p_cal, yv[cal], a)
            call = conformal_calls(p_te, th)
            yt = yv[test]
            in_set = np.where(yt == 1, (call == 1) | (call == 2), (call == 0) | (call == 2))
            R = res[a]
            R["cov"].append(float(in_set.mean()))
            R["cov0"].append(float(in_set[yt == 0].mean()))
            R["cov1"].append(float(in_set[yt == 1].mean()))
            R["no"].append(float((call == 0).mean()))
            R["yes"].append(float((call == 1).mean()))
            R["uncertain"].append(float((call == 2).mean()))
            R["empty"].append(float((call == 3).mean()))
            conf = (call == 0) | (call == 1)
            R["acc_confident"].append(float((call[conf] == yt[conf]).mean()) if conf.any() else np.nan)
            R["no_wrong"].append(float(yt[call == 0].mean()) if (call == 0).any() else np.nan)         # confident "no" but has diabetes
            R["yes_wrong"].append(float(1 - yt[call == 1].mean()) if (call == 1).any() else np.nan)    # confident "yes" but healthy
            u = call == 2
            R["uncertain_cases"].append(float(yt[u].mean()) if u.any() else np.nan)
    out = {}
    for a, R in res.items():
        out[str(a)] = {k: float(np.nanmean(v)) for k, v in R.items()}
        out[str(a)]["cov0_low"] = float(np.percentile(R["cov0"], 5))
        out[str(a)]["cov1_low"] = float(np.percentile(R["cov1"], 5))
        out[str(a)]["target"] = 1 - a
    return out


def page_thresholds(p_oof, y, alphas=ALPHAS) -> dict:
    """The two risk cut-offs the page uses, set on every woman's out-of-fold risk (cross-conformal calibration)."""
    return {str(a): class_thresholds(p_oof, y, a) for a in alphas}


# ------------------------------------------------------------------ the whole milestone
def run(X, y, p_oof, cutoffs: dict, rerun: bool = False) -> tuple:
    """Everything Milestone 7 reports. Returns (metrics dict, bootstrap tables for the page).

    The coverage simulation (60 simulated studies of 100 refits each) takes about six minutes, so it is saved in
    reports/m7_coverage.json and reused; run(..., rerun=True) repeats it.
    """
    import json
    from .data import ROOT
    cache = ROOT / "reports" / "m7_coverage.json"
    t = bootstrap_tables(X, y)
    point = fit_current(X, y).predict_proba(X)[:, 1]
    summ, lo, hi = interval_summary(X, y, t, point, cutoffs)
    if cache.exists() and not rerun:
        coverage = json.loads(cache.read_text())
    else:
        coverage = coverage_check(X, y)
        cache.write_text(json.dumps(coverage, indent=1))
    res = {"n_boot": B, "level": [LEVEL[0], LEVEL[1]], "intervals": summ, "spread_check": spread_check(X, y, t),
           "coverage": coverage, "conformal": conformal_check(X, y),
           "page_thresholds": page_thresholds(np.asarray(p_oof), np.asarray(y))}
    return res, t


def export_for_page(t: dict) -> dict:
    """The bootstrap tables as plain lists for the web page (three decimals is far finer than the page shows)."""
    return {"features": t["features"], "grid": t["grid"], "base": [round(float(v), 4) for v in t["base"]],
            "contribution": {f: [[round(float(v), 3) for v in row] for row in t["contribution"][f]] for f in t["features"]}}


def tables_from_json(d: dict) -> dict:
    """The page's bootstrap tables (reports/bootstrap.json) back into arrays, for charts and tests."""
    return {"features": d["features"], "grid": d["grid"], "base": np.array(d["base"]),
            "contribution": {f: np.array(d["contribution"][f]) for f in d["features"]}}
