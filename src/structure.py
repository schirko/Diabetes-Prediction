"""Milestone 9: is there structure in the Pima data that the labels don't show?

Two questions, both asked without looking at who has diabetes until the end.
  E23  Do the women fall into natural groups (clusters)? Does a map of the data (PCA, UMAP) show any? If a group exists,
       does knowing which group a woman is in help the model?
  E24  Which records are unusual, and are they data errors or just rare women? Do unusual women get worse predictions?

Everything here uses the model's six inputs, filled and rescaled the same way the model does it. The cluster-as-input
test runs inside the cross-validation, so the clusters are found on training women only (no leak).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from . import nonlinear as nl
from .contest import corrected_ttest
from .data import ROOT, load_data
from .modeling import SEED, current_features

KS = list(range(2, 9))
N_NULL = 30            # reference data sets drawn from one smooth blob with the same averages and correlations
N_STABLE = 30          # resamples for the stability check
MAPS_CSV = ROOT / "reports" / "m9_maps.csv"


def prepare(X: pd.DataFrame) -> np.ndarray:
    """The model's six inputs: gaps filled with the median, extremes held at the 1st and 99th percentile, rescaled."""
    cols = current_features(list(X.columns))
    F = X[cols].copy()
    F = F.fillna(F.median())
    F = F.clip(F.quantile(0.01), F.quantile(0.99), axis=1)
    return ((F - F.mean()) / F.std()).to_numpy()


# ----------------------------------------------------------------------------- maps
def pca_summary(Z: np.ndarray, cols) -> dict:
    p = PCA().fit(Z)
    return {"explained": [float(v) for v in p.explained_variance_ratio_],
            "loadings": {f"PC{i + 1}": {c: float(v) for c, v in zip(cols, p.components_[i])} for i in range(3)}}


def make_maps(Z: np.ndarray, y: np.ndarray) -> tuple:
    """PCA and UMAP in two dimensions, with a score for how faithfully each keeps each woman's neighbours."""
    import umap
    pca = PCA(n_components=2).fit_transform(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(Z)
    trust = {"pca": float(trustworthiness(Z, pca, n_neighbors=15)), "umap": float(trustworthiness(Z, um, n_neighbors=15))}
    frame = pd.DataFrame({"pc1": pca[:, 0], "pc2": pca[:, 1], "umap1": um[:, 0], "umap2": um[:, 1], "outcome": y})
    return frame, trust


# ----------------------------------------------------------------------------- clusters
def _null(Z: np.ndarray, rng) -> np.ndarray:
    return rng.multivariate_normal(Z.mean(axis=0), np.cov(Z, rowvar=False), size=len(Z))


def cluster_scan(Z: np.ndarray) -> dict:
    """For each number of clusters: silhouette (1 = tight, well-separated; 0 = overlapping), the same score on
    structureless reference data, a stability score, and the mixture model's BIC (lower is a better fit)."""
    rng = np.random.default_rng(SEED)
    nulls = [_null(Z, rng) for _ in range(N_NULL)]
    out = {}
    for k in KS:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        ref = [silhouette_score(n, KMeans(k, n_init=5, random_state=SEED).fit_predict(n)) for n in nulls]
        # stability: cluster two resamples, label every woman with each, and see whether the two labellings agree
        aris = []
        for _ in range(N_STABLE):
            a, b = (KMeans(k, n_init=5, random_state=int(rng.integers(1e6))).fit(Z[rng.integers(0, len(Z), len(Z))]) for _ in range(2))
            aris.append(adjusted_rand_score(a.predict(Z), b.predict(Z)))
        gm = GaussianMixture(k, n_init=3, random_state=SEED).fit(Z)
        out[str(k)] = {"silhouette": float(sil), "null_mean": float(np.mean(ref)), "null_high": float(np.percentile(ref, 95)),
                       "stability": float(np.mean(aris)), "bic": float(gm.bic(Z))}
    return out


def cluster_profile(Z: np.ndarray, X: pd.DataFrame, y: pd.Series, k: int) -> list:
    labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    cols = current_features(list(X.columns))
    rows = []
    for c in range(k):
        m = labels == c
        row = {"cluster": c, "women": int(m.sum()), "diabetes_rate": float(y[m].mean())}
        row.update({col: float(X.loc[m, col].median()) for col in cols})
        rows.append(row)
    return sorted(rows, key=lambda r: r["Glucose"])


class WithClusters(BaseEstimator):
    """The base model plus cluster information, with the clusters found on the training women only.

    kind="member"  adds one 0/1 column per cluster.
    kind="distance" adds the distance to each cluster centre.
    kind="anomaly" adds one column, the isolation-forest oddness score.
    """

    def __init__(self, base, k=3, kind="member"):
        self.base, self.k, self.kind = base, k, kind

    def _z(self, X):
        cols = current_features(list(X.columns))
        F = X[cols].fillna(self.fill_)
        F = F.clip(self.lo_, self.hi_, axis=1)
        return ((F - self.mu_) / self.sd_).to_numpy()

    def _extra(self, X):
        Z = self._z(X)
        if self.kind == "anomaly":
            return pd.DataFrame({"oddness": -self.iso_.score_samples(Z)}, index=X.index)
        if self.kind == "distance":
            d = self.km_.transform(Z)
            return pd.DataFrame(d, index=X.index, columns=[f"dist{i}" for i in range(d.shape[1])])
        lab = self.km_.predict(Z)
        return pd.DataFrame({f"cl{i}": (lab == i).astype(float) for i in range(self.k)}, index=X.index)

    def fit(self, X, y):
        cols = current_features(list(X.columns))
        F = X[cols]
        self.fill_ = F.median()
        F = F.fillna(self.fill_)
        self.lo_, self.hi_ = F.quantile(0.01), F.quantile(0.99)
        F = F.clip(self.lo_, self.hi_, axis=1)
        self.mu_, self.sd_ = F.mean(), F.std()
        Z = ((F - self.mu_) / self.sd_).to_numpy()
        self.km_ = KMeans(self.k, n_init=10, random_state=SEED).fit(Z)
        self.iso_ = IsolationForest(n_estimators=200, random_state=SEED).fit(Z)
        self.model_ = clone(self.base).fit(pd.concat([X, self._extra(X)], axis=1), y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(pd.concat([X, self._extra(X)], axis=1))


def feature_test(X: pd.DataFrame, y: pd.Series, cv, k: int) -> dict:
    """Do cluster memberships, distances or an oddness score improve the published model? Paired over the same splits."""
    n_test = int(len(y) / 5)
    base_auc, _, _ = nl.run_splits(nl.curves_model(), X, y, cv)
    out = {"base_auc": float(base_auc.mean())}
    for kind in ["member", "distance", "anomaly"]:
        auc, _, _ = nl.run_splits(WithClusters(nl.curves_model(), k=k, kind=kind), X, y, cv)
        out[kind] = {"auc": float(auc.mean()), **corrected_ttest(auc, base_auc, len(y) - n_test, n_test)}
    return out


# ----------------------------------------------------------------------------- odd records
def oddness(X_all: pd.DataFrame, Z: np.ndarray) -> pd.DataFrame:
    """Three different measures of how unusual each woman is, plus a plain-rules check on the raw values."""
    iso = -IsolationForest(n_estimators=300, random_state=SEED).fit(Z).score_samples(Z)
    lof = -LocalOutlierFactor(n_neighbors=20).fit(Z).negative_outlier_factor_
    mu = Z.mean(axis=0)
    inv = np.linalg.inv(np.cov(Z, rowvar=False))
    maha = np.array([mahalanobis(z, mu, inv) for z in Z])
    return pd.DataFrame({"isolation": iso, "lof": lof, "mahalanobis": maha}, index=X_all.index)


def odd_records(X_all: pd.DataFrame, scores: pd.DataFrame, top: int = 15) -> tuple:
    """The most unusual women by consensus rank, and what makes each one unusual (largest standardised value)."""
    ranks = scores.rank(ascending=False)
    consensus = ranks.mean(axis=1).sort_values()
    z_all = (X_all - X_all.mean()) / X_all.std()
    rows = []
    for idx in consensus.index[:top]:
        col = z_all.loc[idx].abs().idxmax()
        rows.append({"row": int(idx), "consensus_rank": float(consensus[idx]),
                     "most_unusual_value": col, "value": float(X_all.loc[idx, col]), "z": float(z_all.loc[idx, col]),
                     **{c: (None if pd.isna(X_all.loc[idx, c]) else float(X_all.loc[idx, c])) for c in X_all.columns}})
    return rows, consensus


def error_check(y: np.ndarray, p: np.ndarray, score: np.ndarray, top_share: float = 0.05) -> dict:
    """Do unusual women get worse predictions? Compares the most unusual 5% with everyone else."""
    cut = np.quantile(score, 1 - top_share)
    odd = score >= cut
    err = np.abs(y - p)
    from scipy.stats import spearmanr
    return {"n_odd": int(odd.sum()), "rate_odd": float(y[odd].mean()), "rate_rest": float(y[~odd].mean()),
            "mean_error_odd": float(err[odd].mean()), "mean_error_rest": float(err[~odd].mean()),
            "auc_odd": float(roc_auc_score(y[odd], p[odd])) if 0 < y[odd].sum() < odd.sum() else None,
            "auc_rest": float(roc_auc_score(y[~odd], p[~odd])),
            "spearman_error_vs_oddness": float(spearmanr(score, err)[0])}


def rule_flags(X_all: pd.DataFrame) -> dict:
    """Values that are suspicious on plain medical or measurement grounds, whatever the other inputs say."""
    return {"insulin_over_600": int((X_all.Insulin > 600).sum()), "skin_over_60": int((X_all.SkinThickness > 60).sum()),
            "bmi_over_60": int((X_all.BMI > 60).sum()), "pregnancies_over_14": int((X_all.Pregnancies > 14).sum()),
            "bp_under_40": int((X_all.BloodPressure < 40).sum()), "glucose_under_50": int((X_all.Glucose < 50).sum())}


# ----------------------------------------------------------------------------- "does she look like the training women?"
def reference(X_all: pd.DataFrame) -> dict:
    """What the page needs to measure how far a woman's inputs are from the typical Pima woman (Mahalanobis distance)."""
    cols = current_features(list(X_all.columns))
    F = X_all[cols]
    fill = F.median()
    F = F.fillna(fill)
    lo, hi = F.quantile(0.01), F.quantile(0.99)
    F = F.clip(lo, hi, axis=1)
    mu, sd = F.mean(), F.std()
    Z = ((F - mu) / sd).to_numpy()
    inv = np.linalg.inv(np.cov(Z, rowvar=False))
    d = np.sqrt(np.einsum("ij,jk,ik->i", Z, inv, Z))
    return {"features": cols, "fill": [float(fill[c]) for c in cols], "lo": [float(lo[c]) for c in cols],
            "hi": [float(hi[c]) for c in cols], "mu": [float(mu[c]) for c in cols], "sd": [float(sd[c]) for c in cols],
            "inv": [[float(v) for v in row] for row in inv], "q95": float(np.quantile(d, 0.95)), "q99": float(np.quantile(d, 0.99)),
            "distances": d}


def export_reference(ref: dict) -> dict:
    """The small reference the page needs to compute the same distance in JavaScript."""
    keep = ["features", "fill", "lo", "hi", "mu", "sd", "inv", "q95", "q99"]
    return {k: ref[k] for k in keep}


def distance(ref: dict, table: pd.DataFrame) -> np.ndarray:
    """Distance of each row from the typical woman. A missing input is filled with the Pima median, as the model does."""
    F = table.reindex(columns=ref["features"]).fillna(dict(zip(ref["features"], ref["fill"])))
    F = F.clip(pd.Series(ref["lo"], index=ref["features"]), pd.Series(ref["hi"], index=ref["features"]), axis=1)
    Z = ((F - pd.Series(ref["mu"], index=ref["features"])) / pd.Series(ref["sd"], index=ref["features"])).to_numpy()
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, np.array(ref["inv"]), Z))


def reach_check(X_all: pd.DataFrame, y: np.ndarray, p_oof: np.ndarray) -> dict:
    """Would a 'looks unlike the training women' flag have warned about the NHANES women of Milestone 8?"""
    from . import external
    from scipy.stats import spearmanr
    from . import uncertainty
    import json
    ref = reference(X_all)
    nh = external.predictions()
    d_nh = distance(ref, nh)
    flag = d_nh > ref["q95"]
    yn, pn = nh["Outcome"].to_numpy(), nh["p"].to_numpy()

    def group(m):
        return {"n": int(m.sum()), "positives": int(yn[m].sum()), "mean_predicted": float(pn[m].mean()), "observed": float(yn[m].mean()),
                "auc": float(roc_auc_score(yn[m], pn[m])) if 0 < yn[m].sum() < m.sum() else None}

    # are unusual women also the ones whose range is widest? (Milestone 7 tables)
    t = uncertainty.tables_from_json(json.loads((ROOT / "reports" / "bootstrap.json").read_text()))
    cols = t["features"]
    filled = X_all[cols].fillna(X_all[cols].median()).to_numpy()
    pr = uncertainty.risks_from_tables(t, filled)
    width = np.percentile(pr, 97.5, axis=1) - np.percentile(pr, 2.5, axis=1)
    return {"q95": ref["q95"], "q99": ref["q99"], "median_pima": float(np.median(ref["distances"])),
            "median_nhanes": float(np.median(d_nh)), "nhanes_over_q95": float(flag.mean()), "nhanes_over_q99": float((d_nh > ref["q99"]).mean()),
            "flagged": group(flag), "typical": group(~flag),
            "width_vs_distance": float(spearmanr(ref["distances"], width)[0]),
            "pima_over_q95_width": float(np.median(width[ref["distances"] > ref["q95"]])),
            "pima_rest_width": float(np.median(width[ref["distances"] <= ref["q95"]]))}


def eight_input_check(X_all: pd.DataFrame) -> dict:
    """The same oddness check using all eight raw columns (insulin and skinfold included, as recorded), to see
    whether the known extreme readings stand out when they are allowed to count."""
    F = X_all.fillna(X_all.median())
    Z = ((F - F.mean()) / F.std()).to_numpy()
    sc = oddness(X_all, Z)
    rank = sc.rank(ascending=False).mean(axis=1).rank()
    picks = {"insulin_846": X_all.Insulin.idxmax(), "skinfold_99": X_all.SkinThickness.idxmax(), "bmi_67": X_all.BMI.idxmax(),
             "pregnancies_17": X_all.Pregnancies.idxmax(), "bp_24": X_all.BloodPressure.idxmin()}
    return {k: {"row": int(i), "rank_of_768": float(rank[i])} for k, i in picks.items()}


def run(cv, p_oof: np.ndarray, rerun_maps: bool = False) -> dict:
    X_all, y = load_data()
    cols = current_features(list(X_all.columns))
    Xc = X_all[cols]
    Z = prepare(X_all)
    yv = y.to_numpy()

    res = {"pca": pca_summary(Z, cols)}
    if rerun_maps or not MAPS_CSV.exists():
        frame, trust = make_maps(Z, yv)
        frame.to_csv(MAPS_CSV, index=False)
        res["trust"] = trust
    else:
        frame = pd.read_csv(MAPS_CSV)
        res["trust"] = {"pca": float(trustworthiness(Z, frame[["pc1", "pc2"]].to_numpy(), n_neighbors=15)),
                        "umap": float(trustworthiness(Z, frame[["umap1", "umap2"]].to_numpy(), n_neighbors=15))}
    scan = cluster_scan(Z)
    res["scan"] = scan
    best = max(scan, key=lambda k: scan[k]["silhouette"])
    res["best_k"] = int(best)
    res["profile_k"] = 3
    res["profiles"] = {str(k): cluster_profile(Z, X_all, y, k) for k in sorted({int(best), 3})}
    res["feature_test"] = feature_test(Xc, y, cv, 3)

    scores = oddness(X_all, Z)
    rows, consensus = odd_records(X_all, scores)
    rank6 = scores.rank(ascending=False).mean(axis=1).rank()
    res["odd_records"] = rows
    res["rules"] = rule_flags(X_all)
    res["error_check"] = {c: error_check(yv, p_oof, scores[c].to_numpy()) for c in scores}
    res["agreement"] = {f"{a}_{b}": float(scores[a].corr(scores[b], method="spearman"))
                        for a, b in [("isolation", "lof"), ("isolation", "mahalanobis"), ("lof", "mahalanobis")]}
    res["eight_inputs"] = eight_input_check(X_all)
    res["six_inputs"] = {k: {"row": v["row"], "rank_of_768": float(rank6[v["row"]])} for k, v in res["eight_inputs"].items()}
    res["reach"] = reach_check(X_all, yv, p_oof)
    # does removing the most unusual women from TRAINING (tested on everyone) change the model?
    flagged = consensus.index[: int(0.02 * len(y))]
    res["n_removed"] = int(len(flagged))
    keep = ~X_all.index.isin(flagged)
    aucs_all, aucs_trim = [], []
    for tr, te in cv.split(Xc, y):
        m_all = clone(nl.curves_model()).fit(Xc.iloc[tr], y.iloc[tr])
        tr_trim = [i for i in tr if keep[i]]
        m_trim = clone(nl.curves_model()).fit(Xc.iloc[tr_trim], y.iloc[tr_trim])
        aucs_all.append(roc_auc_score(yv[te], m_all.predict_proba(Xc.iloc[te])[:, 1]))
        aucs_trim.append(roc_auc_score(yv[te], m_trim.predict_proba(Xc.iloc[te])[:, 1]))
    n_test = int(len(y) / 5)
    res["trim_test"] = {"auc_all": float(np.mean(aucs_all)), "auc_trimmed": float(np.mean(aucs_trim)),
                        **corrected_ttest(aucs_trim, aucs_all, len(y) - n_test, n_test)}
    return res
