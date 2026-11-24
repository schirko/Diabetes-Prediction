"""Charts shared by the notebooks and the project page.

Each function returns a matplotlib Figure. build_site saves them as PNGs;
the notebooks just display them. All of them read reports/metrics.json,
reports/oof.csv and reports/model.json, so run `python -m src.evaluate` first.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import LABELS, ROOT, load_data
from .evaluate import confusion_at

REPORTS = ROOT / "reports"
GREEN, WARM, INK, MUTED, GRID = "#23855A", "#B26B16", "#1C211E", "#5F6762", "#E6E4DC"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def _metrics():
    return json.loads((REPORTS / "metrics.json").read_text())


def missing_values():
    """Share of each measurement recorded as an impossible zero."""
    X, _ = load_data()
    pct = (X.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    bars = ax.barh([LABELS[c][0] for c in pct.index], pct.values, height=0.5, color=GREEN)
    for b, v, n in zip(bars, pct.values, X.isna().sum()[pct.index]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%  ({n} of {len(X)})", va="center", color=INK)
    ax.set_xlim(0, 70)
    ax.set_xlabel("Share of women with no real reading (%)")
    ax.set_title("Zeros that really mean 'not measured'")
    ax.grid(axis="y", visible=False)
    return fig


def lucky_split():
    """The original recipe scored on 200 different random splits."""
    ls = _metrics()["lucky_split"]
    accs = np.array(ls["all"]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(accs, bins=np.arange(67.5, 85.5, 1.3), color=GREEN, alpha=0.85, edgecolor="white", linewidth=2)
    top = ax.get_ylim()[1] * 1.35          # headroom so the two labels sit above the bars
    ax.set_ylim(0, top)
    ax.axvline(ls["original_seed_accuracy"] * 100, color=WARM, linewidth=2)
    ax.text(ls["original_seed_accuracy"] * 100 + 0.3, top * 0.97, "the split the original\nnotebook happened to use",
            color=INK, va="top", fontsize=10)
    ax.axvline(ls["mean"] * 100, color=INK, linewidth=1.2)
    ax.text(ls["mean"] * 100 - 0.3, top * 0.97, f"average\n{ls['mean']*100:.0f}%", color=INK, va="top", ha="right",
            fontsize=10)
    ax.set_xlabel("Accuracy on the 20% test set (%)")
    ax.set_ylabel("Number of splits")
    ax.set_title("Same method, 200 different random splits")
    ax.grid(axis="x", visible=False)
    return fig


def model_comparison():
    """Mean AUC with a one-standard-deviation whisker for every candidate."""
    comp = pd.DataFrame(_metrics()["comparison"]).T.sort_values("auc")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ypos = np.arange(len(comp))
    colors = [WARM if n == "Logistic regression" else GREEN for n in comp.index]
    ax.errorbar(comp["auc"], ypos, xerr=comp["auc_sd"], fmt="none", ecolor=MUTED, elinewidth=1.4, capsize=3)
    ax.scatter(comp["auc"], ypos, s=70, c=colors, zorder=3, edgecolor="white", linewidth=1.5)
    for y, (name, r) in zip(ypos, comp.iterrows()):
        ax.text(r["auc"] + r["auc_sd"] + 0.008, y, f"{r['auc']:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(ypos, comp.index)
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel("AUC: chance of ranking a woman with diabetes above one without (0.5 = coin flip)")
    ax.set_title("Six models, the same 50 train/test splits")
    ax.grid(axis="y", visible=False)
    return fig


def threshold_tradeoff(which: str = "baseline"):
    """Cases caught and healthy women correctly cleared, for every cut-off.

    which="baseline" is the eight-input Milestone 1 model (notebook 3);
    which="current" is the published model (the project page).
    """
    m = _metrics()
    oof = pd.read_csv(REPORTS / "oof.csv")
    y = oof["outcome"]
    p = oof["p_current" if which == "current" else "p_model"].to_numpy()
    cut = (m["current"] if which == "current" else m)["threshold_screening"]["threshold"]
    ts = np.arange(0.02, 0.99, 0.01)
    sens = [confusion_at(y, p, t)["sensitivity"] * 100 for t in ts]
    spec = [confusion_at(y, p, t)["specificity"] * 100 for t in ts]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ts, sens, color=GREEN, linewidth=2)
    ax.plot(ts, spec, color=WARM, linewidth=2)
    ax.text(0.80, 22, "Cases caught\n(sensitivity)", color=INK, fontsize=10, ha="center")
    ax.text(0.80, 86, "Healthy women cleared\n(specificity)", color=INK, fontsize=10, ha="center", va="top")
    if which == "current":
        marks = [(c["cutoff"], f"{r}\u00d7") for r, c in m["m5"]["costs"]["ratios"].items()] + [(0.5, "0.50")]
        marks = [(t, lab) for t, lab in marks]
        for t, label in marks:
            ax.axvline(t, color=GREEN if label.endswith("\u00d7") else MUTED, linewidth=1)
            ax.text(t + 0.008, 3, label, color=INK, fontsize=9.5)
    else:
        for t, label in [(cut, "screening\ncut-off"), (0.5, "default\n0.50")]:
            ax.axvline(t, color=MUTED, linewidth=1)
            ax.text(t + 0.01, 3, label, color=INK, fontsize=9.5)
    ax.set_xlabel("Cut-off: flag everyone whose predicted risk is at or above this")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 102)
    ax.set_title("Moving the cut-off trades missed cases for false alarms")
    return fig


def calibration_plot():
    """Predicted risk against the share who actually had diabetes."""
    cal = _metrics()["calibration"]
    pred = [b["predicted"] * 100 for b in cal["bins"]]
    obs = [b["observed"] * 100 for b in cal["bins"]]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1)
    ax.plot(pred, obs, color=GREEN, linewidth=2, marker="o", markersize=7, markeredgecolor="white",
            markeredgewidth=1.5)
    ax.text(62, 52, "perfect\ncalibration", color=MUTED, fontsize=9.5)
    ax.set_xlabel("Predicted risk (%), 8 equal-sized groups")
    ax.set_ylabel("Actually had diabetes (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Predicted vs actual")
    return fig


def odds_ratios(model=None):
    """How much each input moves the odds, per one standard deviation.

    Pass a model dict from export_model(); without one, the current model in reports/ is used.
    """
    model = model or json.loads((REPORTS / "model.json").read_text())
    s = pd.Series(model["odds_ratio_per_sd"], index=[LABELS[f][0] for f in model["features"]]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.33 * len(s)))
    ax.axvline(1, color=MUTED, linewidth=1)
    ax.hlines(np.arange(len(s)), 1, s.values, color=GREEN, linewidth=2)
    ax.scatter(s.values, np.arange(len(s)), s=70, color=GREEN, zorder=3, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(s.values):
        ax.text(v + 0.06 if v >= 1 else v - 0.06, i, f"×{v:.2f}", va="center", ha="left" if v >= 1 else "right",
                color=INK, fontsize=10)
    ax.set_yticks(np.arange(len(s)), s.index)
    ax.set_xlim(0.6, 3.5)
    ax.set_xlabel("Odds multiplier for a one-standard-deviation increase (1 = no effect)")
    ax.set_title("What the model leans on")
    ax.grid(axis="y", visible=False)
    return fig


def strategy_comparison():
    """Milestone 2: every missing-data strategy on the same 50 splits."""
    comp = pd.DataFrame(_metrics()["m2"]["strategies"]).T.sort_values("auc")
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ypos = np.arange(len(comp))
    colors = [WARM if n.startswith("Median fill (") else GREEN for n in comp.index]
    ax.errorbar(comp["auc"], ypos, xerr=comp["auc_sd"], fmt="none", ecolor=MUTED, elinewidth=1.4, capsize=3)
    ax.scatter(comp["auc"], ypos, s=70, c=colors, zorder=3, edgecolor="white", linewidth=1.5)
    for yy, (_, r) in zip(ypos, comp.iterrows()):
        ax.text(r["auc"] + r["auc_sd"] + 0.003, yy, f"{r['auc']:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(ypos, comp.index)
    ax.set_xlim(0.78, 0.885)
    ax.set_xlabel("AUC (whisker = split-to-split wobble). Orange is the Milestone 1 baseline.")
    ax.set_title("Seven ways to handle the gaps")
    ax.grid(axis="y", visible=False)
    return fig


def recovery():
    """Milestone 2: how well each method guesses readings we deliberately hid."""
    rec = _metrics()["m2"]["recovery"]
    methods = ["Median", "Nearest neighbours", "Iterative"]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    width = 0.34
    for k, (col, color) in enumerate([("Insulin", GREEN), ("SkinThickness", WARM)]):
        vals = [max(rec[col][m]["r2"], 0) * 100 for m in methods]
        bars = ax.bar(np.arange(3) + (k - 0.5) * (width + 0.03), vals, width, color=color,
                      label=LABELS[col][0].split(",")[0])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", color=INK, fontsize=10)
    ax.set_xticks(np.arange(3), methods)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of the variation recovered (%)")
    ax.set_title("Can a hidden reading be guessed back?")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def imputation_spread():
    """Milestone 2: how far each woman's estimate moves across 30 plausible fills."""
    sp = _metrics()["m2"]["spread"]
    width, miss = np.array(sp["width"]), np.array(sp["insulin_missing"])
    bins = np.arange(0, 21, 1)
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.hist(np.clip(width[~miss], 0, 20), bins=bins, histtype="step", linewidth=2, color=GREEN,
            label="Insulin was measured")
    ax.hist(np.clip(width[miss], 0, 20), bins=bins, histtype="step", linewidth=2, color=WARM,
            label="Insulin was missing")
    ax.set_xlabel("How far her estimate moves across the 30 fills (percentage points; last bar = 20 or more)")
    ax.set_ylabel("Number of women")
    ax.set_title("One woman, 30 plausible ways to fill the gaps")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return fig


def m3_models():
    """Milestone 3: every candidate on overall AUC and on AUC for women 45 and over."""
    comp = pd.DataFrame(_metrics()["m3"]["comparison"]).T.sort_values("auc")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    ypos = np.arange(len(comp))
    colors = [WARM if n.startswith("Straight") else GREEN for n in comp.index]
    for ax, col, title, lim in [(axes[0], "auc", "All women", (0.82, 0.86)),
                                (axes[1], "auc_45plus", "Women 45 and over", (0.72, 0.80))]:
        ax.scatter(comp[col], ypos, s=60, c=colors, zorder=3, edgecolor="white", linewidth=1.5)
        for yy, v in zip(ypos, comp[col]):
            ax.text(v + (lim[1] - lim[0]) * 0.03, yy, f"{v:.3f}", va="center", color=INK, fontsize=9.5)
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC")
    axes[0].set_yticks(ypos, comp.index)
    axes[0].set_xticks([0.82, 0.83, 0.84, 0.85])
    axes[1].set_xticks([0.72, 0.74, 0.76, 0.78])
    fig.suptitle("Nine models, the same 50 splits. Orange is the straight-line model from Milestone 2.",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.84)
    return fig


def age_fit():
    """Milestone 3: the real diabetes rate by age, against what each model predicts."""
    fit = pd.DataFrame(_metrics()["m3"]["age_fit"]).T
    x = np.arange(len(fit))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(x, fit["straight"] * 100, color=WARM, linewidth=2, label="Straight lines")
    ax.plot(x, fit["curves"] * 100, color=GREEN, linewidth=2, label="Curves")
    ax.scatter(x, fit["actual"] * 100, s=46, color=INK, zorder=4, label="Actual rate")
    ax.text(x[-1] + 0.15, fit["straight"].iloc[-1] * 100, "straight\nlines", color=INK, fontsize=9.5, va="center")
    ax.text(x[-1] + 0.15, fit["curves"].iloc[-1] * 100 - 2, "curves", color=INK, fontsize=9.5, va="center")
    ax.text(x[-1] + 0.15, fit["actual"].iloc[-1] * 100 - 7, "actual", color=INK, fontsize=9.5, va="center")
    ax.set_xticks(x, [f"{a}\n({int(n)})" for a, n in zip(fit.index, fit["women"])])
    ax.set_xlim(-0.4, len(fit) - 0.1)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Age group (number of women)")
    ax.set_ylabel("Diagnosed with diabetes (%)")
    ax.set_title("Risk peaks around 50; a straight line can't turn down")
    return fig


def shapes():
    """Milestone 3: how glucose, BMI and age move the odds, curved model against straight lines."""
    sh = _metrics()["m3"]["shapes"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, f in zip(axes, ["Glucose", "BMI", "Age"]):
        g = np.array(sh[f]["grid"])
        ax.plot(g, np.exp(sh[f]["straight"]), color=WARM, linewidth=2)
        ax.plot(g, np.exp(sh[f]["curves"]), color=GREEN, linewidth=2)
        ax.axhline(1, color=MUTED, linewidth=1)
        ax.set_yscale("log")
        ax.set_title(LABELS[f][0].split(",")[0], fontsize=11.5)
        ax.set_xlabel(LABELS[f][1])
    axes[0].set_ylabel("Odds vs a typical woman (log scale)")
    axes[0].set_ylim(0.015, 20)
    axes[0].set_yticks([0.02, 0.05, 0.1, 0.25, 1, 4, 10], ["×0.02", "×0.05", "×0.1", "×0.25", "×1", "×4", "×10"])
    axes[0].minorticks_off()
    for ax in axes[1:]:
        ax.minorticks_off()
    age = np.array(sh["Age"]["grid"])
    axes[2].text(age[-1], np.exp(sh["Age"]["straight"][-1]) * 1.3, "straight", color=INK, fontsize=9.5, ha="right")
    axes[2].text(age[-1], np.exp(sh["Age"]["curves"][-1]) * 0.6, "curved", color=INK, fontsize=9.5, ha="right", va="top")
    fig.suptitle("What each input does to the odds", x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.82)
    return fig


def m4_contest():
    """Milestone 4: each model's AUC minus the published model's, with the corrected 95% interval."""
    t = pd.DataFrame(_metrics()["m4"]["table"]).T
    t = t.drop(index="Curves (M3, as published)").sort_values("vs_published")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ypos = np.arange(len(t))
    worse = (t["vs_high"] < 0).to_numpy()
    for yy, (_, r), bad in zip(ypos, t.iterrows(), worse):
        color = WARM if bad else GREEN
        ax.hlines(yy, r["vs_low"] * 100, r["vs_high"] * 100, color=color, linewidth=2)
        ax.scatter(r["vs_published"] * 100, yy, s=60, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(5.6, yy, f"{r['auc']:.3f}", va="center", ha="right", color=INK, fontsize=10)
    ax.axvline(0, color=INK, linewidth=1)
    ax.text(0.2, len(t) - 0.35, "published model", color=INK, fontsize=9.5)
    ax.text(5.6, len(t) - 0.35, "AUC", color=MUTED, fontsize=9.5, ha="right")
    ax.set_yticks(ypos, t.index)
    ax.set_xlim(-5.5, 6)
    ax.set_ylim(-0.6, len(t) - 0.1)
    ax.set_xlabel("AUC difference from the published model, in hundredths (bars: 95% interval)")
    ax.set_title("No tuned model beats the published one")
    ax.grid(axis="y", visible=False)
    return fig


def m5_rebalance():
    """Milestone 5: rebalancing the training data does not help; it only distorts the risks."""
    t = pd.DataFrame(_metrics()["m5"]["rebalancing"]).T
    names = ["Published model", "Class weights", "SMOTE"]
    t = t.loc[names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ypos = np.arange(len(t))[::-1]
    colors = [GREEN, WARM, WARM]
    a.barh(ypos, t["false_alarms_for_85"], color=colors, height=0.55)
    for yy, v in zip(ypos, t["false_alarms_for_85"]):
        a.text(v + 2, yy, f"{v:.0f}", va="center", color=INK)
    a.set_yticks(ypos, names)
    a.set_xlim(0, 200)
    a.set_xlabel("False alarms needed to catch 85% of cases")
    a.set_title("Same ranking", fontsize=12)
    a.grid(axis="y", visible=False)
    b.scatter(t["mean_risk"] * 100, ypos, color=colors, s=70, zorder=3, edgecolor="white", linewidth=1.5)
    for yy, v in zip(ypos, t["mean_risk"]):
        b.text(v * 100 + 1.5, yy, f"{v * 100:.0f}%", va="center", color=INK)
    b.axvline(_metrics()["prevalence"] * 100, color=INK, linewidth=1)
    b.text(_metrics()["prevalence"] * 100 + 0.6, len(t) - 0.55, "real rate 35%", color=INK, fontsize=9.5)
    b.set_yticks(ypos, [""] * len(t))
    b.set_xlim(25, 55)
    b.set_ylim(-0.6, len(t) - 0.2)
    b.set_xlabel("Average predicted risk")
    b.set_title("Distorted risks", fontsize=12)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def m5_cost():
    """Milestone 5: cost per 100 women at every cut-off, for each stated cost of a missed case."""
    m5 = _metrics()["m5"]
    grid = np.array(m5["costs"]["cost_curves"]["grid"])
    n_all = m5["costs"]["ratios"]["2"]["cost_per_100"]["Test everybody"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    for ax, (r, v) in zip(axes, m5["costs"]["ratios"].items()):
        ax.plot(grid, m5["costs"]["cost_curves"][r], color=GREEN, linewidth=2)
        ax.axhline(n_all, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.axvline(m5["costs"]["old_cutoff"], color=WARM, linewidth=1, linestyle=(0, (4, 3)))
        c = v["cutoff"]
        y_at = v["cost_per_100"]["Cost-based cut-off"]
        ax.scatter([c], [y_at], color=GREEN, s=70, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(c + 0.02, y_at - 11, f"cut-off {c:.2f}", color=INK, fontsize=9.5)
        ax.set_title(f"Missed case = {r}\u00d7", fontsize=12)
        ax.set_xlabel("Cut-off")
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Cost per 100 women\n(in false-alarm units)")
    axes[0].text(0.88, n_all + 2, "test everybody", color=MUTED, fontsize=9, ha="right")
    axes[0].text(m5["costs"]["old_cutoff"] + 0.015, 93, "old 85% rule", color=WARM, fontsize=9)
    fig.tight_layout()
    return fig


def m5_decision_curve():
    """Milestone 5: net benefit of the model against testing everybody or nobody."""
    m5 = _metrics()["m5"]["costs"]
    d = m5["decision_curve"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(d["pt"], d["model"], color=GREEN, linewidth=2.2)
    ax.plot(d["pt"], d["all"], color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)))
    ax.axhline(0, color=INK, linewidth=1)
    for r, v in m5["ratios"].items():
        ax.axvline(v["cutoff"], color=GREEN, linewidth=0.8, alpha=0.6)
        ax.text(v["cutoff"] + 0.007, 0.37, f"{r}\u00d7", color=INK, fontsize=9.5)
    ax.text(0.5, 0.16, "Use the model", color=GREEN, fontsize=10, fontweight="bold")
    ax.text(0.43, -0.05, "Test everybody", color=MUTED, fontsize=10)
    ax.text(0.5, 0.012, "Test nobody", color=INK, fontsize=10)
    ax.set_xlim(0.02, 0.8)
    ax.set_ylim(-0.08, 0.42)
    ax.set_xlabel("Cut-off (flag a woman at or above this risk)")
    ax.set_ylabel("Net benefit per woman")
    ax.set_title("Net benefit of the model against testing everybody or nobody")
    return fig


def m5_bands():
    """Milestone 5: share of cases caught in each age band, old 85% rule against the cost-based cut-offs."""
    b = _metrics()["m5"]["bands"]
    bands = [r["band"] for r in b["old_cutoff"]]
    series = [("Old 85% rule (0.28)", [r["sensitivity"] for r in b["old_cutoff"]], WARM)]
    shades = {"2": "#9CCBB5", "5": GREEN, "10": "#15563A"}
    for r, v in b["by_ratio"].items():
        series.append((f"{r}\u00d7 cut-off ({_metrics()['m5']['costs']['ratios'][r]['cutoff']:.2f})",
                       [x["sensitivity"] for x in v["at_cost_cutoff"]], shades[r]))
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    w = 0.2
    for i, (name, vals, color) in enumerate(series):
        xs = np.arange(len(bands)) + (i - 1.5) * w
        ax.bar(xs, [v * 100 for v in vals], width=w * 0.92, color=color, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v * 100 + 1.2, f"{v * 100:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(bands)), [f"Age {x}" for x in bands])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percent of cases caught")
    ax.set_title("Women under 30: 62% to 95% of cases caught, depending on the cut-off")
    ax.legend(loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="x", visible=False)
    return fig


def m6_importance():
    """Milestone 6: which inputs matter. Permutation importance (left) and SHAP share for two model types (right)."""
    m6 = _metrics()["m6"]
    names = {"Pregnancies": "Pregnancies", "Glucose": "Glucose", "BloodPressure": "Blood pressure", "BMI": "BMI",
             "DiabetesPedigreeFunction": "Family history", "Age": "Age"}
    perm = m6["permutation"]
    order = sorted(perm, key=lambda f: perm[f]["mean"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1, 1.05]})
    ys = np.arange(len(order))
    means = np.array([perm[f]["mean"] for f in order])
    lows = np.array([perm[f]["low"] for f in order])
    highs = np.array([perm[f]["high"] for f in order])
    a1.barh(ys, means, color=GREEN, height=0.55)
    a1.errorbar(means, ys, xerr=[means - lows, highs - means], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    a1.axvline(0, color=INK, linewidth=1)
    a1.set_yticks(ys, [names[f] for f in order])
    a1.set_xlabel("Drop in AUC when the input is scrambled")
    a1.set_title("Scrambling glucose costs the most")
    a1.grid(axis="y", visible=False)
    pub, bst = m6["shap"]["share"], m6["boosted"]["boosted_share"]
    order2 = sorted(pub, key=lambda f: pub[f])
    ys2 = np.arange(len(order2))
    a2.barh(ys2 + 0.19, [pub[f] * 100 for f in order2], height=0.36, color=GREEN, label="Published model")
    a2.barh(ys2 - 0.19, [bst[f] * 100 for f in order2], height=0.36, color=WARM, label="Boosted trees (XGBoost)")
    a2.set_yticks(ys2, [names[f] for f in order2])
    a2.set_xlabel("Share of total SHAP importance (%)")
    a2.set_title("Two model types rank the inputs alike")
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def m6_curves():
    """Milestone 6: partial dependence (average) and ICE (individual women) for glucose, BMI and age."""
    from . import explain
    from .modeling import current_features
    X, y = load_data()
    Xc = X[current_features(list(X.columns))]
    model = explain.fit_current(Xc, y)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, (f, title, unit) in zip(axes, [("Glucose", "Glucose", "mg/dL"), ("BMI", "BMI", "kg/m\u00b2"), ("Age", "Age", "years")]):
        grid, avg, ice = explain.partial_dependence(model, Xc, f)
        for row in ice:
            ax.plot(grid, row * 100, color=GREEN, alpha=0.16, linewidth=1)
        ax.plot(grid, avg * 100, color=INK, linewidth=2.4)
        ax.set_xlabel(f"{title} ({unit})")
        ax.set_title(title)
        if f == "BMI":
            ax.axvline(30, color=WARM, linewidth=1.2, linestyle="--")
            ax.text(30.6, 93, "flat stretch\nstarts near 30", color=WARM, fontsize=9, va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
    axes[0].set_ylabel("Predicted risk (%)")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    return fig


def m6_counterfactual():
    """Milestone 6: share of flagged women a realistic change in glucose or BMI would clear."""
    cf = _metrics()["m6"]["counterfactual"]
    ratios = list(cf)
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    w = 0.25
    series = [("Glucose lower by up to 20 mg/dL", "glucose_alone", GREEN), ("BMI lower by up to 10%", "bmi_alone", WARM),
              ("Either one", "either", "#15563A")]
    for i, (name, key, color) in enumerate(series):
        xs = np.arange(len(ratios)) + (i - 1) * w
        vals = [cf[r][key] * 100 for r in ratios]
        ax.bar(xs, vals, width=w * 0.92, color=color, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(ratios)), [f"{r}\u00d7 cut-off ({cf[r]['cutoff']:.2f})\n{cf[r]['flagged']} of 768 flagged" for r in ratios])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of flagged women cleared")
    ax.set_title("Share of flagged women a realistic change would clear")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def m7_ranges():
    """Milestone 7: the 95% bootstrap range around every woman's estimated risk, women sorted by risk."""
    from . import uncertainty
    X, y = load_data()
    from .modeling import current_features
    Xc = X[current_features(list(X.columns))]
    t = uncertainty.tables_from_json(json.loads((REPORTS / "bootstrap.json").read_text()))
    filled = Xc.fillna(Xc.median())[t["features"]].to_numpy()
    p = uncertainty.risks_from_tables(t, filled)
    lo, hi = np.percentile(p, 2.5, axis=1), np.percentile(p, 97.5, axis=1)
    point = uncertainty.fit_current(Xc, y).predict_proba(Xc)[:, 1]
    order = np.argsort(point)
    xs = np.arange(len(order))
    cut = _metrics()["m5"]["published"]["cutoff"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.fill_between(xs, lo[order] * 100, hi[order] * 100, color=GREEN, alpha=0.25, linewidth=0, label="95% range")
    ax.plot(xs, point[order] * 100, color=GREEN, linewidth=2, label="Estimated risk")
    ax.axhline(cut * 100, color=INK, linewidth=1, linestyle="--")
    ax.text(len(order) * 0.02, cut * 100 + 2.5, f"cut-off {cut:.2f}", color=INK, fontsize=9.5)
    ax.set_xlabel("768 women, sorted from lowest to highest estimated risk")
    ax.set_ylabel("Risk (%)")
    ax.set_xlim(0, len(order))
    ax.set_ylim(0, 100)
    ax.set_title("Ranges are narrow for low risks and widest from about 30% up")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 0.92))
    return fig


def m7_zones():
    """Milestone 7: where the conformal rule says 'too uncertain', drawn over the women's out-of-fold risks."""
    m7 = _metrics()["m7"]
    oof = pd.read_csv(REPORTS / "oof.csv")
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.hist(oof.loc[oof.outcome == 0, "p_current"], bins=bins, color=GREEN, alpha=0.75, label="Women without diabetes", edgecolor="white")
    ax.hist(oof.loc[oof.outcome == 1, "p_current"], bins=bins, color=WARM, alpha=0.75, label="Women with diabetes", edgecolor="white")
    top = ax.get_ylim()[1]
    th = m7["page_thresholds"]
    ax.axvspan(th["0.1"]["low"], th["0.1"]["high"], color=INK, alpha=0.10, linewidth=0)
    ax.axvspan(th["0.2"]["low"], th["0.2"]["high"], color=INK, alpha=0.22, linewidth=0)
    ax.text(0.57, top * 0.98, "lighter band: too uncertain at the 90% level\ndarker band: too uncertain at the 80% level", ha="left", va="top",
            fontsize=9.5, color=INK)
    ax.set_xlabel("Estimated risk")
    ax.set_ylabel("Women")
    ax.set_xlim(0, 1)
    ax.set_title("Women in the grey band get no confident call")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, bbox_to_anchor=(1.0, 0.80))
    ax.grid(axis="x", visible=False)
    return fig


def m7_coverage():
    """Milestone 7: does the conformal guarantee hold on unseen women, and what share of women get a call?"""
    c = _metrics()["m7"]["conformal"]
    alphas = list(c)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1, 1]})
    w = 0.34
    for j, (key, label, color) in enumerate([("cov0", "Women without diabetes", GREEN), ("cov1", "Women with diabetes", WARM)]):
        xs = np.arange(len(alphas)) + (j - 0.5) * w
        vals = [c[a][key] * 100 for a in alphas]
        a1.bar(xs, vals, width=w * 0.92, color=color, label=label)
        for x, v in zip(xs, vals):
            a1.text(x, v - 7, f"{v:.0f}%", ha="center", fontsize=10, color="white")
    for i, a in enumerate(alphas):
        a1.hlines(c[a]["target"] * 100, i - 0.45, i + 0.45, color=INK, linewidth=1.6, linestyle="--")
    a1.set_xticks(np.arange(len(alphas)), [f"{round((1 - float(a)) * 100)}% level\n(target {c[a]['target'] * 100:.0f}%)" for a in alphas])
    a1.set_ylim(0, 100)
    a1.set_ylabel("Women found in the rule's set (%)")
    a1.set_title("The guarantee holds on average")
    a1.legend(frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.38))
    a1.grid(axis="x", visible=False)
    labels = [f"{round((1 - float(a)) * 100)}% level" for a in alphas]
    no = np.array([c[a]["no"] for a in alphas]) * 100
    unc = np.array([c[a]["uncertain"] for a in alphas]) * 100
    yes = np.array([c[a]["yes"] for a in alphas]) * 100
    ys = np.arange(len(alphas))
    a2.barh(ys, no, color=GREEN, height=0.5, label="Confident no")
    a2.barh(ys, unc, left=no, color="#B9B5A8", height=0.5, label="Too uncertain")
    a2.barh(ys, yes, left=no + unc, color=WARM, height=0.5, label="Confident yes")
    for y_, n_, u_, s_ in zip(ys, no, unc, yes):
        a2.text(n_ / 2, y_, f"{n_:.0f}%", ha="center", va="center", color="white", fontsize=10)
        if u_ > 5:
            a2.text(n_ + u_ / 2, y_, f"{u_:.0f}%", ha="center", va="center", color=INK, fontsize=10)
        a2.text(n_ + u_ + s_ / 2, y_, f"{s_:.0f}%", ha="center", va="center", color="white", fontsize=10)
    a2.set_yticks(ys, labels)
    a2.set_xlim(0, 100)
    a2.set_xlabel("Share of women (%)")
    a2.set_title("What the rule says about each woman")
    a2.legend(frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.38))
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def m8_shift():
    """Milestone 8: how different are the NHANES women from the Pima women, on the model's inputs?"""
    from . import external
    X, _ = load_data()
    nh = external.load_table()
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3))
    for ax, (col, label, hi) in zip(axes, [("Age", "Age (years)", 85), ("Glucose", "Glucose (mg/dL)", 260),
                                           ("BMI", "BMI", 60), ("Pregnancies", "Pregnancies", 12)]):
        bins = np.arange(0, 13) - 0.5 if col == "Pregnancies" else np.linspace(X[col].dropna().min() * 0 + (15 if col == "Age" else 40 if col == "Glucose" else 15 if col == "BMI" else 0), hi, 25)
        ax.hist(X[col].dropna(), bins=bins, density=True, histtype="stepfilled", color=WARM, alpha=0.45, label="Pima women")
        ax.hist(nh[col].dropna(), bins=bins, density=True, histtype="stepfilled", color=GREEN, alpha=0.45, label="NHANES women")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_ylabel("Share of women")
    fig.suptitle("NHANES women are older, lighter and have had fewer pregnancies", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def m8_calibration():
    """Milestone 8: the model's ranking holds on NHANES, but its risks are far too high."""
    m8 = _metrics()["m8"]
    cur = _metrics()["current"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    names = list(m8["bands"])
    xs = np.arange(len(names))
    pred = [m8["bands"][b]["mean_predicted"] * 100 for b in names]
    obs = [m8["bands"][b]["observed"] * 100 for b in names]
    w = 0.38
    a1.bar(xs - w / 2, pred, width=w * 0.94, color=WARM, label="Model's average risk")
    a1.bar(xs + w / 2, obs, width=w * 0.94, color=GREEN, label="Share who had diabetes")
    for x, a, b in zip(xs, pred, obs):
        a1.text(x - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=9.5, color=INK)
        a1.text(x + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=9.5, color=INK)
    a1.set_xticks(xs, [f"{b}\n({m8['bands'][b]['n']} women)" for b in names], fontsize=9)
    a1.set_ylim(0, 100)
    a1.set_ylabel("Percent")
    a1.set_title("The risks run far too high on NHANES")
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a1.grid(axis="x", visible=False)
    rows = [("Pima women,\nunseen (50 splits)", cur["oof_auc"], cur["oof_auc_ci"][0], cur["oof_auc_ci"][1], WARM),
            ("NHANES women,\nthe model", m8["auc"]["value"], m8["auc"]["low"], m8["auc"]["high"], GREEN),
            ("NHANES women,\nglucose alone", m8["glucose_only"]["auc"], m8["glucose_only"]["low"], m8["glucose_only"]["high"], MUTED)]
    for i, (lab, v, lo, hi, col) in enumerate(rows):
        a2.hlines(i, lo, hi, color=col, linewidth=2.2)
        a2.plot([v], [i], "o", color=col, markersize=8)
        a2.text(hi + 0.006, i, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
    a2.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    a2.invert_yaxis()
    a2.set_xlim(0.75, 1.0)
    a2.set_xlabel("AUC (dot) with 95% range (line)")
    a2.set_title("The ranking holds up")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def m8_cutoffs():
    """Milestone 8: what the three cut-offs do per 100 women, in Pima and in NHANES."""
    m8 = _metrics()["m8"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), sharey=True)
    ratios = list(m8["cutoffs"])
    for ax, key, title in [(axes[0], "cutoffs_pima", "Pima women (35 in 100 have diabetes)"),
                           (axes[1], "cutoffs", "NHANES women (4 in 100)")]:
        xs = np.arange(len(ratios))
        caught = [m8[key][r]["caught_per_100"] for r in ratios]
        missed = [m8[key][r]["missed_per_100"] for r in ratios]
        false = [m8[key][r]["false_alarms_per_100"] for r in ratios]
        w = 0.27
        ax.bar(xs - w, caught, width=w * 0.94, color=GREEN, label="Cases caught")
        ax.bar(xs, missed, width=w * 0.94, color=WARM, label="Cases missed")
        ax.bar(xs + w, false, width=w * 0.94, color="#B9B5A8", label="False alarms")
        for x, a, b, c in zip(xs, caught, missed, false):
            ax.text(x - w, a + 1, f"{a:.0f}" if a >= 10 else f"{a:.1f}", ha="center", fontsize=9, color=INK)
            ax.text(x, b + 1, f"{b:.1f}", ha="center", fontsize=9, color=INK)
            ax.text(x + w, c + 1, f"{c:.0f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(xs, [f"{r}x\n(cut-off {m8[key][r]['cutoff']:.2f})" for r in ratios], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Women per 100")
    axes[0].set_ylim(0, 62)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


ALL = {
    "missing": missing_values,
    "lucky_split": lucky_split,
    "model_comparison": model_comparison,
    "threshold": lambda: threshold_tradeoff("current"),
    "shapes": shapes,
    "age_fit": age_fit,
    "m3_models": m3_models,
    "m4_contest": m4_contest,
    "m5_rebalance": m5_rebalance,
    "m5_cost": m5_cost,
    "m5_bands": m5_bands,
    "m6_importance": m6_importance,
    "m6_curves": m6_curves,
    "m6_counterfactual": m6_counterfactual,
    "m7_ranges": m7_ranges,
    "m7_zones": m7_zones,
    "m7_coverage": m7_coverage,
    "m8_shift": m8_shift,
    "m8_calibration": m8_calibration,
    "m8_cutoffs": m8_cutoffs,
    "strategies": strategy_comparison,
    "recovery": recovery,
    "spread": imputation_spread,
}


def save_all(out_dir):
    """Write every chart as a PNG into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in ALL.items():
        fig = make()
        fig.savefig(out_dir / f"{name}.png")
        plt.close(fig)
