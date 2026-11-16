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


ALL = {
    "missing": missing_values,
    "lucky_split": lucky_split,
    "model_comparison": model_comparison,
    "threshold": lambda: threshold_tradeoff("current"),
    "shapes": shapes,
    "age_fit": age_fit,
    "m3_models": m3_models,
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
