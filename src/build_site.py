"""Build the static project site.

    python -m src.build_site

Reads reports/ (run `python -m src.evaluate` first) and the executed notebooks,
and writes a self-contained site into SITE_DIR. Set SITE_DIR in a .env file at
the project root (see .env.example); without it the site goes to ./site.

Everything in the output folder is generated. Never edit it by hand: change
site_template/index.html, the notebooks or docs/*.md and rebuild.
"""
import base64
import datetime as dt
import html
import json
import os
import re
import shutil

import markdown
import nbformat
import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from . import figures
from .data import LABELS, ROOT

PROJECT_NAME = "Diabetes Risk Predictor"
SLUG = "diabetes-prediction"
REPO_URL = "https://github.com/schirko/Diabetes-Prediction"
REPORTS = ROOT / "reports"

NOTEBOOK_BLURBS = {
    "01_data_audit": "What is really in the table: the zeros that mean \"not measured\", how common diabetes is, "
                     "and which measurements differ between the two groups.",
    "02_honest_evaluation": "Why the original 81% can't be trusted, a leak-free pipeline, and six models compared "
                            "on the same 50 splits.",
    "03_thresholds_and_calibration": "Choosing a cut-off for screening, checking the probabilities, results by age, "
                                     "and the eight-input baseline model.",
    "04_missing_data": "Seven ways to handle the missing readings, a test of whether insulin can be guessed back, "
                       "and why the published model now leaves insulin and skinfold out.",
    "05_curved_effects": "Why the model over-flagged older women, nine curved and boosted models compared, "
                         "insulin's second hearing, and the curved model the page now runs.",
        "07_decisions_and_costs": "Does rebalancing the data help, what a missed case costs, where the cut-off should go, and "
                              "whether age groups need their own.",
    "06_model_contest": "Nine model types tuned fairly with nested cross-validation, a corrected test of the "
                        "differences, a stacked ensemble, and why the page keeps its model.",
}
DOCS = ["OVERVIEW", "GLOSSARY", "FINDINGS", "MILESTONES", "ENHANCEMENTS_AND_BUGS", "DECISIONS"]

PAGE_CSS = """
:root { --bg:#f6f5f1; --surface:#fff; --ink:#1c211e; --muted:#4a524d; --line:#e2e0d8; --accent:#2E6A4C; --code:#f3f2ec; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16.5px/1.6 Inter,system-ui,sans-serif; }
.wrap { max-width:900px; margin:0 auto; padding:8px 20px 60px; }
h1,h2,h3 { font-family:"Space Grotesk",Inter,system-ui,sans-serif; line-height:1.2; }
h1 { font-size:34px; margin:32px 0 12px; } h2 { font-size:24px; margin:44px 0 8px; } h3 { font-size:18px; }
a { color:var(--accent); }
pre { background:var(--code); border:1px solid var(--line); border-radius:10px; padding:12px 14px; overflow-x:auto;
      font:13.5px/1.5 ui-monospace,Consolas,monospace; margin:10px 0; }
code { font:0.92em ui-monospace,Consolas,monospace; background:var(--code); padding:1px 4px; border-radius:4px; }
pre code { background:none; padding:0; }
.out { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--accent); }
.out-table { overflow-x:auto; margin:10px 0; }
table { border-collapse:collapse; background:var(--surface); font-size:14px; }
th,td { border:1px solid var(--line); padding:6px 11px; text-align:left; vertical-align:top; }
.out-table td { text-align:right; font-variant-numeric:tabular-nums; }
th { background:#f1f0eb; }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:10px; background:#fff; margin:8px 0; }
blockquote { margin:16px 0; padding:12px 18px; background:var(--surface); border:1px solid var(--line);
             border-left:4px solid var(--accent); border-radius:10px; }
blockquote p { margin:0; }
.lbl { font-size:11.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:16px 0 -4px; }
footer { margin-top:48px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:14px; }
"""


def site_dir():
    """SITE_DIR from the environment or the .env file, else ./site."""
    env = ROOT / ".env"
    if "SITE_DIR" not in os.environ and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SITE_DIR="):
                os.environ["SITE_DIR"] = line.split("=", 1)[1].strip().strip('"')
    from pathlib import Path
    return Path(os.environ.get("SITE_DIR", ROOT / "site"))


def backbar() -> str:
    """The portfolio's shared back bar. Markup only; the portfolio owns the stylesheet."""
    return f"""<div class="sx-backbar">
  <a class="sx-back" href="/scott/">
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg>
    Portfolio
  </a>
  <span class="sx-crumb">{PROJECT_NAME}</span>
  <span class="sx-links"><a href="/scott/{SLUG}/">Project page</a><a href="/scott/{SLUG}/docs/OVERVIEW.html">Start here</a><a href="/scott/{SLUG}/docs/FINDINGS.html">Findings</a><a href="/scott/{SLUG}/docs/GLOSSARY.html">Glossary</a></span>
</div>"""


def shell(title: str, body: str) -> str:
    """Wrap a sub-page (notebook or doc) in the shared page frame."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{html.escape(title)} · {PROJECT_NAME}</title>
<link rel="stylesheet" href="/scott/backbar.css">
<link rel="icon" href="/scott/favicon.ico" sizes="32x32">
<link rel="icon" type="image/png" sizes="16x16" href="/scott/favicon-16.png">
<link rel="icon" type="image/png" sizes="32x32" href="/scott/favicon-32.png">
<link rel="apple-touch-icon" href="/scott/apple-touch-icon.png">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{PAGE_CSS}{HtmlFormatter(style="friendly").get_style_defs(".hl")}</style></head>
<body>{backbar()}<div class="wrap">
{body}
<footer><a href="/scott/{SLUG}/">{PROJECT_NAME}</a> · built {dt.date.today():%B %d, %Y}</footer>
</div></body></html>"""


def md(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def render_notebook(path) -> tuple:
    """Turn an executed notebook into a small HTML page. Returns (title, html)."""
    nb = nbformat.read(path, as_version=4)
    parts, title = [], path.stem
    fmt = HtmlFormatter(cssclass="hl", nowrap=False)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            m = re.match(r"#\s+(.+)", cell.source)
            if m and title == path.stem:
                title = m.group(1).strip()
            parts.append(md(re.sub(r"\]\(\.\./docs/(\w+)\.md\)", r"](../docs/\1.html)", cell.source)))
            continue
        parts.append('<div class="lbl">Code</div>' + highlight(cell.source, PythonLexer(), fmt))
        outs = []
        for o in cell.get("outputs", []):
            data = o.get("data", {})
            if o.output_type == "stream":
                outs.append(f'<pre class="out">{html.escape(o.text)}</pre>')
            elif "image/png" in data:
                outs.append(f'<img alt="Chart produced by the code above" src="data:image/png;base64,{data["image/png"]}">')
            elif "text/html" in data:
                table = re.sub(r"<style.*?</style>", "", data["text/html"], flags=re.S)
                outs.append(f'<div class="out-table">{table}</div>')
            elif "text/plain" in data:
                outs.append(f'<pre class="out">{html.escape(data["text/plain"])}</pre>')
        if outs:
            parts.append('<div class="lbl">Result</div>' + "".join(outs))
    return title, shell(title, "\n".join(parts))


def pct(v, digits=0):
    return f"{v * 100:.{digits}f}%"


def tokens(m: dict) -> dict:
    """Every number on the front page, formatted, straight from metrics.json."""
    cur, m2 = m["current"], m["m2"]              # headline numbers describe the current (published) model
    comp, ls, scr, dft = m["comparison"], m["lucky_split"], cur["threshold_screening"], cur["threshold_default"]
    old = next(b for b in cur["by_age"] if b["band"].startswith("45"))
    logistic_aucs = [v["auc"] for k, v in m2["strategies"].items() if not k.startswith("Gradient")]
    best_recovery = max(v["r2"] for v in m2["recovery"]["Insulin"].values())
    b10 = m["m3"]["b10"]
    young = next(b for b in cur["by_age"] if b["band"].startswith("21"))
    m5 = m["m5"]
    pub = m5["published"]                          # the published cut-off now comes from a stated cost (M5), not the 85% rule
    tp, fn, fp, tn = per_100([pub["counts"]["tp"], pub["counts"]["fn"], pub["counts"]["fp"], pub["counts"]["tn"]])
    _, fn_d, fp_d, _ = per_100([dft["tp"], dft["fn"], dft["fp"], dft["tn"]])
    p100 = {"p100_tp": tp, "p100_fn": fn, "p100_fp": fp, "p100_tn": tn, "p100_pos": tp + fn, "p100_neg": fp + tn,
            "p100_fn_def": fn_d, "p100_fp_def": fp_d}
    ratio_rows = {}
    for r, v in m5["costs"]["ratios"].items():
        ctp, cfn, cfp, ctn = per_100([v["counts"]["tp"], v["counts"]["fn"], v["counts"]["fp"], v["counts"]["tn"]])
        ratio_rows |= {f"r{r}_cut": f"{v['cutoff']:.2f}", f"r{r}_tp": str(ctp), f"r{r}_fn": str(cfn),
                       f"r{r}_fp": str(cfp), f"r{r}_tn": str(ctn), f"r{r}_flag": str(ctp + cfp),
                       f"r{r}_cost": f"{v['cost_per_100']['Cost-based cut-off']:.0f}",
                       f"r{r}_young": pct(m5["bands"]["by_ratio"][r]["at_cost_cutoff"][0]["sensitivity"])}
    reb, c5 = m5["rebalancing"], m5["costs"]["ratios"]["5"]
    pol5 = m5["bands"]["by_ratio"]["5"]["policies"]
    extra = ratio_rows | {
        "pub_ratio": str(pub["ratio"]), "cut85": f"{scr['threshold']:.2f}", "cut85_pct": pct(scr["threshold"]),
        "old_ratio": f"{m5['costs']['old_implied_ratio']:.1f}",
        "cost5_old": f"{c5['cost_per_100']['Old rule (catch 85%)']:.0f}",
        "cost5_all": f"{c5['cost_per_100']['Test everybody']:.0f}",
        "cost5_glu": f"{c5['cost_per_100']['Glucose of 140 or more']:.0f}",
        "cost5_none": f"{c5['cost_per_100']['Test nobody']:.0f}",
        "fa_pub": f"{reb['Published model']['false_alarms_for_85']:.0f}",
        "fa_cw": f"{reb['Class weights']['false_alarms_for_85']:.0f}",
        "fa_smote": f"{reb['SMOTE']['false_alarms_for_85']:.0f}",
        "risk_cw": pct(reb["Class weights"]["mean_risk"]), "risk_smote": pct(reb["SMOTE"]["mean_risk"]),
        "risk_pub": pct(reb["Published model"]["mean_risk"]),
        "band_single": f"{pol5['Single cost-based cut-off']['cost_per_100']:.0f}",
        "band_each": f"{pol5['Cheapest per band']['cost_per_100']:.0f}",
        "slope_45": f"{m5['bands']['calibration'][2]['slope']:.2f}",
        "slope_30": f"{m5['bands']['calibration'][0]['slope']:.2f}",
    }
    return {k: str(v) for k, v in p100.items()} | extra | {
        "project_name": PROJECT_NAME, "slug": SLUG, "repo_url": REPO_URL,
        "built": f"{dt.date.today():%B %d, %Y}",
        "lucky_mean": pct(ls["mean"]), "lucky_min": pct(ls["min"]), "lucky_max": pct(ls["max"]),
        "lucky_beat": pct(ls["share_below_original"]),
        "auc": f"{cur['oof_auc']:.2f}", "auc_lo": f"{cur['oof_auc_ci'][0]:.2f}", "auc_hi": f"{cur['oof_auc_ci'][1]:.2f}",
        "n_inputs": str(len(cur["features"])),
        "strat_lo": f"{min(logistic_aucs):.3f}", "strat_hi": f"{max(logistic_aucs):.3f}",
        "rec_insulin": pct(best_recovery), "flips": str(m2["spread"]["flips_total"]),
        "flips_pct": pct(m2["spread"]["flips_total"] / m["n"]),
        "sens_default": pct(dft["sensitivity"]), "sens_screen": pct(scr["sensitivity"]),
        "miss_default": pct(1 - dft["sensitivity"]), "cut": f"{pub['cutoff']:.2f}",
        "fn_default": str(dft["fn"]), "fp_default": str(dft["fp"]), "fp_screen": str(scr["fp"]),
        "insulin_missing": pct(m["missing"]["Insulin"] / m["n"]),
        "insulin_n": str(m["missing"]["Insulin"]), "skin_n": str(m["missing"]["SkinThickness"]),
        "auc_lr": f"{comp['Logistic regression']['auc']:.3f}", "auc_rf": f"{comp['Random forest']['auc']:.3f}",
        "auc_gb": f"{comp['Gradient boosting']['auc']:.3f}", "auc_dt": f"{comp['Decision tree']['auc']:.3f}",
        "auc_glu": f"{comp['Glucose only (logistic)']['auc']:.3f}",
        "cal_slope": f"{cur['calibration']['slope']:.2f}", "spec_old": pct(old["specificity"]),
        "fp_m2": str(b10["straight"]["all"]["fp"]), "spec_old_m2": pct(b10["straight"]["age_45plus"]["specificity"]),
        "auc_old": f"{old['auc']:.2f}", "auc_young": f"{young['auc']:.2f}",
        "sens_young": pct(b10["curves"]["under_30"]["sensitivity"]),
        "cut_pct": pct(pub["cutoff"]),
        "auc_straight": f"{m['m3']['comparison']['Straight lines (Milestone 2)']['auc']:.2f}",
        "rule_tp": str(m["glucose_140_rule"]["tp"]), "model_tp_rule": str(cur["vs_glucose_rule"]["tp"]),
        "m4_pub": f"{m['m4']['table']['Curves (M3, as published)']['auc']:.3f}",
        "m4_ens": f"{m['m4']['table']['Stacked ensemble']['auc']:.3f}",
        "m4_low": f"{min(v['auc'] for k, v in m['m4']['table'].items() if k not in ('Curves, tuned', 'Curves (M3, as published)', 'Stacked ensemble', 'k-nearest neighbours')):.3f}",
        "sens_young_m2": pct(b10["straight"]["under_30"]["sensitivity"]),
    }


def per_100(counts) -> list:
    """Scale counts to 100 women, rounding so they add up to exactly 100 (largest remainder)."""
    total = sum(counts)
    raw = [c / total * 100 for c in counts]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[:100 - sum(out)]:
        out[i] += 1
    return out


def roadmap_rows() -> str:
    """Pull the summary table out of docs/MILESTONES.md so the page can't disagree with it."""
    rows = []
    for line in (ROOT / "docs" / "MILESTONES.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(M\d+)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|", line)
        if m:
            num, name, status, question = (s.strip() for s in m.groups())
            cls = "pill done" if status.startswith("Done") else "pill"
            rows.append(f'<tr><td><b>{num}</b> {html.escape(name)}</td><td>{html.escape(question)}</td>'
                        f'<td><span class="{cls}">{html.escape(status.split(" ")[0])}</span></td></tr>')
    return "\n".join(rows)


def build():
    out = site_dir()
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    model = json.loads((REPORTS / "model.json").read_text())
    oof = pd.read_csv(REPORTS / "oof.csv")

    if out.exists():
        shutil.rmtree(out)                       # the folder is ours; rebuild it from nothing
    (out / "notebooks").mkdir(parents=True)
    (out / "docs").mkdir()
    figures.save_all(out / "images")

    items = []
    for i, path in enumerate(sorted((ROOT / "notebooks").glob("*.ipynb")), 1):
        title, page = render_notebook(path)
        (out / "notebooks" / f"{path.stem}.html").write_text(page, encoding="utf-8")
        title = re.sub(r"^\d+\.\s*", "", title)
        items.append(f'  <li><a href="notebooks/{path.stem}.html"><span class="num">{i}</span>'
                     f'<span class="nbtitle">{html.escape(title)}</span></a>'
                     f'<span class="blurb">{html.escape(NOTEBOOK_BLURBS.get(path.stem, ""))}</span></li>')

    for name in DOCS:
        text = (ROOT / "docs" / f"{name}.md").read_text(encoding="utf-8")
        text = re.sub(r"\]\((\w+)\.md\)", r"](\1.html)", text)      # links between the docs
        title = {"OVERVIEW": "Start here", "ENHANCEMENTS_AND_BUGS": "Enhancements and bugs"}.get(
            name, name.replace("_", " ").title())
        (out / "docs" / f"{name}.html").write_text(shell(title, md(text)), encoding="utf-8")

    data = {
        "model": {k: model[k] for k in ["features", "median", "base_logit", "grid", "contribution", "range"]},
        "labels": LABELS,
        "cut": metrics["m5"]["published"]["cutoff"],
        "presets": {r: v["cutoff"] for r, v in metrics["m5"]["costs"]["ratios"].items()},
        "oof": {"p": [round(float(v), 6) for v in oof["p_current"]], "y": [int(v) for v in oof["outcome"]]},
        "check": model["check"],
    }
    page = (ROOT / "site_template" / "index.html").read_text(encoding="utf-8")
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    fill = tokens(metrics) | {"notebook_list": "\n".join(items), "roadmap_rows": roadmap_rows()}
    for key, value in fill.items():
        page = page.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{\w+\}\}", page)
    if left:
        raise SystemExit(f"Unfilled placeholders in template: {sorted(set(left))}")
    (out / "index.html").write_text(page, encoding="utf-8")

    # Every image the page refers to must exist.
    for src in re.findall(r'src="(images/[^"]+)"', page):
        if not (out / src).exists():
            raise SystemExit(f"index.html refers to {src}, which was not generated")

    # Portfolio rule: every generated page must carry the shared back bar, favicons and no-index tag.
    for f in out.rglob("*.html"):
        text = f.read_text(encoding="utf-8")
        for needle in ("sx-backbar", "/scott/backbar.css", "/scott/favicon.ico", 'content="noindex, nofollow"'):
            if needle not in text:
                raise SystemExit(f"{f.relative_to(out)} is missing {needle}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Site written to {out}  ({size / 1024:.0f} KB in {sum(1 for f in out.rglob('*') if f.is_file())} files)")


if __name__ == "__main__":
    build()
