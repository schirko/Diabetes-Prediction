"""Scaffold that writes the notebooks' structure and explanatory text.

    python tools/make_notebooks.py      rewrites every notebook WITHOUT outputs
    python -m src.run_notebooks         then fills the outputs back in

Each milestone adds its notebook here. Running this overwrites hand edits made
directly in the .ipynb files, so make text changes here.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import sys, warnings
sys.path.insert(0, "..")          # so the notebook can import the project's src/ folder
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")"""


# A two-or-three sentence summary for non-analysts, shown in a box at the top of each notebook.
PLAIN = {
    "01_data_audit.ipynb":
        "Before building anything, we checked the data. Five measurements use zero to mean \"not measured\", and "
        "insulin is missing for half the women. About 35% of the women developed diabetes, and every measurement is "
        "higher on average among them.",
    "02_honest_evaluation.ipynb":
        "The original 81% came from one lucky test. Tested fairly, 50 times, the same kind of model scores about 77% "
        "accuracy, and a simple model does as well as more complex ones.",
    "03_thresholds_and_calibration.ipynb":
        "A risk is only useful once you choose a cut-off for acting on it. Lowering the cut-off to 0.24 catches 85% of "
        "cases instead of 57%, at the cost of more false alarms. The risks themselves can be trusted, but the model "
        "works worst for women 45 and over. (Milestone 5, notebook 07, later replaces the 85% rule with a cut-off "
        "chosen from costs.)",
    "04_missing_data.ipynb":
        "Half the insulin readings are missing. No method of filling them in helps, and the guesses can change the "
        "screening decision for 53 women, so the model now leaves insulin and skin thickness out.",
    "10_external_validation.ipynb":
        "We tested the finished model, unchanged, on 961 women from a US national health survey. It still ranks women well "
        "(AUC 0.88, about the same as on the Pima women), but it is no better than glucose alone, and its risks are far too "
        "high: it predicts about 25% on average where about 4% had diabetes. The ranking travels; the risk numbers do not, "
        "and they would need rescaling for any new population.",
    "09_uncertainty.ipynb":
        "Every risk the page shows is an estimate from 768 women, so it could have come out differently. A 95% range around "
        "each risk shows how much (typically 19 points wide, widest in the middle of the scale), and a simulation shows the "
        "range is a little short of honest: it holds about 91% of the time. A second tool, conformal prediction, calls each "
        "woman \"no\", \"yes\" or \"too uncertain\", with a guarantee that holds on women it has never seen.",
    "08_explainability.ipynb":
        "Glucose, BMI and age are the only inputs the model clearly relies on, and a very different kind of model (boosted "
        "trees) ranks them the same way. Risk rises steadily with glucose, rises with age until about 50 and then falls, and "
        "rises with BMI until about 30 before levelling off. For a woman the model flags, a realistic change in glucose or "
        "BMI would clear her only about three times in ten, because most flagged women are well above the cut-off.",
    "07_decisions_and_costs.ipynb":
        "Rebalancing the data (class weights, SMOTE) does nothing a different cut-off can't, and it distorts the "
        "risks. What decides the cut-off is how bad a missed case is compared with an unnecessary test. At five times "
        "as bad the cut-off is 17%; it catches 33 of every 35 cases, with 30 false alarms per 100 women. The same "
        "cut-off works for every age group, which also fixes the drop in cases caught among women under 30.",
    "06_model_contest.ipynb":
        "Nine kinds of model were each given a fair chance: their settings were tuned automatically, on data kept "
        "apart from the test. None beat the simple curved model the page already uses, and combining several models "
        "didn't help either. The page keeps its model.",
    "05_curved_effects.ipynb":
        "The model over-flagged older women because it assumed risk keeps rising with age, when it actually peaks "
        "around 50. Letting age, glucose and BMI curve fixed much of this: 40 fewer false alarms overall, and twice "
        "as many healthy older women correctly cleared. It costs some missed cases among women under 30.",
}
HELP = ("New to the terms? The [glossary](../docs/GLOSSARY.md) explains every one, and the "
        "[overview](../docs/OVERVIEW.md) covers the whole project in five minutes.")


def build(name, title, intro, sections):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    plain = f"\n\n> **In plain terms.** {PLAIN[name]}" if name in PLAIN else ""
    cells = [nbf.v4.new_markdown_cell(f"# {title}{plain}\n\n{intro}\n\n{HELP}"), nbf.v4.new_code_cell(SETUP)]
    for s in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {s['h']}\n\n**What we're looking for.** {s['look']}"))
        cells.append(nbf.v4.new_code_cell(s["code"].strip()))
        tail = f"**How to read it.** {s['read']}\n\n**What it means.** {s['means']}"
        if s.get("next"):
            tail += f"\n\n**What's next and why.** {s['next']}"
        cells.append(nbf.v4.new_markdown_cell(tail))
    nb.cells = cells
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / name)


# ---------------------------------------------------------------- notebook 1
build(
    "01_data_audit.ipynb",
    "1. Data audit: what is really in this table?",
    "The data is the Pima Indians Diabetes dataset: 768 women aged 21 and over, of Pima heritage, examined near "
    "Phoenix, Arizona by the National Institute of Diabetes and Digestive and Kidney Diseases. Each row has eight "
    "measurements and whether the woman was diagnosed with diabetes within five years.\n\n"
    "Before any modeling, this notebook checks whether the numbers can be trusted as written. "
    "They can't, and that shapes everything after it.",
    [
        dict(
            h="1.1 First look",
            look="The size of the table and any values that are impossible for a living person.",
            code="""
from src.data import load_raw
raw = load_raw()
print(raw.shape)
raw.describe().T[["min", "50%", "max"]]""",
            read="Each row is one measurement. `min` is the smallest value recorded, `50%` is the middle value "
                 "(the median) and `max` is the largest.",
            means="Five measurements have a minimum of 0: glucose, blood pressure, skinfold thickness, insulin and BMI. "
                  "Nobody has a blood pressure or a BMI of zero. The file has no blank cells, so a zero was used to "
                  "mean \"not recorded\".",
            next="Count how many of these hidden gaps there are, because the answer decides how much we can lean "
                 "on each column.",
        ),
        dict(
            h="1.2 How much is missing",
            look="The number of impossible zeros in each column.",
            code="""
from src.data import missing_summary
from src import figures
display(missing_summary())
figures.missing_values();""",
            read="Each bar is one measurement. Its length is the share of the 768 women who have a zero where a "
                 "real reading should be.",
            means="Insulin is missing for 374 women (49%) and skinfold thickness for 227 (30%). The other three "
                  "columns are missing for 5% or fewer. The original college notebook filled these with a single "
                  "typical value worked out from the whole table. For insulin that means half the column was "
                  "invented, and the fill value itself was pulled down by the zeros it was meant to replace.",
            next="From here on, `load_data()` turns those zeros into proper missing values. Filling them in happens "
                 "later, inside the model pipeline, so it is learned from training rows only.",
        ),
        dict(
            h="1.3 Is a missing reading itself a clue?",
            look="Whether women with no insulin or skinfold reading were diagnosed more or less often. If they were, "
                 "the gap carries information and should not be papered over.",
            code="""
from src.data import load_data
X, y = load_data()
pd.DataFrame({
    "diabetes rate when recorded": [y[X[c].notna()].mean() for c in ["Insulin", "SkinThickness"]],
    "diabetes rate when missing":  [y[X[c].isna()].mean()  for c in ["Insulin", "SkinThickness"]],
}, index=["Insulin", "SkinThickness"])""",
            read="Each row compares two groups of women: those with a real reading and those without. "
                 "The numbers are the share diagnosed with diabetes (0.33 = 33%).",
            means="The rates are close: 33% against 37% for insulin, 33% against 39% for skinfold. "
                  "Missing readings lean slightly toward diabetes, but not enough to be a strong signal on their own.",
            next="Milestone 2 tests this properly by comparing several ways of handling the gaps (E08).",
        ),
        dict(
            h="1.4 How common is diabetes here?",
            look="The share of women diagnosed. That number is the score a model gets for doing nothing.",
            code="""
print(y.value_counts().rename({0: "no diabetes", 1: "diabetes"}))
print(f"\\nShare with diabetes: {y.mean():.1%}")
print(f"Accuracy from always answering 'no diabetes': {1 - y.mean():.1%}")""",
            read="The counts are women in each group. The last line is the accuracy of a \"model\" that never "
                 "predicts diabetes for anyone.",
            means="268 of 768 women (35%) were diagnosed. Always answering \"no\" is right 65% of the time while "
                  "catching nobody. So 65% is the floor, and accuracy alone is a poor yardstick: a model can look "
                  "decent while missing most of the people it exists to find.",
            next="Later notebooks report how many true cases are caught (sensitivity) alongside accuracy.",
        ),
        dict(
            h="1.5 Which measurements differ between the two groups?",
            look="A quick, honest read of direction: for each measurement, is the typical value higher or lower "
                 "among women who were diagnosed?",
            code="""
med = X.groupby(y).median().T
med.columns = ["no diabetes", "diabetes"]
med["difference"] = med["diabetes"] - med["no diabetes"]
med""",
            read="Each row is a measurement. The two columns are the median for each group, using real readings only.",
            means="Every measurement is higher in the diabetes group. Glucose shows the largest gap (140 against 107), "
                  "followed by insulin, age (36 against 27) and BMI. The original notebook said glucose, insulin, "
                  "family history and pregnancies were *lower* among women with diabetes. That was a misreading of a "
                  "crowded chart (B07).",
            next="Notebook 2 turns to modeling, starting with why the original 81% accuracy can't be taken at face value.",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 2
build(
    "02_honest_evaluation.ipynb",
    "2. Honest evaluation: how good is the model, really?",
    "The original notebook reported 81% accuracy. This notebook checks that number, replaces the evaluation "
    "with one that can be trusted, and compares six models on equal terms.",
    [
        dict(
            h="2.1 Reproduce the original result",
            look="The original recipe, run exactly as written, to confirm we start from the same place.",
            code="""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.data import load_raw

d = load_raw().astype(float)
d["Glucose"] = d["Glucose"].replace(0, d["Glucose"].mean())   # filled using the WHOLE table
d["BMI"] = d["BMI"].replace(0, d["BMI"].median())
X4, y_raw = d[["Pregnancies", "Glucose", "BMI", "Age"]], d["Outcome"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X4, y_raw, test_size=0.2, random_state=0)
original = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(f"Accuracy on the test set: {original.score(X_te, y_te):.1%}  ({len(y_te)} women)")""",
            read="One model, trained on 80% of the women and scored on the other 20% (154 women).",
            means="The 81% reproduces. But it rests on one particular random split, and 154 women is a small test "
                  "set. One or two lucky rows move the score by a point.",
            next="Repeat the same recipe with different random splits to see how much the score depends on luck.",
        ),
        dict(
            h="2.2 The same recipe on 200 different splits",
            look="The spread of scores when nothing changes except which women land in the test set.",
            code="""
from src.evaluate import lucky_split_demo
from src import figures
ls = lucky_split_demo()
print(f"Original split: {ls['original_seed_accuracy']:.1%}")
print(f"Average of 200: {ls['mean']:.1%}   lowest {ls['min']:.1%}   highest {ls['max']:.1%}")
print(f"Splits that scored below the original: {ls['share_below_original']:.0%}")
figures.lucky_split();""",
            read="Each bar counts how many of the 200 splits landed at that accuracy. The black line is the average. "
                 "The orange line is the split the original notebook used.",
            means="The same method scores anywhere from 68% to 84%, and averages 77%. The original split beat 88% "
                  "of the others. The 81% was a fortunate draw. The honest figure for that recipe is about 77%, "
                  "against a do-nothing floor of 65%.",
            next="Replace the single split with repeated cross-validation, which averages over many splits by design, "
                 "and fix the leak in how the gaps were filled.",
        ),
        dict(
            h="2.3 A leak-free pipeline, scored by repeated cross-validation",
            look="A trustworthy score for logistic regression using all eight measurements.",
            code="""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data import load_data

X, y = load_data()                      # zeros are now proper missing values

pipe = make_pipeline(
    SimpleImputer(strategy="median"),   # fill gaps with the TRAINING fold's median
    StandardScaler(),                   # put every measurement on the same scale
    LogisticRegression(max_iter=1000),
)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)   # 50 train/test splits
res = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "accuracy", "recall"])
for k in ["roc_auc", "accuracy", "recall"]:
    scores = res[f"test_{k}"]
    print(f"{k:9s} mean {scores.mean():.3f}   spread (sd) {scores.std():.3f}")""",
            read="Cross-validation cuts the data into 5 parts, trains on 4 and tests on the 5th, and rotates so every "
                 "woman is tested once. Doing that 10 times with different cuts gives 50 scores. `mean` is their "
                 "average and `sd` is how much they wobble. `roc_auc` (AUC) is the chance the model gives a higher "
                 "risk to a woman with diabetes than to one without: 0.5 is a coin flip, 1.0 is perfect. `recall` is "
                 "the share of true cases caught, also called sensitivity.",
            means="AUC is 0.835, accuracy 77%, and recall 56%. The pipeline matters because the fill-in values and "
                  "the rescaling are re-learned inside each training fold, so nothing about the test women leaks in. "
                  "The recall is the uncomfortable number: at the standard setting the model misses 44% of the women "
                  "who have diabetes.",
            next="See whether a more complex model does better, scored on exactly the same 50 splits.",
        ),
        dict(
            h="2.4 Six models on equal terms",
            look="Whether tree-based models beat plain logistic regression, and how far any of them is from two "
                 "simple yardsticks: doing nothing, and using glucose alone.",
            code="""
from src.evaluate import compare_models
from src import figures
table = compare_models(X, y)
display(table[["auc", "auc_sd", "avg_precision", "brier", "accuracy", "sensitivity"]].sort_values("auc", ascending=False))
figures.model_comparison();""",
            read="Each row is a model. `auc` is explained above and `auc_sd` is its wobble across the 50 splits. "
                 "`brier` measures how far the predicted probabilities are from what happened (lower is better). "
                 "In the chart, each dot is the average AUC and the whisker is one standard deviation either side.",
            means="Random forest (0.838) and logistic regression (0.835) are tied: the gap is a tenth of the wobble "
                  "between splits. Gradient boosting (0.823) and a single decision tree (0.807) are behind. Glucose "
                  "alone reaches 0.791, so the other seven measurements together add about 0.04. With 768 rows and "
                  "eight inputs there isn't enough data for a complex model to find patterns a straight-line model "
                  "misses.",
            next="Logistic regression goes forward for Milestone 1: it ties for best, its probabilities are easy to "
                 "explain, and it can run inside a web page. Milestone 4 gives the complex models a fair second "
                 "chance with tuning and nested cross-validation (E12).",
        ),
        dict(
            h="2.5 Did dropping four columns matter?",
            look="The original notebook dropped blood pressure, skinfold, insulin and family history because their "
                 "correlation with the outcome looked low. Does a four-input model lose anything?",
            code="""
four = ["Pregnancies", "Glucose", "BMI", "Age"]
res4 = cross_validate(pipe, X[four], y, cv=cv, scoring=["roc_auc"])
print(f"All eight inputs: AUC {res['test_roc_auc'].mean():.3f}")
print(f"Four inputs:      AUC {res4['test_roc_auc'].mean():.3f}")""",
            read="Two versions of the same model, scored on the same 50 splits.",
            means="0.835 against 0.833. The four dropped columns add almost nothing in a straight-line model, so the "
                  "original instinct was reasonable, even though picking columns by looking at the whole table is "
                  "itself a small leak. All eight stay in for now so the app can show each one's contribution.",
            next="Notebook 3 takes the chosen model and asks the practical questions: where to set the cut-off, "
                 "whether the probabilities can be believed, and who it works less well for.",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 3
build(
    "03_thresholds_and_calibration.ipynb",
    "3. From a score to a decision",
    "A model outputs a probability. Someone still has to decide what probability is high enough to act on. "
    "This notebook picks that cut-off for a screening setting, checks that the probabilities are believable, "
    "and exports the final model for the web page.",
    [
        dict(
            h="3.1 One honest prediction per woman",
            look="A predicted probability for every woman, made by a model that never saw her during training.",
            code="""
from sklearn.metrics import roc_auc_score
from src.data import load_data
from src.evaluate import out_of_fold, bootstrap_auc, confusion_at
from src.modeling import logistic

X, y = load_data()
p = out_of_fold(logistic(), X, y)       # average of each woman's 10 held-out predictions
lo, hi = bootstrap_auc(y, p)
print(f"AUC {roc_auc_score(y, p):.3f}   95% interval {lo:.3f} to {hi:.3f}")
pd.Series(confusion_at(y, p, 0.50))""",
            read="`tp` are women with diabetes who were flagged, `fn` are women with diabetes who were missed, `fp` are "
                 "healthy women flagged by mistake, and `tn` are healthy women correctly cleared. Sensitivity is "
                 "tp / (tp + fn). Specificity is tn / (tn + fp). `ppv` is the share of flagged women who do have diabetes.",
            means="The AUC is 0.836, and the interval says the true value is plausibly anywhere from 0.81 to 0.86. "
                  "At the default cut-off of 0.50 the model catches 152 of 268 cases (57%) and misses 116.",
            next="For screening, a missed case costs more than a false alarm (a false alarm leads to a follow-up "
                 "blood test). So look at what a lower cut-off buys.",
        ),
        dict(
            h="3.2 Choosing the cut-off",
            look="The cut-off that catches at least 85% of true cases, and what it costs in false alarms.",
            code="""
from src.evaluate import pick_threshold, TARGET_SENSITIVITY
from src import figures
t = pick_threshold(y, p)
print(f"Highest cut-off that still catches {TARGET_SENSITIVITY:.0%} of cases: {t:.2f}")
display(pd.DataFrame([confusion_at(y, p, 0.50), confusion_at(y, p, t)], index=["default 0.50", f"screening {t:.2f}"]).T)
figures.threshold_tradeoff();""",
            read="The green line is the share of true cases caught at each cut-off. The orange line is the share of "
                 "healthy women correctly cleared. Moving left catches more cases and raises more false alarms.",
            means="Lowering the cut-off from 0.50 to 0.24 raises cases caught from 57% to 85% (229 of 268) and cuts "
                  "missed cases from 116 to 39. The price is 188 false alarms instead of 61, and 54% of all women "
                  "get flagged. Of those flagged, 55% do have diabetes. Whether that trade is worth it depends on "
                  "what a follow-up test costs. The 85% target is a judgment call, not something the data decides.",
            next="Check the model against the simplest rule a clinic could use instead.",
        ),
        dict(
            h="3.3 Does it beat a one-number rule?",
            look="A comparison with \"flag anyone whose 2-hour glucose is 140 mg/dL or more\", the standard line "
                 "for impaired glucose tolerance.",
            code="""
rule = confusion_at(y, X["Glucose"].fillna(0).to_numpy(), 140)
model_same_alarms = confusion_at(y, p, 0.50)
pd.DataFrame([rule, model_same_alarms], index=["glucose >= 140", "model at 0.50"])[
    ["tp", "fn", "fp", "tn", "sensitivity", "specificity"]]""",
            read="Two ways of flagging women, chosen because they raise almost the same number of false alarms "
                 "(62 and 61), so the cases caught can be compared directly.",
            means="For the same number of false alarms, the model catches 152 cases and the glucose rule catches 135. "
                  "That is 17 more women found, a real but modest gain. Glucose does most of the work, which matches "
                  "notebook 2.",
            next="A cut-off only makes sense if the probabilities themselves can be believed, so check calibration.",
        ),
        dict(
            h="3.4 Can the probabilities be believed?",
            look="Whether women given a 30% risk turn out to have diabetes about 30% of the time.",
            code="""
from src.evaluate import calibration
cal = calibration(y, p)
print(f"Calibration slope {cal['slope']:.2f} (ideal 1.00), intercept {cal['intercept']:.2f} (ideal 0.00), Brier {cal['brier']:.3f}")
figures.calibration_plot();""",
            read="The women are sorted by predicted risk and cut into 8 equal groups. Each dot plots a group's "
                 "average predicted risk against the share who really had diabetes. Dots on the diagonal mean "
                 "the probabilities are accurate.",
            means="The dots stay close to the diagonal and the slope is 0.96, so the probabilities are well "
                  "calibrated for this group of women. A predicted 30% does mean roughly 3 in 10. That holds only "
                  "for a population like this one, where 35% have diabetes.",
            next="An average can hide groups the model serves badly, so split the results by age.",
        ),
        dict(
            h="3.5 Who does it work less well for?",
            look="The same cut-off applied to three age bands.",
            code="""
from src.evaluate import by_age
pd.DataFrame(by_age(X, y, p, t)).set_index("band")""",
            read="Each row is an age band. `prevalence` is the share with diabetes in that band. Sensitivity and "
                 "specificity use the screening cut-off from 3.2.",
            means="Diabetes is far less common under 30 (21%) than over 30 (about 49%). With one cut-off for "
                  "everyone, the model catches 80% of cases in the youngest band and 91% in the oldest, but it clears "
                  "only 22% of healthy women aged 45 and over. For older women it flags nearly everyone, which makes "
                  "it close to useless as a filter there. The AUC also falls with age, from 0.86 to 0.73.",
            next="This is the main weakness to work on. Milestone 3 tests whether letting the effect of age and "
                 "glucose bend (rather than follow a straight line) fixes it (E10).",
        ),
        dict(
            h="3.6 The final model",
            look="What the model leans on, and the export the web page uses.",
            code="""
import json
from src.evaluate import export_model
from src.data import LABELS
model = export_model(X, y)              # fitted on all 768 women
display(pd.DataFrame({
    "odds multiplier per 1 SD": model["odds_ratio_per_sd"],
    "1 SD is": model["scale"],
    "unit": [LABELS[f][1] for f in model["features"]],
}, index=model["features"]).sort_values("odds multiplier per 1 SD", ascending=False))
figures.odds_ratios(model);""",
            read="Each row is an input. The multiplier says how much the odds of diabetes change when that input "
                 "rises by one standard deviation (a typical step), with the others held fixed. Above 1 raises "
                 "the odds, below 1 lowers them, and 1 means no effect.",
            means="Glucose dominates: a 30 mg/dL rise roughly triples the odds. BMI (×1.9 per 7 points) and "
                  "pregnancies (×1.5 per 3) follow. Insulin and blood pressure come out slightly below 1, which "
                  "looks odd but is common when inputs overlap: once glucose is known, insulin adds little, and half "
                  "of it was filled in anyway. These are associations in this group of women, not causes.",
            next="This eight-input model is the Milestone 1 baseline. Notebook 4 (Milestone 2) takes on the biggest "
                 "loose end, the 49% of insulin values that are missing, and ends by changing which inputs the "
                 "published model uses.",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 4 (Milestone 2)
build(
    "04_missing_data.ipynb",
    "4. Missing data: does it matter how the gaps are filled?",
    "Insulin is missing for 49% of the women and skinfold thickness for 30%. Milestone 1 filled every gap with "
    "the column's median, the simplest option. This notebook tests whether a smarter method does better (E08), "
    "then measures how much one woman's estimate depends on what was guessed for her (E09). "
    "It ends with a decision about the published model.",
    [
        dict(
            h="4.1 Seven strategies, the same 50 splits",
            look="Whether any way of handling the gaps scores better than the median fill from Milestone 1.",
            code="""
from src.data import load_data
from src.missing import compare_strategies, strategies
from src import figures

X, y = load_data()
for name in strategies():
    print("-", name)
table = compare_strategies(X, y)
display(table[["auc", "auc_sd", "brier"]].sort_values("auc", ascending=False))
figures.strategy_comparison();""",
            read="Each row is one way of dealing with the gaps. The first six all end in the same logistic "
                 "regression, so only the gap handling differs. \"Flags\" adds a yes/no column saying a reading was "
                 "missing. \"Nearest neighbours\" copies from the 10 most similar women. \"Iterative\" predicts each "
                 "missing value from the woman's other measurements. `auc` is the average over 50 splits and `auc_sd` "
                 "is how much it wobbles from split to split.",
            means="The six logistic versions land between 0.832 and 0.838. The wobble between splits is 0.027, about "
                  "five times the whole range. Gradient boosting, which accepts gaps without any filling, is last at "
                  "0.823. Nothing clever beats the median, and leaving insulin and skinfold out entirely scores as "
                  "well as anything (0.838).",
            next="Averages can hide a small but consistent edge, so compare each strategy with the baseline one "
                 "split at a time.",
        ),
        dict(
            h="4.2 Split by split against the baseline",
            look="Whether any strategy beats the median fill consistently, even by a little.",
            code="""
paired = table[["auc_vs_baseline", "splits_better_than_baseline"]].drop(index="Median fill (Milestone 1)")
paired.sort_values("auc_vs_baseline", ascending=False)""",
            read="`auc_vs_baseline` is the average AUC gain over the median fill, measured on identical splits. "
                 "`splits_better_than_baseline` is the share of the 50 splits where the strategy won (0.5 is a coin flip).",
            means="Dropping the two columns wins 76% of the splits, but by an average of 0.002, which nobody would "
                  "notice in use. Adding \"was missing\" flags to the median fill is a coin flip (52%) and slightly "
                  "worse on average. So the gaps carry no useful signal, which matches the weak link seen in "
                  "notebook 1.",
            next="The model results say filling doesn't help. Check directly why: how well can a missing reading "
                 "be guessed at all?",
        ),
        dict(
            h="4.3 Can a missing reading be guessed back?",
            look="The accuracy of each fill method, tested the only way possible: hide readings we do have, "
                 "ask the method to guess them, and compare with the truth.",
            code="""
from src.missing import recovery_test
for col in ["Insulin", "SkinThickness"]:
    print(col)
    display(recovery_test(X, col))
figures.recovery();""",
            read="A quarter of the known readings are hidden, 20 times over. `r2` is the share of the real variation "
                 "the guesses recover: 0 means no better than guessing the median for everyone, 1 would be perfect. "
                 "`typical_miss` is the average distance between guess and truth, in the measurement's own units.",
            means="The best method recovers 29% of the variation in insulin and misses by 65 mu U/mL on average, "
                  "against 80 for the plain median. Skinfold does a little better (40%, missing by 6 mm against 8). "
                  "Insulin can't be reconstructed from the other measurements with any accuracy. Whatever goes into "
                  "those 374 gaps is mostly noise, which explains why no fill method improved the model.",
            next="If the fills are mostly noise, a woman's estimate partly depends on the luck of the fill. "
                 "Measure how much.",
        ),
        dict(
            h="4.4 One woman, 30 plausible fills",
            look="How far each woman's estimate moves when the gaps are filled 30 different plausible ways "
                 "instead of one \"best\" way. This is called multiple imputation.",
            code="""
from src.missing import multiple_imputation, spread_summary
from src.evaluate import pick_threshold, out_of_fold
from src.modeling import logistic

draws = multiple_imputation(X, y)            # 768 women x 30 fills, every prediction held out
cut_m1 = pick_threshold(y, out_of_fold(logistic(), X, y))
spread = spread_summary(X, y, draws, cut_m1)
display(pd.DataFrame(spread["by_group"]).T[["women", "median_width", "p90_width"]])
figures.imputation_spread();""",
            read="For each woman, the width is the gap between her 5th and 95th percentile estimates across the 30 "
                 "fills, in percentage points. `median_width` is the typical woman in the group and `p90_width` is "
                 "one of the widest 10%. The chart counts women at each width; the last bar collects 20 and over.",
            means="A woman with no insulin reading typically moves 5.3 points across the fills, and one in ten "
                  "moves about 10. Women with every reading still move 3.2 points, because the model itself was "
                  "trained on filled-in data and shifts with each fill. So a single estimate such as \"31%\" hides "
                  "a spread of several points that comes purely from guesswork.",
            next="A few points only matters if it changes the decision, so count the women who cross the cut-off.",
        ),
        dict(
            h="4.5 Who lands on both sides of the cut-off?",
            look="Women who are flagged under some fills and cleared under others.",
            code="""
print(f"Cut-off {spread['threshold']:.2f}, {spread['draws']} fills")
display(pd.DataFrame(spread["by_group"]).T[["women", "flips", "flip_share"]])
print(f"Women whose decision depends on the fill: {spread['flips_total']} of {len(y)}")""",
            read="A woman counts as a flip if she is flagged in more than 10% but fewer than 90% of the fills. "
                 "`flip_share` is the share of the group that flips.",
            means="53 of 768 women (7%) get a different screening decision depending on which plausible fill is used: "
                  "8% of those with insulin missing and 6% of those with every reading. For those women the outcome "
                  "is set by the fill, not by anything measured.",
            next="Put the evidence together and decide what the published model should do.",
        ),
        dict(
            h="4.6 Decision: leave insulin and skinfold out",
            look="The results for a six-input model with insulin and skinfold removed, set against the "
                 "eight-input baseline from notebook 3.",
            code="""
from src.evaluate import summarise
from src.modeling import current_features, LEFT_OUT
from src.data import FEATURES

cols = current_features(FEATURES)
print("Left out:", LEFT_OUT)
print("Women who still have a gap in the six remaining inputs:", int(X[cols].isna().any(axis=1).sum()))

def row(s):
    t = s["threshold_screening"]
    return {"inputs": len(s["features"]), "AUC": s["oof_auc"], "cut-off": t["threshold"],
            "cases caught": t["tp"], "missed": t["fn"], "false alarms": t["fp"],
            "calibration slope": s["calibration"]["slope"],
            "cleared, age 45+": s["by_age"][2]["specificity"]}

base = summarise(X, y, out_of_fold(logistic(), X, y))
six = summarise(X[cols], y, out_of_fold(logistic(), X[cols], y))
pd.DataFrame([row(base), row(six)], index=["Milestone 1 (eight inputs)", "Milestone 2 (six inputs)"]).T""",
            read="Two models side by side, each scored on predictions for women it had not seen. The cut-off is "
                 "re-chosen for each so that at least 85% of true cases are caught.",
            means="The six-input model matches the baseline on every line: AUC 0.838 against 0.836, 231 cases caught "
                  "against 229, and 194 false alarms against 188. The gaps fall from 374 women to 44, so almost "
                  "nobody's estimate rests on a filled-in value any more. The decision (D09) is to publish the "
                  "six-input model: it is as accurate, it stops inventing half a column, and it asks for two fewer "
                  "measurements. What it does not fix is the weak result for women 45 and over, where only 22% of "
                  "healthy women are cleared (B10).",
            next="Milestone 3 lets effects bend and interact. Insulin gets one more hearing there, in the form "
                 "where it should matter most: combined with glucose as a rough measure of insulin resistance (E29). "
                 "The same milestone goes after B10.",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 5 (Milestone 3)
build(
    "05_curved_effects.ipynb",
    "5. Curved effects: why older women were over-flagged",
    "Every model so far assumed each input pushes risk along a straight line: each extra year of age adds the "
    "same amount. Milestone 2 left one open bug, B10: with one cut-off for everyone, the model clears only 22% "
    "of healthy women aged 45 and over. This notebook finds out why, tests nine ways of letting effects bend or "
    "interact (E10, E11), gives insulin one more hearing (E29), and ends by changing the published model.",
    [
        dict(
            h="5.1 What the straight line gets wrong",
            look="Whether the model's predicted risk matches the real diabetes rate at every age, not just on average.",
            code="""
from sklearn.metrics import roc_auc_score
from src.data import load_data, FEATURES
from src.modeling import current_features
from src.evaluate import out_of_fold, confusion_at, pick_threshold
from src import nonlinear as nl
from src import figures

X, y = load_data()
Xc = X[current_features(FEATURES)]                    # the six inputs from Milestone 2
p_straight = out_of_fold(nl.lr_pipeline(), Xc, y)     # the Milestone 2 model, held-out predictions
nl.calibration_by_age(X, y, p_straight)""",
            read="Each row is a five-year age group. `actual` is the share of those women diagnosed with diabetes; "
                 "`predicted` is the model's average risk for them, from predictions made without seeing them.",
            means="The real rate climbs from 14% at 21 to 24, to 71% at 50 to 54, then falls to 41% and 28%. "
                  "A straight line can only keep climbing, so the model says about 50% for women 60 and over, "
                  "who are really at 28%. It is also too high for the youngest group (20% against 14%) and too low "
                  "in the thirties. The fall after 55 may partly be survivorship (women who would develop diabetes "
                  "often already have it), and only 54 women are over 55, so it is a real pattern but a noisy one.",
            next="Before fixing the shape, check how much of B10 the shape explains, and how much comes from "
                 "something else.",
        ),
        dict(
            h="5.2 B10 is two problems, not one",
            look="For each age band: how common diabetes is, how well the model separates cases inside the band, "
                 "and how many healthy women are cleared, first with the shared cut-off, then with a cut-off "
                 "chosen for that band alone.",
            code="""
shared_cut = pick_threshold(y, p_straight)
rows = []
for band, m in nl.age_band_masks(X).items():
    rows.append({
        "band": band, "women": int(m.sum()), "have diabetes": y[m].mean(),
        "AUC inside band": roc_auc_score(y[m], p_straight[m]),
        "cleared, shared cut-off": confusion_at(y[m].to_numpy(), p_straight[m], shared_cut)["specificity"],
        "cleared, band's own cut-off": nl.band_specificity(y, p_straight, m),
    })
print(f"Shared cut-off: {shared_cut:.2f}")
pd.DataFrame(rows).set_index("band")""",
            read="\"Cleared\" means healthy women correctly not flagged (specificity). The band's own cut-off is "
                 "set so it still catches 85% of that band's cases, so the last column shows how well the model "
                 "sorts women within the band, free of the base rate.",
            means="Half of women 45 and over have diabetes, against 21% under 30. A model that is right about that "
                  "will give older women high risks, and a shared cut-off then flags most of them. That part of B10 "
                  "is the base rate, not a fault. But even with its own cut-off the band clears only 39%, because "
                  "the model separates older women less well (AUC 0.74 against 0.86 under 30). The wrong age shape "
                  "from 5.1 is one likely cause of that weaker separation, and that part can be fixed.",
            next="Try models that let the effects bend, and see whether separation among older women improves.",
        ),
        dict(
            h="5.3 Nine models on the same 50 splits",
            look="Whether curves, interactions or boosted models beat the straight-line model overall and, "
                 "above all, for women 45 and over.",
            code="""
# Takes a few minutes: two of the models are Explainable Boosting Machines, fitted 50 times each.
for name in nl.candidates(Xc.columns):
    print("-", name)
table, probs = nl.compare(Xc, y)
display(table[["auc", "auc_sd", "brier", "auc_45plus", "spec_45plus"]].sort_values("auc", ascending=False))
figures.m3_models();""",
            read="Every model uses the same six inputs and the same 50 train/test splits. `auc` is overall; "
                 "`auc_45plus` is the AUC among women 45 and over only; `spec_45plus` is the share of healthy older "
                 "women cleared when that band's own cut-off catches 85% of its cases. \"Curves\" are splines: lines "
                 "allowed to change slope at a few points. Extreme values are held at the 1st and 99th percentiles "
                 "first, so a curve can't swing wildly where there are only a handful of women. The Explainable Boosting Machine (EBM) learns a curve per "
                 "input by itself. The last model is gradient boosting forced so risk can only rise with glucose, "
                 "BMI and family history.",
            means="Curves for glucose, BMI and age come out on top: AUC 0.853 against 0.838, and among women 45 and "
                  "over 0.791 against 0.744, with 64% of healthy older women cleared instead of 39%. A curve for age "
                  "alone gets most of the gain for older women (0.786). Adding the age-by-glucose interaction on top "
                  "of the curves changes nothing, and log scales add nothing. The EBM finds similar shapes but scores "
                  "lower (0.841), and the constrained boosting model is the weakest (0.832). With 768 women, a few "
                  "well-chosen curves beat a model that has to discover every shape itself.",
            next="Nine candidates were tried and the best was picked, which flatters the winner slightly. Check "
                 "the result split by split, then on a completely fresh set of splits.",
        ),
        dict(
            h="5.4 Does the gain survive a second look?",
            look="Whether the winner beats the straight line consistently, and whether the ranking holds on 50 "
                 "new splits the models were never compared on.",
            code="""
display(table[["auc_vs_baseline", "wins"]].sort_values("auc_vs_baseline", ascending=False))

from sklearn.model_selection import RepeatedStratifiedKFold
fresh = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2026)
check, _ = nl.compare(Xc, y, names=[nl.BASELINE, "Curve for age only", nl.CURVES], cv=fresh)
check[["auc", "auc_vs_baseline", "wins", "auc_45plus", "spec_45plus"]]""",
            read="`auc_vs_baseline` is the average gain over the straight-line model on identical splits; `wins` is "
                 "the share of the 50 splits where the model scored higher. The second table repeats the top three "
                 "on a different random set of splits (seed 2026 instead of 42).",
            means="The curves model beats the straight line on 86% of splits, by 0.016 on average. On the fresh "
                  "splits the order is the same and the gain is 0.014, winning 78% of the time. The gain is small but "
                  "consistent, and it is concentrated where B10 lives: the 45-and-over AUC rises from 0.745 to 0.791 "
                  "on the fresh splits too.",
            next="Look at the shapes the curves model learned, to check they make sense.",
        ),
        dict(
            h="5.5 What the curves look like",
            look="Whether each learned curve is believable, not just whether it scores well.",
            code="""
figures.shapes();""",
            read="Each panel shows how one input changes the odds of diabetes compared with a typical woman (who has "
                 "the middle value of every measurement). Orange is the straight-line model, green the curved one. "
                 "The vertical scale is logarithmic, so ×2 and ×0.5 are the same distance from ×1.",
            means="Glucose stays close to the straight line over the range where most women are (80 to 180 mg/dL), "
                  "so bending it changes little. BMI climbs steeply up to about 30, is roughly flat from 30 to about "
                  "42, then climbs again; the flat stretch may be noise and is worth revisiting in Milestone 6. Age is "
                  "the real change: odds rise to a peak near 50 and then turn down, where the straight line keeps "
                  "rising. That matches the rates in 5.1, so the curve is following the data. The flat ends are the "
                  "clipping at work, beyond the 1st and 99th percentiles.",
            next="Measure what the new shape does to B10 and to the screening numbers as a whole.",
        ),
        dict(
            h="5.6 B10 before and after",
            look="The screening results with one shared cut-off, the setting B10 was found in, for both models. "
                 "Each cut-off is re-chosen so that 85% of all cases are caught.",
            code="""
p_curves = probs[nl.CURVES]
older = (X["Age"] >= 45).to_numpy()
rows = {}
for name, p in [("Straight lines (Milestone 2)", p_straight), ("Curves (Milestone 3)", p_curves)]:
    cut = pick_threshold(y, p)
    every, old = confusion_at(y, p, cut), confusion_at(y[older].to_numpy(), p[older], cut)
    rows[name] = {"cut-off": cut, "cases caught": every["tp"], "missed": every["fn"],
                  "false alarms": every["fp"], "healthy women 45+ cleared": old["specificity"]}
display(pd.DataFrame(rows))
lo, hi = nl.bootstrap_band_gain(y, p_curves, p_straight, older)
print(f"Gain in older women cleared (own cut-off), 95% interval: {lo:+.2f} to {hi:+.2f}")
figures.age_fit();""",
            read="Both models are scored on held-out predictions. The interval comes from resampling the 133 older "
                 "women 1,000 times. The chart compares the real rate in each age group (dots) with each model's "
                 "average prediction (lines).",
            means="At the same 85% catch rate, false alarms fall from 194 to 154, and the share of healthy women "
                  "45 and over who are cleared nearly doubles, from 22% to 43%. The price is 2 more missed cases "
                  "overall (39 against 37). The interval is wide, from slightly worse to much better, because only "
                  "67 healthy older women are in the data; the direction is clear, the size less so. The chart shows "
                  "why: the green line follows the dots down after 55, the orange line does not.",
            next="Check who pays for the gain. A higher cut-off and lower risks for young women could cost "
                 "something in the youngest band.",
        ),
        dict(
            h="5.7 Who pays for it?",
            look="Cases caught and missed in each age band, for both models at their shared cut-offs.",
            code="""
rows = []
for name, p in [("straight", p_straight), ("curves", p_curves)]:
    cut = pick_threshold(y, p)
    for band, m in nl.age_band_masks(X).items():
        c = confusion_at(y[m].to_numpy(), p[m], cut)
        rows.append({"model": name, "band": band, "caught": c["tp"], "missed": c["fn"],
                     "false alarms": c["fp"], "sensitivity": c["sensitivity"]})
pd.DataFrame(rows).pivot(index="band", columns="model", values=["caught", "missed", "false alarms", "sensitivity"])""",
            read="The same two models and cut-offs as 5.6, broken down by age band. Sensitivity is the share of "
                 "each band's cases that are caught.",
            means="The gain is not free for everyone. Under 30, the curved model catches 60 of 84 cases against 67, "
                  "so sensitivity falls from 80% to 71%, while 23 fewer young women get false alarms. In the 30s and "
                  "early 40s it catches 6 more. This happens because the curved model gives young women lower, more "
                  "accurate risks, and one shared cut-off then lets more young cases through. It is a cut-off "
                  "problem, not a ranking problem: the AUC inside the youngest band is unchanged (0.85). Logged as "
                  "B12.",
            next="One cut-off for everyone is a policy choice, and Milestone 5 is where cut-offs are set from costs. "
                 "Band-specific cut-offs are added there as E31. First, insulin's second hearing.",
        ),
        dict(
            h="5.8 Insulin's second hearing (E29)",
            look="Whether insulin adds anything in the forms where it should matter most, tested only on the 394 "
                 "women who have a real reading.",
            code="""
nl.insulin_retest(X, y)""",
            read="Four versions of the straight-line model, all on the same 50 splits of those 394 women: the six "
                 "current inputs, then the same plus raw insulin, log insulin, or the log of glucose times insulin, "
                 "a rough measure of insulin resistance. `wins` is the share of splits where the version beat the "
                 "six-input model.",
            means="None helps. All three lose 0.003 of AUC on average and win only about a quarter of the splits. "
                  "Even where insulin was measured, it carries nothing that glucose and BMI don't already provide. "
                  "Decision D09, leaving it out, stands.",
            next="Put it together and decide what the page should run.",
        ),
        dict(
            h="5.9 Decision: publish the curves model",
            look="The full set of results for the curved model next to the Milestone 2 model, and its export "
                 "for the web page.",
            code="""
from src.evaluate import summarise

def row(s):
    t, d = s["threshold_screening"], s["threshold_default"]
    return {"AUC": s["oof_auc"], "cut-off": t["threshold"], "cases caught": t["tp"], "missed": t["fn"],
            "false alarms": t["fp"], "caught at 0.50": d["sensitivity"],
            "calibration slope": s["calibration"]["slope"], "cleared, age 45+": s["by_age"][2]["specificity"]}

before, after = summarise(Xc, y, p_straight), summarise(Xc, y, out_of_fold(nl.curves_model(), Xc, y))
display(pd.DataFrame([row(before), row(after)], index=["Milestone 2 (straight)", "Milestone 3 (curves)"]).T)

export = nl.export_curves_model(Xc, y)
check = nl.predict_from_export(export, export["check"]["inputs"])
print("Web page arithmetic vs scikit-learn, largest difference:", abs(check - export["check"]["probabilities"]).max())""",
            read="Each column is one model, scored on predictions for women it had not seen. The last line checks "
                 "the web page's version of the model: it stores each input's curve as a lookup table and adds up "
                 "the contributions, and that sum must match scikit-learn.",
            means="The curved model is better on almost every line: AUC 0.854 against 0.838, 40 fewer false "
                  "alarms for 2 more missed cases, and calibration still close to ideal (slope 0.94). Decision D11: "
                  "the page now runs it. B10 is closed as improved: 43% of healthy older women are now cleared "
                  "instead of 22%, and the "
                  "rest is mostly the base rate and a data limit. The trade-off for young women is open as B12. "
                  "Because the model is a sum of one curve per input, the page can still show exactly what each "
                  "measurement contributes.",
            next="Milestone 4 gives every model type a fair, tuned contest with nested cross-validation, now with "
                 "the curves model as the one to beat.",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 6 (Milestone 4)
build(
    "06_model_contest.ipynb",
    "6. A fair contest: does any tuned model beat the curves?",
    "Milestones 1 to 3 compared models using mostly default settings, which can be unfair to models that need "
    "tuning, such as support vector machines, boosted trees and neural networks. This notebook gives nine model "
    "types a properly tuned contest (E12), tests the differences formally (E13), tries combining models into one "
    "(E14), and decides whether the published curves model stays.",
    [
        dict(
            h="6.1 How the contest stays fair",
            look="A set-up where tuning can't flatter any model, and every model faces exactly the same test.",
            code="""
from src.data import load_data, FEATURES
from src.modeling import current_features
from src import contest
from src import figures

X, y = load_data()
Xc = X[current_features(FEATURES)]              # the same six inputs for every model
print("Outer splits (scoring):", contest.OUTER_FOLDS, "folds x", contest.OUTER_REPEATS, "repeats")
print("Inner splits (tuning): ", contest.INNER_FOLDS, "folds, inside each outer training set")
print("Tuning trials per model per outer split:", contest.N_TRIALS)
for name, (_, space) in contest.contestants().items():
    print(f"- {name:28s} {'tuned' if space else 'fixed settings'}")""",
            read="Most models have settings (hyperparameters) that must be chosen, such as how many neighbours to "
                 "consult or how deep a tree may grow. Tuning means trying many settings and keeping the best. The "
                 "danger is choosing settings that suit the test women by luck. Nested cross-validation prevents it: "
                 "the outer splits score each model, and inside each outer training set a second, inner split is "
                 "used only for tuning. Optuna does the tuning, trying 30 combinations per model per outer split and "
                 "learning from earlier tries which areas look promising.",
            means="Ten contestants face the same 15 outer splits. Two are not tuned: the published curves model, "
                  "exactly as on the page, and a stacked ensemble whose members use settings fixed in advance. The "
                  "other eight are tuned, including the curves model itself, so it gets no unfair advantage over "
                  "the rest.",
            next="Run the contest. It takes several minutes, so the results are saved and reloaded here.",
        ),
        dict(
            h="6.2 The league table",
            look="Each model's score on women it never saw, with an honest interval around it.",
            code="""
# Reloads reports/m4_contest.json if it exists; pass rerun=True to run all 150 tuned fits again (several minutes).
results = contest.run_contest(Xc, y)
table = contest.league_table(results, Xc, y)
table[["auc", "auc_low", "auc_high", "brier", "auc_45plus", "seconds"]]""",
            read="`auc` is the average over the 15 outer splits, with `auc_low` and `auc_high` as its 95% interval. "
                 "`brier` measures how close the predicted risks are to what happened (lower is better). "
                 "`auc_45plus` is the AUC among women 45 and over. `seconds` is how long the whole nested run took "
                 "for that model.",
            means="The curves models lead: tuned 0.854, published 0.852. Then the stacked ensemble at 0.849, the "
                  "support vector machine and random forest at 0.840, straight lines at 0.839, XGBoost 0.837, the "
                  "neural network 0.835, gradient boosting 0.832 and k-nearest neighbours 0.827. The intervals are "
                  "wide (about 0.81 to 0.89) and overlap heavily: with 768 women, every model's score is uncertain "
                  "by several hundredths. The curves models also have the best Brier scores and the best AUC for "
                  "older women.",
            next="Overlapping intervals don't settle it, because every model faced the same splits. Compare each "
                 "one with the published model split by split.",
        ),
        dict(
            h="6.3 Is anything really better? The corrected test",
            look="For each model, the difference from the published model, and whether chance can explain it.",
            code="""
display(table[["vs_published", "vs_low", "vs_high", "p_value", "wins"]].drop(index=contest.PUBLISHED))
figures.m4_contest();""",
            read="`vs_published` is the average AUC difference from the published model on the same 15 splits, with "
                 "its 95% interval (`vs_low` to `vs_high`). `p_value` is the chance of a difference at least this "
                 "large if the two models were really equal; below 0.05 is the usual bar. `wins` is the share of "
                 "splits where the model beat the published one. In the chart, a bar that crosses the black line "
                 "means the difference could be zero.",
            means="Nothing beats the published model. The tuned curves model is ahead by 0.002, but its interval "
                  "runs from -0.003 to +0.007 and p is 0.36: indistinguishable. Every other model is behind, by 0.003 "
                  "(the ensemble) to 0.025 (k-nearest neighbours). Only k-nearest neighbours is clearly worse "
                  "(p = 0.02); the rest are probably worse, losing on 73% to 93% of splits, but the test can't rule "
                  "out chance for any single one of them.",
            next="The test used here is a corrected one. See why the correction matters, using the earlier claim "
                 "that curves beat straight lines.",
        ),
        dict(
            h="6.4 Why the correction matters",
            look="The curves-versus-straight-lines comparison from Milestone 3, re-tested with a naive t-test and "
                 "with the corrected one.",
            code="""
import numpy as np
from scipy import stats
curves, straight = np.array(results[contest.PUBLISHED]["auc"]), np.array(results["Straight lines, tuned"]["auc"])
n_test = len(y) / contest.OUTER_FOLDS
print(f"Curves ahead by {np.mean(curves - straight):.3f} AUC, winning {np.mean(curves > straight):.0%} of splits")
print(f"Naive paired t-test:      p = {stats.ttest_rel(curves, straight).pvalue:.3f}")
print(f"Corrected resampled test: p = {contest.corrected_ttest(curves, straight, len(y) - n_test, n_test)['p']:.3f}")""",
            read="A naive t-test treats the 15 splits as 15 independent experiments. They aren't: each pair of "
                 "training sets shares most of its women, so the scores move together and look more consistent than "
                 "they really are. The corrected test (Nadeau and Bengio, 2003) widens the uncertainty to allow for "
                 "that overlap.",
            means="The naive test says the curves' lead over straight lines (0.013, winning 87% of splits) is "
                  "convincing (p = 0.006). The corrected test says it is suggestive, not proven (p = 0.16). Milestone "
                  "3's conclusion stands in direction, with the gain consistent across splits and concentrated in "
                  "older women, but it is recorded in the findings as likely rather than firm. This is the main "
                  "reason to use a corrected test: an uncorrected one would have made most of the contest look "
                  "decisive too.",
            next="Check whether tuning found stable settings, or wandered.",
        ),
        dict(
            h="6.5 What did tuning choose?",
            look="Whether the best settings are consistent from split to split. Stable choices suggest a real "
                 "optimum; scattered ones suggest the setting barely matters.",
            code="""
for name in ["Curves, tuned", "Support vector machine", "XGBoost"]:
    print(name)
    display(contest.tuned_settings(results, name).describe().loc[["min", "50%", "max"]])""",
            read="Each table summarises the settings Optuna chose in the 15 outer splits: the smallest, the middle "
                 "(50%) and the largest. For the curves model, `C` controls smoothing (smaller means smoother curves) "
                 "and `n_knots` how many places a curve may bend.",
            means="The curves model chose stronger smoothing than the published model in every split (C between "
                  "0.01 and 0.16, against 0.3 on the page), and 3 bend points rather than 4 in 9 of the 15 splits. "
                  "In other words, slightly smoother curves are preferred, which fits the suspicion that BMI's flat "
                  "stretch is noise. The support "
                  "vector machine's C ranged from 0.02 to 96, meaning its score barely depends on it. XGBoost almost "
                  "always chose depth 2, very shallow trees, which behave much like adding up one effect per input: "
                  "the same structure as the curves model.",
            next="See whether combining model types helps.",
        ),
        dict(
            h="6.6 Does combining models help? (E14)",
            look="Whether a stacked ensemble, which blends four different model types, beats its best member.",
            code="""
from src.contest import stacked_ensemble
print("Members:", ", ".join(name for name, _ in stacked_ensemble().estimators), "| combined by: logistic regression")
table.loc[[contest.PUBLISHED, "Stacked ensemble"], ["auc", "brier", "auc_45plus", "vs_published", "p_value"]]""",
            read="Stacking trains four models (the curves model, a support vector machine, a random forest and "
                 "gradient boosting), then a final logistic regression learns how much to trust each one's risk. "
                 "The final step learns from predictions the members made on data they hadn't seen, so it isn't "
                 "fooled by their over-confidence.",
            means="The ensemble scores 0.849, slightly below the curves model alone (0.852), with a worse Brier "
                  "score. Blending only helps when the members get different women wrong; here they mostly agree, "
                  "and the weaker members dilute the best one. It also takes 17 times longer to run and is far "
                  "harder to explain.",
            next="Make the decision.",
        ),
        dict(
            h="6.7 Decision: keep the published model",
            look="The case for and against changing the page's model.",
            code="""
table.loc[[contest.PUBLISHED, "Curves, tuned", "Stacked ensemble", "Support vector machine"],
          ["auc", "auc_low", "auc_high", "brier", "auc_45plus", "vs_published", "p_value"]]""",
            read="The published model against the three closest rivals, on every measure the contest produced.",
            means="Decision D14: the page keeps the curves model from Milestone 3. The tuned version's 0.002 gain is "
                  "within noise, and switching would change every number on the page for no real improvement. The "
                  "preference for smoother curves is noted for Milestone 6, which looks at what the model does with "
                  "each input. The bigger lesson is about the data: on 768 women and six inputs, every reasonable "
                  "method lands between 0.83 and 0.85. The ceiling is set by the information in the measurements, "
                  "not by the choice of algorithm.",
            next="Milestone 5 turns from ranking to deciding: where the cut-off should go once the costs of missed "
                 "cases and false alarms are stated, and whether different age groups need different cut-offs (B12).",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 7
build(
    "07_decisions_and_costs.ipynb",
    "7. Deciding: rebalancing, costs and the cut-off",
    "Until now the project asked how well the model ranks women by risk. Screening needs a decision: at what risk do we "
    "send a woman for a follow-up test? This notebook asks three questions. Does rebalancing the data, a standard "
    "remedy when one group is smaller, help (E15)? What does each cut-off cost, once we say how bad a missed case is "
    "compared with an unnecessary test (E16)? And do different age groups need different cut-offs (E31, the fix for B12)?",
    [
        dict(
            h="7.1 The set-up",
            look="How lopsided the two groups are, and what the page used before this milestone.",
            code="""
import json
import numpy as np
from src.data import load_data, FEATURES
from src.modeling import current_features
from src import decision, figures

X, y = load_data()
Xc = X[current_features(FEATURES)]
m5 = json.load(open("../reports/metrics.json"))["m5"]
print("Women with diabetes:", int(y.sum()), "| without:", int((y == 0).sum()), f"| share with diabetes: {y.mean():.0%}")
print("Old rule: the highest cut-off that still catches 85% of cases =", m5["costs"]["old_cutoff"])""",
            read="The counts are the whole data: 268 women with diabetes and 500 without. The old rule was set in Milestone 1: "
                 "pick the highest cut-off that still catches at least 85% of the real cases.",
            means="The groups are lopsided but not extremely so: about one woman in three has diabetes. The 85% target was a "
                  "round number chosen early, without any statement of what it costs to miss a case. This notebook replaces it "
                  "with a reason.",
            next="First, the usual remedy for lopsided groups: rebalance the training data.",
        ),
        dict(
            h="7.2 Does rebalancing help? (E15)",
            look="Whether giving the smaller group more weight, or inventing extra training cases for it, finds more cases for "
                 "the same number of false alarms.",
            code="""
reb = decision.compare_rebalancing(Xc, y)
import pandas as pd
table = pd.DataFrame(reb["table"]).T
table[["auc", "brier", "calibration_slope", "mean_risk", "sensitivity_at_half", "false_alarms_for_85"]]""",
            read="Three versions of the published model, scored on the same 50 splits. `Class weights` makes each woman with "
                 "diabetes count about twice as much as a healthy woman when the model learns. `SMOTE` (synthetic minority "
                 "over-sampling) invents new cases by blending two real ones, inside each training set only, so the test women "
                 "never influence it. `auc` measures ranking and `brier` measures the quality of the risks (lower is better). "
                 "`mean_risk` is the average predicted risk; the real rate is 35%. `sensitivity_at_half` is the share of cases "
                 "caught at a cut-off of 0.50. `false_alarms_for_85` is how many of the 500 healthy women must be flagged to "
                 "catch 85% of the cases.",
            means="The ranking is unchanged: AUC 0.853, 0.853 and 0.851. Catching 85% of cases takes 154, 152 and 153 false "
                  "alarms, a difference of two women out of 500. What changed is the risks: the average predicted risk rises "
                  "from 35% to 44% and 43%, the Brier score gets worse (0.149 to 0.157 and 0.158), and the calibration slope "
                  "drops. At the usual 0.50 cut-off both rebalanced models catch more cases (79% and 77%, against 61%), but "
                  "that is only because they flag more women.",
            next="Check that claim directly: is a rebalanced model just the published model with a different cut-off?",
        ),
        dict(
            h="7.3 Rebalancing is a cut-off in disguise",
            look="Whether the published model, with its cut-off moved, reproduces what the class-weighted model does at 0.50.",
            code="""
from src.evaluate import confusion_at
published = np.array(reb["oof"]["Published model"])
rows = []
for name in ["Class weights", "SMOTE"]:
    p = np.array(reb["oof"][name])
    c = confusion_at(y, p, 0.5)
    flagged = c["tp"] + c["fp"]
    grid = np.arange(0.02, 0.9, 0.005)
    t = min(grid, key=lambda t: abs(sum(confusion_at(y, published, t)[k] for k in ("tp", "fp")) - flagged))
    d = confusion_at(y, published, t)
    rows.append({"model": name, "its cut-off": 0.5, "flags": flagged, "catches": c["tp"], "false alarms": c["fp"],
                 "published model's matching cut-off": round(t, 3), "catches (published)": d["tp"], "false alarms (published)": d["fp"]})
pd.DataFrame(rows).set_index("model")""",
            read="Each row takes a rebalanced model at its cut-off of 0.50, counts how many women it flags, then finds the cut-off "
                 "at which the published model flags the same number. If rebalancing found extra cases, the published model "
                 "would catch fewer at that matching cut-off.",
            means="The class-weighted model at 0.50 flags 330 women, catching 211 with 119 false alarms. The published model "
                  "with its cut-off moved to 0.35 flags the same 330 and gets the identical 211 and 119. SMOTE is within a "
                  "woman or two (206 caught against 208). Rebalancing found nothing that a cut-off couldn't, and in exchange it "
                  "made the risks overestimate: it is a cut-off move that also breaks the probabilities. Decision D18: the "
                  "page does not rebalance.",
            next="So the cut-off is the lever. The next question is where to put it, and that needs a cost.",
        ),
        dict(
            h="7.4 What a missed case costs (E16)",
            look="The cut-off that is cheapest overall, once we say how bad a missed case is compared with a false alarm.",
            code="""
for r, v in m5["costs"]["ratios"].items():
    print(f"Missed case = {r}x a false alarm  ->  cut-off 1/(1+{r}) = {v['cutoff']:.2f}")
rows = []
for r, v in m5["costs"]["ratios"].items():
    rows.append({"a missed case is": f"{r}x a false alarm", "cut-off": v["cutoff"], **{k: v["per_100"][k] for k in ("tp", "fn", "fp", "tn")}})
pd.DataFrame(rows).set_index("a missed case is").rename(columns={"tp": "caught", "fn": "missed", "fp": "false alarms", "tn": "healthy, not flagged"})""",
            read="Suppose flagging a healthy woman costs 1 unit (an unnecessary blood test) and missing a woman with diabetes "
                 "costs 2, 5 or 10 units. For a woman whose true risk is p, flagging her costs 1 - p units on average (the "
                 "chance she is healthy) and not flagging her costs ratio times p. Flagging is cheaper when p is at least "
                 "1 / (1 + ratio). So the best cut-off is simply that number. The table shows what each cut-off does to every 100 "
                 "women screened (35 of them have diabetes).",
            means="At 2 times the cut-off is 0.33: 28 of 35 cases caught, 17 false alarms. At 5 times it is 0.17: 33 caught, "
                  "30 false alarms, and 63 of every 100 women are flagged. At 10 times it is 0.09: 34 caught, 39 false alarms, "
                  "73 flagged. Each step down in cut-off catches a few more cases and flags many more healthy women. The old "
                  "85% rule (0.28) quietly assumed a missed case is only about 2.6 times as bad as a false alarm. Decision "
                  "D17: the page uses 5 times, so the published cut-off is 0.17, and the three choices sit side by side on the "
                  "page so a reader can pick their own.",
            next="Check the arithmetic against the data: is that cut-off really the cheapest, and how precise must it be?",
        ),
        dict(
            h="7.5 Is the cost-based cut-off really the cheapest?",
            look="The total cost per 100 women at every cut-off, and how much the cheapest cut-off moves when the women are resampled.",
            code="""
figures.m5_cost();
pd.DataFrame({r: {"cost-based cut-off": v["cutoff"], "cheapest in this data": v["empirical_best"]["best"],
                  "cheapest, 95% range": f"{v['empirical_best']['low']:.2f} to {v['empirical_best']['high']:.2f}"}
              for r, v in m5["costs"]["ratios"].items()}).T""",
            read="Each panel assumes a different cost for a missed case. The green line is the cost per 100 women at every "
                 "cut-off, in false-alarm units, so lower is better. The dot marks the formula's cut-off. The table compares "
                 "it with the cut-off that happens to be cheapest in this data, and the range that cut-off covers when the 768 "
                 "women are resampled 1,000 times.",
            means="The formula's cut-off lands inside the data's 95% range in all three cases (0.33 in 0.23 to 0.43, 0.17 in 0.09 "
                  "to 0.26, 0.09 in 0.09 to 0.17), so the formula and the data agree. The ranges are wide, which is the "
                  "other lesson: the curves are flat near their lowest point, so a cut-off a few hundredths either side costs "
                  "almost the same. The cut-off does not need to be exact, but the cost ratio matters, because the three curves "
                  "have different minima.",
            next="Compare the choice with the alternatives a reader might suggest.",
        ),
        dict(
            h="7.6 Against the alternatives",
            look="The cost of the cost-based cut-off against testing everybody, testing nobody, the old rule and a glucose rule.",
            code="""
pd.DataFrame({f"{r}x": v["cost_per_100"] for r, v in m5["costs"]["ratios"].items()}).round(1)""",
            read="Cost per 100 women, in false-alarm units, for each policy at each cost ratio. Lower is better. "
                 "`Cost-based cut-off` is the model with its cut-off from 7.4.",
            means="At 5 and 10 times the cost-based cut-off is the cheapest policy. At 5 times it costs 40, against 45 for the old "
                  "rule, 65 for testing everybody, 95 for the glucose-of-140 rule and 175 for testing nobody. At 2 times the old "
                  "rule is a hair cheaper (30.2 against 30.9), since 0.28 is close to 0.33; the difference is noise. The higher "
                  "the cost of a missed case, the more the old rule costs: at 10 times it is 71 against 45.",
            next="The same comparison drawn the way medical decision-making papers draw it.",
        ),
        dict(
            h="7.7 The decision curve",
            look="Whether the model is worth using at all, compared with testing everybody or nobody, at every cut-off.",
            code="""
figures.m5_decision_curve();
pd.DataFrame({r: {"cut-off": v["cutoff"], "net benefit, model": round(v["net_benefit"], 3),
                  "net benefit, test everybody": round(v["net_benefit_all"], 3)}
              for r, v in m5["costs"]["ratios"].items()}).T""",
            read="Net benefit counts the cases found, minus the false alarms weighted by how costly they are relative to a "
                 "missed case: net benefit = true positives per woman - false positives per woman x cut-off / (1 - cut-off). "
                 "Testing nobody scores zero by definition. A model is worth using at a cut-off where its line sits above both "
                 "the zero line and the dashed 'test everybody' line.",
            means="At each of the three chosen cut-offs the model beats testing everybody. The gap is wide at 2 times (0.196 "
                  "against 0.028) and 5 times (0.267 against 0.216), and narrow at 10 times (0.305 against 0.285): if a missed "
                  "case is 10 times worse than a false alarm, you would test almost everyone anyway, and the model's job "
                  "shrinks to sparing the quarter of women who are very unlikely to have diabetes. That is a real limit "
                  "on what screening can do when almost no miss is acceptable.",
            next="Now the question left over from Milestone 3: do age groups need their own cut-offs?",
        ),
        dict(
            h="7.8 Do age groups need different cut-offs? (E31, B12)",
            look="Whether the model's risks are equally trustworthy in each age group, and how many cases each group has caught.",
            code="""
display(pd.DataFrame(m5["bands"]["calibration"]).set_index("band"))
figures.m5_bands();""",
            read="The table compares the average predicted risk with the real rate in each age band, and `slope` is the "
                 "calibration slope (1.00 is ideal; below 1 means the risks are too extreme). The chart shows the share of each "
                 "band's cases caught at the old 0.28 cut-off and at the three cost-based cut-offs.",
            means="The average predicted risk matches the real rate in every band (22% against 21% under 30, 48% against 49% for "
                  "30 to 44, 50% against 50% for 45 and over). The slope is 1.02, 0.91 and 0.77: the risks are right on average, "
                  "but for women 45 and over they are somewhat too extreme, a hint that a shared cut-off is least precise there. "
                  "Under 30 the old rule caught 71% of cases; the 5-times cut-off catches 85%, and the 10-times 95%. At 2 times "
                  "it would be worse (62%), because a higher cut-off hurts the group whose risks are lowest. B12 is fixed by "
                  "choosing the cut-off from a cost, not by adding age rules.",
            next="Test directly whether a cut-off per age band would do better still.",
        ),
        dict(
            h="7.9 Separate cut-offs per age group",
            look="Whether choosing a cut-off for each age band, on training women only, beats one shared cost-based cut-off.",
            code="""
rows = []
for r, v in m5["bands"]["by_ratio"].items():
    for name, pol in v["policies"].items():
        rows.append({"a missed case is": f"{r}x", "policy": name, "cost per 100": round(pol["cost_per_100"], 1),
                     "cut-offs (under 30 / 30-44 / 45+)": " / ".join(f"{b['cutoff']:.2f}" for b in pol["bands"]),
                     "caught under 30": f"{pol['bands'][0]['sensitivity']:.0%}"})
pd.DataFrame(rows).set_index(["a missed case is", "policy"])""",
            read="Three policies, scored the honest way: each age band's cut-off is chosen on training women and used on test "
                 "women, over the same 50 splits. `Single cost-based cut-off` is the formula, shared by every band. `Cheapest "
                 "per band` picks the cheapest cut-off within each band. `Catch 85% in every band` is the old rule applied "
                 "band by band.",
            means="One shared cut-off wins or ties. At 5 times it costs 40.5 against 42.5 for a cut-off per band; at 10 times "
                  "44.5 against 47.7; at 2 times the per-band version is 30.0 against 30.9, about one unit better, which is "
                  "inside the noise. Catching 85% in every band is clearly worse (52 and 79 at 5 and 10 times), because it "
                  "demands a cut-off of 0.42 for older women who are mostly cases anyway. With 66 to 118 cases per band, a "
                  "cut-off fitted to one band mostly fits its noise. Decision D19: no age-specific cut-offs; E31 is closed as "
                  "not needed.",
            next="Milestone 6 turns from deciding to explaining: why did the model give this woman this number?",
        ),
    ],
)

# ---------------------------------------------------------------- notebook 8
build(
    "08_explainability.ipynb",
    "8. Explaining: why did the model give this woman this number?",
    "Until now the notebooks asked how well the model predicts. This one asks why it predicts what it does. Four "
    "questions: which inputs does it rely on (E17); what shape does each effect have, and is the odd flat stretch in the "
    "BMI curve real (a question left over from Milestone 4); do the explanation tools agree with the model's exact "
    "arithmetic, and with a very different kind of model; and for one woman the model flags, what small change would "
    "move her below the cut-off (E18)? Because the published model is a sum of one effect per input, every "
    "explanation here can be checked against an exact answer. That is unusual and is part of the point.",
    [
        dict(
            h="8.1 Which inputs does the model rely on?",
            look="How much worse the model gets on women it has never seen when one input is scrambled.",
            code="""
import json
import numpy as np
from src.data import load_data, FEATURES
from src.modeling import current_features
from src import explain, figures

X, y = load_data()
Xc = X[current_features(FEATURES)]
m6 = json.load(open("../reports/metrics.json"))["m6"]
table = pd.DataFrame(m6["permutation"]).T
table.columns = ["AUC lost (mean)", "low (2.5%)", "high (97.5%)", "share of splits with a loss"]
table.sort_values("AUC lost (mean)", ascending=False)""",
            read="This is permutation importance. For each of the 50 train/test splits the model is fitted on the training "
                 "women. Then one input of the test women is shuffled among them, five times, which breaks its link with the "
                 "outcome, and the fall in AUC is recorded. A bigger fall means the model leaned harder on that input. `low` "
                 "and `high` are the 2.5th and 97.5th percentiles of the fall over the 50 splits, so they describe how much it "
                 "varies from split to split.",
            means="Glucose dominates: scrambling it costs 0.17 of AUC, more than BMI (0.07) and age (0.04) together. The "
                  "other three inputs cost 0.01 or less, and for each of them the range includes zero or sits right at it, so "
                  "the model gains nothing measurable from pregnancies, blood pressure or family history on women it has "
                  "not seen. That is no reason to drop them yet: they cost nothing to keep, they add a little to the odds "
                  "the page displays, and the AUC is a blunt measure.",
            next="Importance says how much an input matters, not how. Next, the shape of each effect.",
        ),
        dict(
            h="8.2 What shape does each effect have?",
            look="How a woman's predicted risk moves as one input changes while everything else stays as it is.",
            code="""
figures.m6_curves();""",
            read="Each faint green line is one of 40 real women (a curve known as an ICE line, for individual conditional "
                 "expectation): her predicted risk if only that one measurement were changed. The bold black line is the "
                 "average of those lines over all 768 women (a partial dependence curve). The lines run parallel because "
                 "the model is a sum of one effect per input: changing glucose moves everyone the same way on the odds "
                 "scale, and the probability scale squeezes the high-risk lines against 100%.",
            means="Glucose rises steadily. Age rises to about 50 and then falls, matching the real rates found in "
                  "Milestone 3 (71% at 50 to 54, 28% at 60 and over). BMI rises sharply to about 30, then runs flat until "
                  "about 40 and climbs again. The parallel lines are an assumption of the model, not a finding: a "
                  "model with interactions could show lines that cross. Milestone 3 tested age by glucose and found "
                  "nothing, which is why the model has none.",
            next="The flat stretch in BMI was noted in Milestone 4 as possibly noise. Test it directly.",
        ),
        dict(
            h="8.3 Is the flat stretch in BMI real?",
            look="Whether the BMI curve really levels off between about 30 and 42, or only looks that way because few women sit there.",
            code="""
bp = m6["bmi_plateau"]
pd.DataFrame({
    "log-odds rise": {"BMI 24 to 30": bp["estimate"]["low_to_start"], "BMI 30 to 42": bp["estimate"]["plateau"]},
    "95% range, low": {"BMI 24 to 30": bp["interval"]["low_to_start"][0], "BMI 30 to 42": bp["interval"]["plateau"][0]},
    "95% range, high": {"BMI 24 to 30": bp["interval"]["low_to_start"][1], "BMI 30 to 42": bp["interval"]["plateau"][1]},
    "odds multiplier": {"BMI 24 to 30": bp["odds"]["24_to_30"], "BMI 30 to 42": bp["odds"]["30_to_42"]},
})""",
            read="The model was refitted on 300 resamples of the 768 women (the bootstrap), and the BMI curve read at 24, 30 and "
                 "42 with every other input at its median. The table shows the rise in log-odds from 24 to 30 and from 30 to 42, "
                 "with the 2.5th and 97.5th percentiles over the 300 refits. Log-odds are the scale the model adds on; "
                 "`odds multiplier` is e raised to that rise.",
            means="From BMI 24 to 30 the odds multiply by about 5.4, and the whole range (3.1 to 10.9) is well above 1. From 30 to "
                  "42 the best estimate is only a 1.2 times rise, with a range from 0.7 to 2.3 times. If the early slope had simply "
                  "continued, the rise over those 12 points would be about 29 times (3.4 in log-odds), far outside the range. So "
                  "the levelling-off is real. What the data cannot say is whether it is exactly flat: a rise of up to 2.3 times "
                  "is still possible, and so is a slight fall. Compare with the second test, below.",
            next="The second test asks whether the curve matters for prediction.",
        ),
        dict(
            h="8.4 Does a straight line for BMI do as well?",
            look="Whether bending BMI improves predictions on women the model hasn't seen.",
            code="""
cv = bp["curve_vs_straight_bmi"]
pd.Series({"AUC, curve for BMI (published)": cv["auc_curve"], "AUC, straight line for BMI": cv["auc_straight"],
           "difference": cv["diff"], "95% range, low": cv["low"], "95% range, high": cv["high"],
           "p-value (corrected t-test)": cv["p"], "share of splits the curve wins": cv["wins"]}).to_frame("value")""",
            read="Two models scored on the same 50 splits: the published one, and one that is identical except that BMI is a "
                 "straight line (curves remain for glucose and age). The difference is tested with the corrected resampled "
                 "t-test used in Milestone 4, which allows for the splits sharing women.",
            means="The curve scores 0.006 higher and wins on 80% of splits, but the range runs from slightly below zero to 0.015 "
                  "and the p-value is 0.15, so this could be chance. The curve is kept, because it matches what the bootstrap "
                  "showed and costs nothing. But the claim is modest: the BMI curve's *shape* is real, and its *benefit* "
                  "for prediction is likely, not proven. This is the same rating that curves-versus-straight-lines received "
                  "in Milestone 4.",
            next="Now check the explanation tools themselves, starting with SHAP.",
        ),
        dict(
            h="8.5 Does SHAP agree with the model's exact answer?",
            look="Whether a standard explanation tool reproduces the contributions we can compute exactly for this model.",
            code="""
sh = m6["shap"]
pd.DataFrame({"mean |SHAP| (log-odds)": sh["mean_abs_shap"], "share": sh["share"]}).sort_values("share", ascending=False).assign(
    **{"largest gap from exact": sh["max_abs_difference"]}).iloc[:1].T.rename(columns=lambda c: "glucose" if c == "Glucose" else c)""",
            read="SHAP gives each input a share of the credit for a woman's prediction, by averaging its effect over every "
                 "order in which inputs could be added. The model here is a sum of one effect per input, so the exact answer "
                 "is simple: each input's contribution minus the average contribution over a background group of 100 women. "
                 "The exact SHAP algorithm was run for all 768 women and compared with that formula. The table shows glucose's "
                 "average size and the single largest gap found anywhere, for any woman or input.",
            means="Glucose carries 36% of the total importance. The largest gap between SHAP and the exact answer anywhere is 0.002 "
                  "in log-odds, and the average gap is under a millionth, so SHAP is working as it should. On a model this "
                  "simple SHAP tells us nothing new. The point is that it is now verified, so it can be trusted on a model "
                  "where no exact answer exists. The next section does that.",
            next="Use SHAP on a boosted-tree model and see whether it tells the same story.",
        ),
        dict(
            h="8.6 Does a very different model tell the same story?",
            look="Whether the ranking of inputs belongs to the data or to the kind of model.",
            code="""
b = m6["boosted"]
pd.DataFrame({"published model": b["published_share"], "boosted trees (XGBoost)": b["boosted_share"]}).mul(100).sort_values(
    "published model", ascending=False)""",
            read="Each column is the share of total mean |SHAP| going to that input, in per cent. The boosted-tree model is "
                 "XGBoost with 200 small trees, fitted on all 768 women. It has no formula to read, so SHAP is the only "
                 "way to see what it uses. The two columns should be compared by order and rough size.",
            means="The order is identical (rank correlation 1.0): glucose, BMI, age, family history, pregnancies, blood pressure. "
                  "The sizes are close too, with the boosted model giving slightly more weight to family history, pregnancies "
                  "and blood pressure. So the ranking is about the data. A model that finds its own shapes, without being told "
                  "where to put curves, ends up where the curved logistic model did.",
            next="Put all this together for one woman, and then ask what would change her result.",
        ),
        dict(
            h="8.7 One woman, explained",
            look="Each input's push on one woman's risk, and the number that results.",
            code="""
import numpy as np
from src import nonlinear as nl
m = nl.export_curves_model(Xc, y)
woman = {"Pregnancies": 2, "Glucose": 122, "BloodPressure": 72.0, "BMI": 31.0, "DiabetesPedigreeFunction": 0.45, "Age": 30}
vals = [woman[f] for f in m["features"]]
risk = 1 / (1 + np.exp(-explain._z(m, vals)))
table = pd.DataFrame({"her value": vals, "typical woman": m["median"],
                      "odds multiplier": [np.exp(np.interp(v, m["grid"][f], m["contribution"][f])) for f, v in zip(m["features"], vals)]},
                     index=m["features"])
print(f"Predicted risk: {risk:.0%}")
table""",
            read="`odds multiplier` is the same number the project page shows beside each bar: how much that input multiplies the "
                 "odds compared with a typical woman (one with the median of every measurement). Multipliers above 1 push "
                 "the risk up and below 1 push it down. Multiplying all six and applying them to a typical woman's odds "
                 "gives the risk.",
            means="This woman's risk is 40%. Her glucose of 122 (a little above typical) multiplies the odds by 1.15, "
                  "her BMI of 31 lowers them slightly (0.93, because the typical BMI here is 32), and everything else is within "
                  "a few per cent of typical. She sits a little above the cut-off at 0.33 and well above at 0.17. Notice "
                  "that no single input explains her number: a woman near the middle on everything lands near the "
                  "population rate, which is 35%.",
            next="Ask the question the page now answers: how far is she from the cut-off?",
        ),
        dict(
            h="8.8 What would change this? (E18)",
            look="The smallest drop in glucose, or in BMI, that would move her below each cut-off.",
            code="""
rows = []
for ratio, cut in [(2, 0.33), (5, 0.17), (10, 0.09)]:
    g = explain.counterfactual(m, vals, cut, "Glucose", explain.LIMITS["Glucose"])
    b = explain.counterfactual(m, vals, cut, "BMI", explain.LIMITS["BMI_share"] * woman["BMI"])
    rows.append({"a missed case is": f"{ratio}x a false alarm", "cut-off": cut,
                 "glucose drop needed (mg/dL)": g["needed"], "BMI drop needed": b["needed"]})
pd.DataFrame(rows).set_index("a missed case is")""",
            read="For each cut-off the code lowers one input at a time, in steps of 1 mg/dL for glucose and 0.1 for BMI, until her "
                 "risk falls below the cut-off, and reports the first step that works. It stops at a modest limit: 20 mg/dL for "
                 "glucose, 10% of her BMI for BMI. A blank (None) means no change within the limit is enough. The page runs "
                 "the same search in your browser, and a test checks that the two agree.",
            means="At the 2 times cut-off (0.33) her glucose would need to be 10 lower (112), or her BMI 2.1 lower (28.9). At 0.17 "
                  "and 0.09, nothing within the limits is enough, because she is too far above those lower cut-offs. This is a "
                  "statement about the model: it says what moves the number. It does not say that lowering her glucose "
                  "would lower her real risk. The model describes patterns across 768 women, not cause and effect.",
            next="Run the same search over every woman the model flags.",
        ),
        dict(
            h="8.9 How many flagged women are one small change away?",
            look="The share of flagged women whom a realistic change would clear, at each of the three cut-offs.",
            code="""
figures.m6_counterfactual();
cf = m6["counterfactual"]
pd.DataFrame({f"{r}x ({v['cutoff']:.2f})": {"women flagged": v["flagged"], "median risk of flagged women": v["flagged_risk_median"],
              "cleared by glucose alone": v["glucose_alone"], "cleared by BMI alone": v["bmi_alone"], "cleared by either": v["either"]}
              for r, v in cf.items()})""",
            read="The search from 8.8 is run for every flagged woman, using the full-data model for both her flag and her "
                 "change. Each bar is the share of flagged women cleared by lowering glucose by up to 20 mg/dL, lowering BMI by "
                 "up to 10%, or either. The table adds how many women are flagged and the median risk among them.",
            means="At the page's 5 times cut-off (0.17), 477 of 768 women are flagged and their median risk is 49%, almost three "
                  "times the cut-off. A realistic change clears only 29% of them (glucose alone 28%, BMI alone 13%; the typical "
                  "change needed is 12 mg/dL of glucose or 1.4 BMI points). Among flagged women whose risk is under 30%, 94% can be "
                  "cleared, so the search does work near the cut-off. The share cleared falls as the cut-off falls, from 35% "
                  "at 0.33 to 24% at 0.09, because a lower cut-off flags more women and moves the threshold further away from "
                  "each of them. The honest summary is that most flagged women are not a small change away.",
            next="Decide what goes on the page and in the findings.",
        ),
        dict(
            h="8.10 What this changes",
            look="Whether anything here alters the published model, the cut-off, or the claims.",
            code="""
pd.Series({"Published model changed": "no",
           "Cut-off changed": "no (0.17 at 5 times)",
           "New on the page": "importance chart, curve shapes, 'What would change this?' box",
           "BMI curve": "shape real; benefit likely, not proven (p = 0.15)",
           "SHAP": "verified against the exact answer (largest gap 0.002 log-odds)"}).to_frame("decision")""",
            read="A summary of the decisions this notebook supports, in one table.",
            means="Nothing about the model changes. Milestone 6 adds explanation, not performance. The two claims it sharpens are "
                  "that glucose is the only input the model clearly needs, and that the BMI curve's shape is real while its "
                  "benefit for prediction is only likely. D20 records that the page explains the model by its exact "
                  "contributions and not by SHAP, because the exact version is simpler and SHAP agrees with it.",
            next="Milestone 7 asks how sure the model is about one woman's estimate, which this notebook did not touch: every "
                 "risk here is a single number with no range around it.",
        ),
    ],
)
# ---------------------------------------------------------------- notebook 9
build(
    "09_uncertainty.ipynb",
    "9. Uncertainty: how sure is the model about one woman?",
    "Every risk so far has been a single number, such as 36%. But the model was fitted to 768 women, and a different 768 "
    "would have produced a slightly different model. This notebook asks two questions. How much could one woman's risk "
    "move (E19)? And can the model say \"I can't call this one\", with a guarantee that the rule works on women it has "
    "never seen (E20)? The first is answered with a bootstrap range, the second with conformal prediction. Both are "
    "checked, not assumed: a simulation tests the range, and 300 random splits test the guarantee.",
    [
        dict(
            h="9.1 How far could one woman's risk move? (E19)",
            look="For each woman, the middle 95% of the risks given by 200 versions of the model, each fitted on a resample of the 768 women.",
            code="""
import json
import numpy as np
from src import uncertainty, figures

m7 = json.load(open("../reports/metrics.json"))["m7"]
iv = m7["intervals"]
print(f"{m7['n_boot']} refits; median 95% range is {iv['median_width']:.1%} wide; "
      f"10% of women under {iv['p10_width']:.1%}, 10% over {iv['p90_width']:.1%}")
figures.m7_ranges()
pd.DataFrame(iv["by_risk"]).T.rename(columns={"n": "women", "median_width": "median range width"})""",
            read="The bootstrap draws 768 women at random, with replacement, from the 768, refits the model, and records every "
                 "woman's risk under it. After 200 refits each woman has 200 risks. The 2.5th and 97.5th percentiles of her 200 "
                 "risks are her range. In the chart each position along the bottom is one woman, ordered by risk. The line is "
                 "her estimate and the band is her range. The table gives the median width of the range within four risk bands.",
            means="The typical range is 19 percentage points wide (a woman at 40% could plausibly be anywhere from about 30% to "
                  "50%). Women with a low risk have tight ranges (6 points under 10%), and the widest ranges are in the middle "
                  "and upper part of the scale (27 points for risks of 30% to 60%, 25 points above that). Every estimate lies "
                  "inside its own range. The range describes the model's estimate of risk, not whether the woman will get diabetes.",
            next="A range is only useful if its width is believable. Compare it with a second, independent measure of the spread.",
        ),
        dict(
            h="9.2 Does the spread agree with the cross-validation fits?",
            look="Whether the bootstrap's spread for each woman matches how much her risk varied across the 50 fits from Milestone 2.",
            code="""
sp = m7["spread_check"]
pd.Series({"bootstrap spread / cross-validation spread x 2 (median)": round(sp["median_ratio"], 2),
           "10th percentile of that ratio": round(sp["p10_ratio"], 2),
           "90th percentile of that ratio": round(sp["p90_ratio"], 2),
           "correlation between the two spreads (log scale)": round(sp["corr_log_sd"], 2)}).to_frame("value")""",
            read="Each of the 50 cross-validation fits used 80% of the women, so its risks vary less than a fit on all women "
                 "would. The standard deviation of a woman's risks across those fits, multiplied by 2 (the square root of 0.8 "
                 "divided by 0.2), estimates the same spread as the bootstrap. A ratio near 1 means the two agree, and a "
                 "high correlation means they agree about which women are the most uncertain.",
            means="They agree well: the ratio is 1.08 in the middle, and the correlation is 0.96. Women whose risk is unstable "
                  "across cross-validation fits are the same women whose range is wide. The 10th to 90th percentile ratio (0.79 to 1.62) shows that individual women can differ.",
            next="Agreement in size is not the same as covering the true risk. Test that in a simulation where the truth is known.",
        ),
        dict(
            h="9.3 Does a 95% range really contain the true risk 95% of the time?",
            look="The share of true risks that fall inside the range, in a simulated world where we know the true risk.",
            code="""
cv = m7["coverage"]
pd.Series({"simulated studies": cv["runs"], "refits per study": cv["boot_per_run"],
           "true risk inside the range (average)": f"{cv['coverage_mean']:.1%}",
           "lowest study": f"{cv['coverage_low']:.1%}", "highest study": f"{cv['coverage_high']:.1%}",
           "median range width": f"{cv['median_width']:.1%}",
           "average bias of the estimate": f"{cv['mean_bias']:+.3f}"}).to_frame("value")""",
            read="In the simulation the model fitted to all 768 women is taken as the truth. New outcomes are drawn from its "
                 "risks, a fresh model and bootstrap range are built from each simulated study, and we count how often the "
                 "true risk lies inside. A range labelled 95% should contain the truth about 95% of the time.",
            means="On average the ranges contain the true risk 91.4% of the time. That is below the 95% label, so the ranges are "
                  "a little too narrow, and individual studies vary a lot (79% to 99%). The estimates themselves are not "
                  "biased (average error -0.002). The page therefore says the range is \"about 90%\", and this notebook "
                  "reports the shortfall rather than hiding it. A wider range, such as 99%, would be safer if exact coverage mattered.",
            next="Where are the ranges widest, and do they affect who is flagged?",
        ),
        dict(
            h="9.4 Who has the widest ranges, and does it change who is flagged?",
            look="The traits of the women with the widest ranges, and how many women have the cut-off inside their range.",
            code="""
wt = iv["widest_traits"]
display(pd.DataFrame({"20 widest ranges": [wt["median_glucose"], wt["median_bmi"], wt["median_age"]],
                      "all women": [wt["all_glucose"], wt["all_bmi"], wt["all_age"]]},
                     index=["median glucose", "median BMI", "median age"]))
pd.DataFrame({f"{r}x ({v['cutoff']:.2f})": {"women with the cut-off inside their range": f"{v['share']:.1%}"}
              for r, v in iv["cutoff_inside"].items()})""",
            read="The first table compares the 20 women with the widest ranges with everyone. The second counts, at each of "
                 "the three cut-offs from Milestone 5, the women whose 95% range includes the cut-off. For those women, a "
                 "different sample of 768 women could have put them on the other side.",
            means="The widest ranges belong to older women with high glucose (median age 60, glucose 152.5), where the data "
                  "is thinnest. About one woman in four to five (22% to 25%, depending on the cut-off) has the cut-off inside her "
                  "range, so for those women the decision to flag is not firmly settled by the data. This is a statement "
                  "about the model's stability, not about her health.",
            next="A range around a risk is one kind of honesty. Now a different one: a yes, no or \"can't call\" with a guarantee.",
        ),
        dict(
            h="9.5 A rule that can say \"too uncertain\" (E20)",
            look="How conformal prediction turns a risk into a call, and a toy example small enough to check by hand.",
            code="""
p_cal = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.60, 0.70, 0.80, 0.90])
y_cal = np.array([0,    0,    0,    1,    0,    0,    1,    1,    1,    1])
th = uncertainty.class_thresholds(p_cal, y_cal, alpha=0.2)
print({k: round(v, 3) for k, v in th.items()})
pd.DataFrame({"risk": [0.10, 0.35, 0.50, 0.85],
              "call": [["confident no", "confident yes", "too uncertain", "neither"][c]
                       for c in uncertainty.conformal_calls([0.10, 0.35, 0.50, 0.85], th)]})""",
            read="Set aside a calibration group of women whose outcomes are known. Score each as 1 minus the probability the model "
                 "gave to her true class, and for each class find the score that 80% of its women fall below (with a small "
                 "correction for having few women). A new woman is told \"no\" if her risk is low enough for the no class, "
                 "\"yes\" if it is high enough for the yes class, and \"too uncertain\" if both fit. Because the score moves in "
                 "one direction with the risk, the whole rule is just two risk cut-offs, `low` and `high`.",
            means="With these ten made-up women the rule says \"no\" below 0.30 (`low`, where \"yes\" starts to fit), \"yes\" above 0.45 "
                  "(`high`, where \"no\" stops fitting), and \"too uncertain\" from 0.30 to 0.45. So 0.10 is a confident no, "
                  "0.35 is too uncertain, and 0.50 and 0.85 are confident yes. The guarantee is made separately for each class, "
                  "so it covers women with diabetes and women without, not only the average.",
            next="Run the same rule on the real data, 300 times, on women the rule has never seen.",
        ),
        dict(
            h="9.6 Does the guarantee hold on unseen women?",
            look="Whether the share of women whose true class is in the rule's set matches the level asked for.",
            code="""
cf = m7["conformal"]
figures.m7_coverage()
pd.DataFrame({f"{round((1 - float(a)) * 100)}% level": {
    "target": f"{v['target']:.0%}", "without diabetes covered": f"{v['cov0']:.1%}",
    "with diabetes covered": f"{v['cov1']:.1%}",
    "worst of 300 splits (no / yes)": f"{v['cov0_low']:.0%} / {v['cov1_low']:.0%}"} for a, v in cf.items()})""",
            read="Each of the 300 runs splits the 768 women at random into 60% to fit, 20% to calibrate the rule and 20% to test it. "
                 "A woman is \"covered\" if her true class is in the rule's set, so a woman with diabetes is covered when she "
                 "is called \"yes\" or \"too uncertain\". The table averages the 300 runs and also shows the worst single run.",
            means="The guarantee holds on average: 89.9% and 90.8% at the 90% level, 80.2% and 79.3% at the 80% level. "
                  "Single runs can fall well below the target (the worst run covered only 65% of women with diabetes at the 80% level), because "
                  "each test group holds only about 150 women. The guarantee is about the average over many such groups, "
                  "not about one particular group.",
            next="A guarantee costs something. What does it cost here?",
        ),
        dict(
            h="9.7 The price of the guarantee",
            look="How many women get no confident call, and how often the confident calls are wrong.",
            code="""
figures.m7_zones();
pd.DataFrame({f"{round((1 - float(a)) * 100)}% level": {
    "confident no": f"{v['no']:.1%}", "too uncertain": f"{v['uncertain']:.1%}", "confident yes": f"{v['yes']:.1%}",
    "confident no, but has diabetes": f"{v['no_wrong']:.1%}", "confident yes, but healthy": f"{v['yes_wrong']:.1%}",
    "uncertain women who have diabetes": f"{v['uncertain_cases']:.1%}"} for a, v in cf.items()})""",
            read="The chart shows the women's risks as piles, coloured by outcome, with the grey bands marking where the rule "
                 "withholds a call. The table gives, at each level, the share of women in each call and the share of "
                 "confident calls that are wrong.",
            means="At the 90% level a third of women (33%) are too uncertain to call, and at the 80% level only 7%. The "
                  "confident \"no\" is right more often than the confident \"yes\": at 90%, a \"no\" is wrong for 7% of women "
                  "and a \"yes\" for 25%, because the guarantee is about each real group, not about each call. Many women "
                  "with moderate risks are healthy, so a \"yes\" means her risk is high enough that the rule cannot rule diabetes out "
                  "at that level, not that she probably has it. The grey band is wide at the 90% level because the two groups "
                  "overlap over a long stretch of the scale (about 22% to 56%), and no rule can tell them apart there.",
            next="Decide what goes on the page.",
        ),
        dict(
            h="9.8 What this changes",
            look="Whether anything here alters the published model, the cut-off, or the claims.",
            code="""
pd.Series({"Published model changed": "no",
           "Cut-off changed": "no (0.17 at 5 times)",
           "New on the page": "95% range on the meter, 'How sure can the model be?' box at the 90% and 80% levels",
           "Range honesty": f"95% label, {cv['coverage_mean']:.0%} in simulation: say 'about 90%'",
           "Conformal calls": "guarantee holds on average over 300 splits; individual splits vary"}).to_frame("decision")""",
            read="A summary of the decisions this notebook supports, in one table.",
            means="Nothing about the model changes. Milestone 7 adds honesty about what the model does not know. The cost-based "
                  "cut-off from Milestone 5 still decides who is flagged for a follow-up test, since it reflects the cost of a "
                  "missed case. The conformal call is a separate message about how much to trust the estimate for this woman.",
            next="Milestone 8 asks how the model behaves for different groups of women and whether the data is enough.",
        ),
    ],
)
# ---------------------------------------------------------------- notebook 10
build(
    "10_external_validation.ipynb",
    "10. Does it work on women who are not in this data?",
    "Everything so far used the same 768 women, however carefully they were split. This notebook does what a splitting scheme "
    "cannot: it takes the finished model, fitted once on all 768 Pima women and never changed, and tests it on a different "
    "group. The group is 961 women aged 21 and over from NHANES 2013 to 2014, a US national health survey, who had the same "
    "two-hour glucose test. The two datasets measure different things in places, so the first sections say what had to be "
    "adjusted. Everything after that is a straight answer to: where does the model hold up, and where does it not (E21, E22)? "
    "Only 37 of the 961 women have diabetes by the definition used here, so every number below carries a wide margin.",
    [
        dict(
            h="10.1 What the outside data is, and what had to change",
            look="How the NHANES women were chosen, how diabetes was defined for them, and which model inputs could not be matched.",
            code="""
import json
import numpy as np
from src import external, figures

m8 = json.load(open("../reports/metrics.json"))["m8"]
nh = external.load_table()
parts = m8["outcome_parts"]
pd.Series({"women (21 and over, glucose test done, HbA1c known)": m8["n"],
           "with HbA1c 6.5% or higher": parts["a1c_6_5"],
           "told by a doctor they have diabetes": parts["diagnosed"],
           "both": parts["both"],
           "counted as having diabetes (either)": m8["positives"],
           "share with diabetes": f"{m8['prevalence']:.1%}",
           "share with diabetes in the Pima data": f"{m8['pima_prevalence']:.1%}",
           "share if every glucose of 200 or more were counted too": f"{((nh.Outcome == 1) | (nh.Glucose >= 200)).mean():.1%}",
           "women missing a pregnancy count": m8["missing"]["Pregnancies"],
           "women missing blood pressure": m8["missing"]["BloodPressure"]}).to_frame("value")""",
            read="Each row is a count of women. NHANES is a one-time survey, so it cannot say who will develop diabetes within "
                 "five years, which is what the Pima outcome records. Instead a woman counts as having diabetes if her blood "
                 "HbA1c (a different test from the glucose drink, reflecting average sugar over about three months) is 6.5% or "
                 "higher, or she reports a doctor's diagnosis. Using HbA1c keeps the outcome from simply restating the glucose "
                 "input. Two more differences: NHANES has only yes or no for a relative with diabetes, not the Pima family-history "
                 "score, so that input is left blank and the model fills in the Pima typical value; and women on diabetes medicine "
                 "are not given the glucose drink, so most women already being treated are missing from this sample.",
            means="Only 37 of 961 women (3.9%) count as having diabetes, against 35% in the Pima data. That is a very different "
                  "population, a different outcome, and a small number of cases to measure anything with. The consequences run "
                  "through the rest of the notebook: any result that depends on how many women have diabetes will not carry "
                  "over, and every result is uncertain by a lot. This is a test of how the model behaves away from home, not a "
                  "like-for-like second trial.",
            next="Look at how different the two groups of women are on the inputs the model uses.",
        ),
        dict(
            h="10.2 How different are the women?",
            look="Whether the NHANES women look like the Pima women on the five inputs the two datasets share.",
            code="""
figures.m8_shift();
shift = pd.DataFrame(m8["shift"]).T[["pima_median", "nhanes_median", "pima_p10", "nhanes_p10", "pima_p90", "nhanes_p90"]]
shift.columns = ["Pima median", "NHANES median", "Pima 10th pct", "NHANES 10th pct", "Pima 90th pct", "NHANES 90th pct"]
display(shift)
print(f"Can a simple model tell the two groups apart from these five inputs? AUC {m8['domain_auc']:.2f} (0.50 = identical groups)")""",
            read="The charts overlay the two groups, one input each. The table gives the middle value and the 10th and 90th "
                 "percentiles (the values that 10% of women fall below and above). The last line asks a different question: a "
                 "separate model is trained to guess which dataset a woman came from, using only these five inputs. If the "
                 "groups were alike it would do no better than a coin flip (0.50).",
            means="They are quite different. NHANES women are older (median 48 against 29), lighter (BMI 27.8 against 32.3) and "
                  "have had fewer pregnancies (median 2 against 3). Glucose is similar in the middle (109 against 117), with a "
                  "longer upper tail in NHANES. A model can tell the groups apart with AUC 0.92, so the model will meet many "
                  "combinations of inputs it saw rarely or never in training. Before looking at any result, we should expect "
                  "some degradation.",
            next="First, the easiest question: does the model still rank women correctly?",
        ),
        dict(
            h="10.3 Does the ranking hold up?",
            look="Whether the model still gives women who have diabetes higher risks than women who don't, and whether it beats the glucose reading alone.",
            code="""
a, g = m8["auc"], m8["glucose_only"]
cur = json.load(open("../reports/metrics.json"))["current"]
pd.DataFrame({"AUC": [cur["oof_auc"], a["value"], g["auc"], m8["a1c_only"]["auc"]],
              "95% range": [f"{cur['oof_auc_ci'][0]:.2f} to {cur['oof_auc_ci'][1]:.2f}", f"{a['low']:.2f} to {a['high']:.2f}",
                            f"{g['low']:.2f} to {g['high']:.2f}", "not computed"]},
             index=["Pima women, unseen (50 splits)", "NHANES women: the model",
                    "NHANES women: glucose alone", "NHANES women: model, HbA1c-only outcome (26 cases)"])""",
            read="AUC is the chance that the model gives a randomly chosen woman with diabetes a higher risk than a randomly "
                 "chosen woman without. The range comes from redrawing the 961 women 1,000 times with replacement. The "
                 "glucose-alone row uses the glucose reading as the score, with no model at all. The last row repeats the "
                 "test with the narrower outcome, HbA1c 6.5% or higher only.",
            means="Yes, the ranking holds. The model's AUC is 0.88 on NHANES (0.82 to 0.93), in line with 0.85 on unseen Pima "
                  "women. The two ranges overlap heavily, so there is no sign of a drop. But the glucose reading alone scores "
                  "0.90, as good or better than the model (the ranges overlap almost entirely, so this is not a difference "
                  "we can rely on in either direction). On this data the model adds nothing measurable beyond glucose, which "
                  "fits notebook 08, where glucose did most of the work.",
            next="Ranking is one thing. Are the risk numbers themselves right?",
        ),
        dict(
            h="10.4 Are the risks right?",
            look="Whether the average predicted risk matches the share of women who have diabetes, overall and band by band.",
            code="""
figures.m8_calibration();
c = m8["calibration"]
display(pd.Series({"average predicted risk": f"{c['mean_predicted']:.1%}", "share who have diabetes": f"{c['observed']:.1%}",
                   "calibration slope (1.0 is ideal)": round(c["slope"], 2),
                   "Brier score of the model": round(c["brier"], 3),
                   "Brier score of saying 'everyone has the average risk'": round(c["brier_prevalence_only"], 3)}).to_frame("value"))
pd.DataFrame(m8["bands"]).T.rename(columns={"n": "women", "mean_predicted": "average predicted", "observed": "share with diabetes"})""",
            read="The left chart splits the 961 women into four groups by predicted risk and compares the model's average with "
                 "the share who actually have diabetes. If the model were right, each pair of bars would match. The Brier "
                 "score is the average squared gap between each woman's predicted risk and her outcome (0 or 1), lower is better. "
                 "Comparing it with the score from quoting everyone the same average risk shows whether the individual risks add "
                 "anything.",
            means="The risks are far too high. The model predicts 25% on average and 3.9% had diabetes. In the top band, "
                  "women predicted at 78% on average are 19% in fact. The model's Brier score (0.124) is more than three times "
                  "worse than quoting the 3.9% average to everyone (0.037), so as probabilities they are worse than quoting the average here. "
                  "The calibration slope of 0.85 shows a smaller second problem, that the risks are somewhat too spread out. Most "
                  "of the gap is the base rate: the model learned from a group where 35% have diabetes. Some of it may be that "
                  "NHANES counts diabetes differently (10.1), and this test cannot tell the two apart. Counting every woman with a "
                  "glucose of 200 or more as well, which is closer to how the Pima outcome was set, raises the share only to 7.1%, "
                  "still far below 25%.",
            next="Is the gap fixable? Shift every risk by the same amount, and score it honestly.",
        ),
        dict(
            h="10.5 Can a simple rescaling fix the risks?",
            look="Whether moving every risk up or down by one fixed amount, learned on part of the NHANES women and tested on the rest, makes the risks usable.",
            code="""
r = m8["recalibrated"]
pd.Series({"average risk after the shift": f"{r['mean_predicted']:.1%}", "share who have diabetes": f"{m8['prevalence']:.1%}",
           "Brier score after the shift": round(r["brier"], 4), "Brier score of quoting the average": round(c["brier_prevalence_only"], 4),
           "AUC after the shift": round(r["auc"], 3), "AUC before": round(a["value"], 3)}).to_frame("value")""",
            read="The shift is found by trying values until the average predicted risk equals the share with diabetes. "
                 "It is fitted on four fifths of the women and used on the held-out fifth, five times over, so the score "
                 "is not flattered by choosing the shift on the same women it is tested on.",
            means="The Brier score falls from 0.124 to 0.032, a little better than quoting the average (0.037). The ranking "
                  "(AUC) does not change, because a shift moves every risk the same way. So the model's ranking plus a "
                  "rescaling for the new population gives sensible risks. But the rescaling needs outside women with known "
                  "outcomes, which is exactly what a new place may not have, and with 37 cases the improvement over "
                  "the average is small.",
            next="What does that mean for the cut-offs the page offers?",
        ),
        dict(
            h="10.6 What the cut-offs do on NHANES",
            look="How many cases each cut-off catches and how many false alarms it raises, per 100 women, beside the Pima figures.",
            code="""
figures.m8_cutoffs();
rows = {}
for r in m8["cutoffs"]:
    v = m8["cutoffs"][r]
    rows[f"{r}x (cut-off {v['cutoff']:.2f})"] = {"women flagged": f"{v['flagged']:.0%}", "cases caught": f"{v['sensitivity']:.0%}",
                                                 "healthy women cleared": f"{v['specificity']:.0%}",
                                                 "flagged women who have diabetes": f"{v['precision']:.1%}"}
pd.DataFrame(rows).T""",
            read="The charts show, for each of the three cost-based cut-offs, the cases caught, cases missed and false alarms "
                 "per 100 women. Pima women are on the left (where the cut-offs were chosen) and NHANES women on the right. "
                 "The table adds the share flagged and, of the women flagged, how many truly have diabetes.",
            means="The share of cases caught is similar on NHANES: at the 5 times cut-off, 33 of 37 (89%) against 94% on the Pima "
                  "women, and at 10 times 36 of 37 (97%). The price is very different. With only 4 in 100 women affected, the 5 "
                  "times cut-off flags 45% of NHANES women, so about 42 in every 100 women get an unnecessary follow-up test, "
                  "and only about 1 in 13 flagged women has diabetes (7.6%). The cut-off came from weighing a missed case against "
                  "a false alarm in a group where 35% have diabetes. In a group where 4% do, the balance would sit "
                  "elsewhere: the cut-off formula would need that group's own rate.",
            next="Check whether the picture is the same for different kinds of women (E22).",
        ),
        dict(
            h="10.7 Does it work equally for everyone? (E22)",
            look="Ranking and risk by age group and by the ethnic group recorded in NHANES.",
            code="""
age = pd.DataFrame(m8["by_age"]).T
age["auc"] = age["auc"].map(lambda v: "n/a" if v is None else f"{v:.2f}")
display(age.rename(columns={"n": "women", "positives": "with diabetes", "auc": "AUC",
                            "mean_predicted": "average predicted", "observed": "share with diabetes"}))
grp = pd.DataFrame(m8["by_group"]).T
grp["auc"] = grp["auc"].map(lambda v: "n/a" if v is None or v != v else f"{v:.2f}")
grp.rename(columns={"n": "women", "positives": "with diabetes", "auc": "AUC",
                    "mean_predicted": "average predicted", "observed": "share with diabetes"})""",
            read="The same two measures as before, computed for each subgroup separately: AUC for the ranking, and average "
                 "predicted risk against the share who have diabetes. Be careful with the AUC column: each subgroup has "
                 "between 0 and 17 women with diabetes.",
            means="The ranking looks good in every subgroup large enough to measure (AUC 0.84 to 0.98), with no group where it "
                  "collapses. With 3 to 17 cases per group those numbers are very rough, so this does not show that the model "
                  "works equally well for everyone; it shows that no group is obviously left out. The risks, by contrast, are too "
                  "high in every group (for example 21% predicted against 2% for women under 45, and 17% against 4% for Asian women). The first thing to fix in a new setting is the level of the risks.",
            next="Summarise what this changes.",
        ),
        dict(
            h="10.8 What this changes",
            look="Whether anything here alters the published model, the cut-off, or the claims.",
            code="""
pd.Series({"Published model changed": "no",
           "Cut-off changed": "no (0.17 at 5 times, from Pima costs and base rate)",
           "Claim that survives": "the ranking: AUC 0.88 on NHANES, close to 0.85 on Pima, but no better than glucose alone",
           "Claim that does not": "the risk numbers: 25% predicted against 3.9% actual",
           "New on the page": "an outside-data section and a plainer warning about where the risks can be trusted",
           "Still unknown": "five-year onset in any other group (NHANES is a one-time snapshot)"}).to_frame("decision")""",
            read="A summary of the decisions this notebook supports, in one table.",
            means="Nothing about the model changes. The page now says what has been tested and what has not: the model has been "
                  "run on one outside group, where it ranked well but its risk numbers were far too high. The honest claim is "
                  "that the ranking generalises and the probabilities do not. That is a useful result for the portfolio as "
                  "much as for the model: a model that scores well on its own data and badly on others is a normal finding, "
                  "and the point of this milestone was to measure it.",
            next="Milestone 9 looks inside the Pima data for groups and odd records the labels don't show.",
        ),
    ],
)
print("notebooks written to", OUT)
