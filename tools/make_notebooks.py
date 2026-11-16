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
import numpy as np
import pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")"""


def build(name, title, intro, sections):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
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
    v = res[f"test_{k}"]
    print(f"{k:9s} mean {v.mean():.3f}   spread (sd) {v.std():.3f}")""",
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
print("notebooks written to", OUT)
