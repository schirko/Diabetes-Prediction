# Start here: the project in five minutes

This page explains the whole project without statistics. Words in **bold** are explained in the
[glossary](GLOSSARY.md). For the full list of conclusions, see the [findings](FINDINGS.md).

Last updated: 2026-11-23, after Milestone 7.

## The question

Can a handful of routine health measurements tell us which women are likely to develop diabetes in the next five
years, well enough to decide who should get a follow-up blood test?

That is a **screening** question. Nobody is diagnosed by this model. It only helps decide who to test first.

## The data

768 women, all of Pima heritage, aged 21 and over, from one community near Phoenix, Arizona. The data was collected
by the US National Institute of Diabetes and Digestive and Kidney Diseases and published in 1988. For each woman
there are eight measurements:

- number of pregnancies
- blood sugar two hours after a sugary drink (**glucose**)
- blood pressure
- skin thickness at the back of the arm
- **insulin** level
- **BMI** (weight for height)
- a family-history score
- age

The data also records whether she was diagnosed with diabetes within five years. 268 of the 768 were (35%).
That is far higher than in most populations, which matters later.

## Where it started

This began as a college project that reported 81% accuracy. Rebuilding it carefully showed three problems:

1. **The 81% was luck.** The college version tested the model once, on one random fifth of the women. Testing the
   same method 200 times on different random fifths gives anything from 68% to 84%, averaging 77%. Judging a model
   on one test is like judging a student on one quiz.
2. **Half the insulin readings were never taken.** The file shows a zero wherever a measurement is missing, and
   nobody has an insulin level of zero. 374 women had no insulin reading. The college version treated those gaps as
   real values.
3. **Accuracy hid the real problem.** A "model" that simply says "no diabetes" to everyone is right 65% of the time
   and catches nobody. What matters for screening is how many women with diabetes are caught, and how many healthy
   women are sent for tests they didn't need.

## What the model does

You give it six measurements for one woman: pregnancies, glucose, blood pressure, BMI, family-history score and age.
It gives back a risk: the chance, out of 100, that a woman like her is diagnosed within five years.

Then a **cut-off** turns the risk into a decision. Anyone at or above the cut-off is sent for a follow-up test. The
project page lets you try the model yourself and move the cut-off.

Where should the cut-off go? That depends on how bad a missed case is compared with an unnecessary test. The project
says a missed case is 5 times as bad, which puts the cut-off at 17%. The page also shows 2 times and 10 times.

## How good is it, in everyday terms?

Picture 100 women like the ones in this data. About 35 of them will develop diabetes.

With the cut-off the project uses (a risk of 17% or more, from a missed case counting 5 times a false alarm):

| | Women |
|---|---|
| Have diabetes and are flagged for a test | 33 |
| Have diabetes but are missed | 2 |
| Healthy but flagged anyway (false alarms) | 30 |
| Healthy and correctly not flagged | 35 |

So the model catches 33 of the 35 women who need help. The price is 30 unnecessary follow-up tests. Of the 63 women
it flags, 33 really do have diabetes.

The cut-off is a choice, not a fact. If a missed case is only 2 times as bad (cut-off 33%), the model catches 28 of 35
with 17 false alarms. If it is 10 times as bad (cut-off 9%), it catches 34 with 39 false alarms. At the usual 50%
cut-off, false alarms fall to 9 but 14 cases are missed.

Another way to put its skill: pick one woman who went on to develop diabetes and one who didn't. About 85 times
out of 100, the model gives the first woman the higher risk. That number is called the **AUC** (0.85). A coin flip
would score 0.50.

## What drives the risk

- **Blood sugar (glucose) matters most, by far.** Going from a low reading (86) to a high one (167) multiplies the
  odds of diabetes about 25 times. Glucose alone gets most of the model's skill.
- **Weight (BMI) and age come next.** Each multiplies the odds by about 5 to 6 from low to high.
- **Age does not rise forever.** In this data the diabetes rate climbs to 71% for women aged 50 to 54, then falls to
  28% for women 60 and over. The first models drew a straight line through age, so they kept predicting high risk
  for the oldest women. The current model lets age curve, which fixed most of that.
- **Family history and pregnancies** matter a little to the odds, but scrambling them does not measurably hurt the
  model on women it hasn't seen. **Blood pressure** barely matters once the others are known.
- **Weight rises, then levels off.** Risk climbs steeply with BMI up to about 30 and then flattens. The levelling-off is
  real, but the data can't say whether it is exactly flat.
- **A very different model agrees.** A boosted-tree model ranks the six inputs in the same order, so the ranking is
  about the data, not the type of model.

These are patterns in this group of women, not proof that one thing causes another.

### What would change one woman's result?

For a woman the model flags, the project page asks: what is the smallest drop in glucose, or in BMI, that would move her
below the cut-off? For most flagged women the answer is "more than is realistic". Only about 3 in 10 could be cleared by
glucose 20 lower or BMI 10% lower, because the typical flagged woman is far above the cut-off (median risk 49%, against
17%). This explains the model; it is not health advice.

### How sure is the model about one woman?

An estimate of 36% looks exact but comes from 768 women. Refitting the model 200 times on resamples of them shows how far
her risk could move: typically about 19 percentage points wide, narrow for low risks and widest in the middle of the scale.
For about one woman in four to five the cut-off lies inside that range. In a test where the true risk was known, the
range held it 91% of the time, a little short of the 95% on its label, so the page calls it "about 90%".

The page also says "no", "yes" or "too uncertain" for each woman, using a rule that is guaranteed to be right about 90% of
the time for women with and without diabetes separately. The price is that a third of women (33%) are too uncertain to
call at that level, and a confident "yes" is still wrong for 25 in 100 women, because the guarantee is about each group,
not each call.

## Main conclusions

1. **A fair test changes the story.** The honest accuracy of the original method is 77%, not 81%, and accuracy was
   the wrong measure anyway.
2. **A simple, carefully built model is the best one here.** Ten models were compared in a fair contest, each given
   the chance to tune its own settings without peeking at the test women. The winner is a straightforward one that
   lets three measurements curve. Neural networks, boosted trees and blends of several models did not do better on
   768 women: every sensible method scores about the same, because the limit is what six measurements can tell us.
3. **Insulin and skin thickness can be dropped.** They are missing too often, can't be reliably filled in, and add
   nothing even when they were measured.
4. **The model's risks can be trusted** for women like these: when it says 30%, about 30 in 100 such women had
   diabetes.
5. **It beats a simple rule.** Flagging everyone with a glucose of 140 or more finds 135 of the 268 women with
   diabetes. The model, raising about the same number of false alarms, finds 156.
6. **It is less sharp for older women**, partly because half of them have diabetes. And with one cut-off for
   everyone, it first caught fewer cases among women under 30 than the earlier model did. Choosing the cut-off from
   costs fixed that: 85% of their cases are caught at the 17% cut-off.
7. **Rebalancing the data doesn't help.** Giving the smaller group extra weight, or inventing extra cases, finds no
   more cases for the same false alarms, and it makes the risks too high. Moving the cut-off does the same job.
8. **The model can be explained exactly.** Glucose, BMI and age are the only inputs it clearly relies on. SHAP, a standard
   explanation tool, matches the model's own arithmetic to within 0.002.
9. **Each estimate comes with a range.** It is about 19 points wide and a little too narrow (91% against a 95% label).
   A separate rule can say "too uncertain", with a guarantee that holds on unseen women.

## What it can't tell you

- **Whether it works for anyone else.** Every woman in the data is from one community, studied decades ago. It has
  not been tested on other groups, men, or present-day patients.
- **Anything about an individual's health.** It is a demonstration of how a risk model is built and judged. It is
  not a medical tool.

## What's next

The project is planned as ten **milestones**. Seven are done. The next ones ask:

- **Milestones 8 to 10:** does it work on other populations; are there hidden groups in the data; and production
  polish.

The full plan is in the [roadmap](MILESTONES.md).

## How to read the rest

| If you have | Read |
|---|---|
| 2 minutes | The project page: try the model and move the cut-off |
| 5 minutes | This page |
| 15 minutes | The [findings](FINDINGS.md): every conclusion, how solid it is, and where the evidence is |
| An hour | The notebooks, in order. Each step shows the code, the result, how to read it and what it means |
| A question about a word | The [glossary](GLOSSARY.md) |
| A question about "why" | The [decisions](DECISIONS.md) log |
