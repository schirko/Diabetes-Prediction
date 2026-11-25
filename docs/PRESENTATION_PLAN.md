# Presentation plan (E34)

The plan for a 12-slide deck that tells the whole project to someone who is not an analyst, written before any slide is
built. Once this plan is approved, the deck is built from it as HTML, PowerPoint (.pptx), PNG slide images and PDF, and linked
from the project page with download buttons, the same way as the meat-prices deck.

Last updated: 2026-11-25. Status: **draft for review, nothing built yet.**

## Who it is for, and what it has to do

- **Audience.** A smart person with little analytical training: a hiring manager, a colleague from another field. No formulas,
  no jargon without a one-line explanation. Every term on a slide is in the [glossary](GLOSSARY.md).
- **Job.** Show that the project was done with care, and that the author knows what the model can and cannot be trusted for.
  The honest "works here, fails there" result is the strongest thing in the project, so it gets a slide of its own.
- **Length.** 12 slides at roughly a minute each, plus an appendix that is not presented but is in the downloads.
- **Rules for every slide.**
  1. One headline that reads as a complete sentence and makes sense alone.
  2. One visual (a chart from the project, a drawn picture, or one big number). Never two charts.
  3. At most about 25 words of body text. The speaker notes carry the rest.
  4. Every number comes from [FINDINGS.md](FINDINGS.md) and `reports/metrics.json`; the build fails if a number on a slide
     no longer matches.
  5. Where a slide could overclaim, it says how solid the answer is (firm, likely or open).

## The story, in one line

A simple question, a first result that turned out to be luck, a fair rebuild, a model that is good at ranking and honest
about its limits, and a test on outsiders that shows where it stops working.

## The slides

Each entry gives the headline, the visual, what to say over it, and where the numbers come from.

### 1. Can six routine measurements tell us who should get a diabetes test?
- **Visual.** Title with one image: the risk meter from the project page (a woman's risk shown as a bar with its range).
- **Say.** This is a screening question: who to test first, not who has diabetes. I rebuilt a college project from the ground up.
- **Source.** [Overview](OVERVIEW.md), "The question".

### 2. The data: 768 women from one community, and 35% were diagnosed within five years
- **Visual.** Big number "768" with a row of 100 small figures, 35 shaded.
- **Say.** Pima women near Phoenix, aged 21 and over, published in 1988. Eight measurements each. Diabetes is far more common
  here than in most populations, which matters later.
- **Source.** Overview, "The data"; model card, "Data".

### 3. The 81% in the original was a lucky split
- **Visual.** The lucky-split chart (notebook 02): 200 repeats spread from 68% to 84%, the original's 81% marked.
- **Say.** Testing once is like judging a student on one quiz. Tested 200 times, the same method averages 77%, and the original
  split beat 88% of the others.
- **Source.** FINDINGS 2. Numbers: 68%, 84%, 77%, 81%, 88%.

### 4. Half the insulin readings were never taken, so the model does without them
- **Visual.** A bar for each measurement showing the share missing (insulin 49%, skinfold 30%).
- **Say.** The file writes a zero where nothing was measured. Filling the gaps seven ways made no difference, and real readings
  added nothing, so insulin and skinfold were dropped.
- **Source.** FINDINGS 5. Numbers: 374 women, 49%, 30%, 29% recoverable.

### 5. What it does: per 100 women like these, it catches 33 of the 35 who will be diagnosed
- **Visual.** The 100-women picture from the project page: 33 caught, 2 missed, 30 false alarms, 35 correctly cleared.
- **Say.** The price of catching almost everyone is 30 healthy women sent for a test they did not need. That trade is a choice,
  covered on slide 8.
- **Source.** FINDINGS 1 and 7. Numbers: 33, 2, 30, 35, AUC 0.85.

### 6. Blood sugar matters most; weight and age follow
- **Visual.** The "low to high" bars: glucose ×25.6, BMI ×6.2, age ×5.5, with a small inset of the age curve that rises then falls.
- **Say.** Going from a low to a high glucose reading multiplies the odds about 25 times. Risk rises with age until about 50, then
  falls in this data. These are patterns, not causes.
- **Source.** FINDINGS 4 and 11. Numbers as above.

### 7. Ten models were given a fair, tuned contest. A simple one won
- **Visual.** The contest chart (notebook 06), the published model highlighted, all others at or below it.
- **Say.** Neural networks, boosted trees and blends did not beat a straightforward model that lets three measurements curve. With
  768 women and six measurements, the limit is the information in the data.
- **Source.** FINDINGS 3. Numbers: 0.83 to 0.85, nine rivals.

### 8. Where to draw the line is a decision about costs, not a fact in the data
- **Visual.** The cut-off chart (notebook 07): caught and false alarms at 2×, 5× and 10×.
- **Say.** If a missed case is 5 times as bad as an unnecessary test, the cut-off is 17%. At 2 times it is 33%, at 10 times 9%. The
  page lets you move it.
- **Source.** FINDINGS 7. Numbers: 0.33, 0.17, 0.09; 28, 33, 34 caught.

### 9. It serves some women less well, and we say so
- **Visual.** AUC by age band as three bars (0.85, 0.80, 0.79) with the share diagnosed (21%, 49%, 50%) beneath.
- **Say.** Half of the women 45 and over have diabetes, so any shared cut-off flags most of them, and the model separates them less
  well. Women under 30 were losing out under the first cut-off; choosing it from costs fixed that.
- **Source.** FINDINGS 8. How solid: likely (133 women).

### 10. It also says how sure it is, including "too uncertain to call"
- **Visual.** The range bar on one woman, beside the three-zone chart (no / too uncertain / yes).
- **Say.** A risk of 36% comes from 768 women and could move about 19 points. A separate rule can decline to call: a third of women
  at the 90% level. A confident "yes" is still wrong 25% of the time.
- **Source.** FINDINGS 12. Numbers: 19 points, 91%, 33%, 25%.

### 11. On women outside the data, the ranking holds but the percentages do not
- **Visual.** The shift chart (notebook 10): predicted 25% against actual 3.9%.
- **Say.** I ran the finished model, unchanged, on 961 US women from a national survey. It ranks them as well as before. Its risk
  numbers are about six times too high, and the cut-off would flag nearly half of them. A notice on the page now warns when someone
  is unlike the women it learned from.
- **Source.** FINDINGS 10 and 13. How solid: likely for the ranking (only 37 cases), firm for the risks.

### 12. The verdict: a good ranking tool for this population, not a medical tool
- **Visual.** Three short columns: "Trust it for", "Do not trust it for", "What would change that".
- **Say.** Good at ranking within this kind of population, honest about its range. Not validated on other groups, not a diagnosis.
  The next step would be outcomes from another population.
- **Source.** [Model card](MODEL_CARD.md), "Intended use" and "Known limits".

## Appendix (in the downloads, not presented)

| # | Slide | Source |
|---|---|---|
| A1 | Why the model gave her this number: importance, SHAP check, the BMI plateau | FINDINGS 11 |
| A2 | Nothing hiding in the data: maps, clusters that do not hold steady, odd records kept | FINDINGS 13 |
| A3 | What did not work: rebalancing, interactions, tuned rivals, age-specific cut-offs | FINDINGS, "What didn't work" |
| A4 | How it is built to be checked: pipelines, 50 splits, tests, GitHub Actions, MLflow, model card | MILESTONES, M10 |
| A5 | Glossary of every term used | GLOSSARY |

## Formats and how they are built

- **HTML.** One self-contained page, 16:9, arrow keys and swipe to move, speaker notes on a toggle. Same look as the project page.
- **PNG.** One image per slide at 1920 by 1080, rendered from the HTML so the two always match.
- **PDF.** The PNGs assembled in order, with the appendix after the main slides.
- **PowerPoint (.pptx).** Built from the same slide definitions with the charts placed as images, speaker notes included, and text
  left editable.
- **Single source.** Every slide is defined once (headline, body, visual, notes, numbers), and all four formats are generated from
  that definition, so they cannot disagree.
- **Check at build time.** Each number on a slide is looked up in `reports/metrics.json` and the build fails on a mismatch. A test
  runs this, and the CI job runs the test.
- **Where it lives.** `deck/` in the project (definitions and build script) and `scott/diabetes-prediction/deck/` on the site, with
  download buttons for all four formats on the project page and a link on the model card.

## Decisions I need from you

1. **Length.** 12 main slides is my default. Fewer (about 8) is possible by folding slides 4 and 9 into the appendix.
2. **Slide 11 as the climax.** I put the outside-data result late and give it a full slide, because it is the most credible part of
   the project. Say if you would rather lead with the model and keep the limits to the closing slide.
3. **First person.** The notes are written as "I" (you presenting). Say if you prefer "we" or neutral wording.
4. **Branding.** I plan the project page's look and the schirko.com/scott backbar link on the HTML version, and no logo on the PPTX.
5. **Anything to add or cut.** For example, a slide on the glucose rule benchmark (21 more cases found), which I currently leave
   to the notes of slide 5.

## What happens next

1. You review this plan and answer the five questions above, or say "go with your defaults".
2. I build the deck and the build-time number check, look at every slide as an image, and fix layout problems.
3. Output goes to the project and the site, with the download buttons added to the project page, and you get the git commands as
   usual. Exit check: all four formats download, open cleanly, and every number matches FINDINGS.md.
