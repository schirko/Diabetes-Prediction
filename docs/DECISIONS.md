# Decisions

Choices that shaped the project and the reason for each. Newest at the bottom.

| ID | Decision | Why |
|---|---|---|
| D01 | Logistic regression is the Milestone 1 model. | It ties the random forest (AUC 0.835 vs 0.838, a gap one tenth the size of the split-to-split wobble), is well calibrated, and can run in a web page. Revisited in M4. |
| D02 | The app runs in the browser, not on a server. | Nothing to host or keep alive, and it fits the static portfolio. A server version is optional (E28). |
| D03 | Screening cut-off set where at least 85% of true cases are caught (0.24). | A missed case costs more than a follow-up blood test. The 85% figure is a judgment call, to be replaced by a cost-based choice in M5 (E16). |
| D04 | All eight inputs are kept. | Four inputs score the same (0.833 vs 0.835), but keeping eight lets the app show every contribution and leaves M2 and M3 something to test. |
| D05 | Notebooks are the source of truth and are re-run with `python -m src.run_notebooks`. | Saved outputs can't drift from the code. Shared logic lives in `src/` so notebooks stay short. |
| D06 | Notebook pages for the site are rendered by a small custom script, not nbconvert's default template. | Pages come out at a fraction of the size and carry the portfolio's back bar and no-index tag. |
| D07 | Accuracy is never reported alone. | Always answering "no diabetes" scores 65% while catching nobody. |
| D08 | The page states plainly that this is not a medical tool. | 768 women from one community in the 1980s. It shows method, not a diagnosis. |
