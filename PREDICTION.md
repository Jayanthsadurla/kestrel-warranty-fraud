# What I expect to score, written before submitting

Measured on June 2026 (model trained only on earlier claims): ROC-AUC 0.91, PR-AUC 0.39 (90% range 0.25 to 0.56), 15 of 22 frauds inside the 38 claims a 40-a-month desk could check.

For the hidden July to September claims I expect **lower**:
- ROC-AUC about 0.85 to 0.90; PR-AUC about 0.25 to 0.40.
- Why lower: partner history is up to three months stale; partners who joined in June to August (126 test claims, 5.6%) have little or no history; the final model has seen one extra month of the new pattern but fraudsters adapt; and only 22 frauds in June means my own estimate is noisy.
- If scored on top-40-per-month: I expect to catch roughly 55 to 70 percent of the frauds a perfect desk could catch, at about 35 to 40 percent precision.
- If scored on plain accuracy at 0.5: I expect about 97%, but a model that flags nothing scores about the same. That number says very little.
