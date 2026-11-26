# Submission form: draft answers (edit anything that is not literally true for you)

**What did you build, and what business outcome does it move? State the number and the money.**
A claim-risk service (one API endpoint plus one screen) that ranks warranty claims so the 40-a-month investigation desk checks the riskiest first, with plain-English reasons. On June 2026, never seen in training: it found 15 of 22 frauds in the 38 claims a 40-a-month desk could check (68%), stopping Rs 20,920 of Rs 45,673 fraud, i.e. Rs 550 per claim checked. After Rs 8,740 goodwill for 23 wrongly held genuine claims and an assumed Rs 9,880 review cost, net is about +Rs 2,300 a month (about +Rs 12,200 before review cost). Honest message: the tool helps but is small; the 1 May no-inspection rule for claims under Rs 2,000 is the bigger lever.

**What does one run cost, and what would a month cost? Show the arithmetic.**
I used no paid calls. One prediction runs a local scikit-learn model in a few milliseconds on a laptop CPU: Rs 0. Kestrel's volume is about 750 claims a month: 750 x Rs 0 = Rs 0 a month (the brief asks this twice; the first copy mentions "Vireo, 650 tickets a week", which looks pasted from another brief. At that volume it is also Rs 0: 650 x 52 / 12 = 2,817 tickets a month x Rs 0.) Hosting is the only cost, and it fits on the smallest server.

**How do you know it works?**
Trained on claims before 1 June 2026 (after removing 681 duplicates and 202 undecided), tested on June 2026: 713 claims, 22 fraud. ROC-AUC 0.91, PR-AUC 0.39 (90% range 0.25 to 0.56; random is 0.03). Top-38 recall 68% (range 48% to 82%), precision 39%. Error rate: 7 of 22 frauds missed, 23 of 38 flags were genuine claims. It gets wrong: brand-new partners with no history, and partners that had been clean. Small sample, so the ranges are wide. Also checked that the live endpoint agrees with predictions.csv on all 2,252 test claims (rank correlation 0.997).

**Did you change, narrow, or push back on the client's ask?**
Yes. (1) Pushed back on 97% accuracy: only about 1 in 30 recent claims is fraud, so flagging nothing scores 96.9% in June and our model scores 95.8% while stopping fraud. I replaced it with "fraud stopped per claim checked", which is what Farhan asked for. (2) Narrowed "flag fraud before we pay" to "rank claims for a 40-a-month desk", never auto-reject. (3) Tested "newer partners are the problem": half true (7% vs 1%, but only 8 of 46 new partners have any fraud), consistent with Meenal's warning.

**What is wrong with what you are handing us?**
See EVIDENCE.md "What is wrong". Main points: only 36 frauds since the 1 May rule change; I expect the July to September score to be lower than June; the service uses a frozen partner snapshot so it differs slightly from predictions.csv (110 of the top 120 the same); Rs 260 per review is my assumption; undecided cases in old Zoho data came across as 0 so some negatives may be wrong; reasons are rule-based hints; hashed partner IDs are weak protection; no monitoring or retraining.

**What did you deliberately leave out, and why that rather than something else?**
Text analysis of the claim description and inspector note (no signal on a first look, and it would add a dependency); a paid-API explanation layer (the brief wants it to start with no key, and the rule-based reasons were enough); a database and logins (a small thing that runs beats a large one); serial-number cleaning (serial reuse showed no link to fraud).

**Anything you built or found that nobody asked for?**
Found that a model trained on pre-May data scores worse than random (ROC-AUC 0.22) on May and June, because the rule change flipped the fraud pattern. Found that the "97% accuracy" target is met by doing nothing. Found fraud is clustered: top 10 partners hold 63% of it.

**What did you use AI for?** (EDIT to match what you really did)
Claude (Sonnet 5.5, free plan, Rs 0) for exploring the data, writing the training code, the API and screen, and drafting the memo and README. It helped most with finding the May regime change and the leakage traps quickly. Things I checked or discarded: [write what you changed or rejected]. Screen recording: [Drive link].

**Your Public Google Drive Link:** [paste the link to the screen recording]

**Someone picks this up on Monday and you are unreachable. The three things they need to know.**
1. Fraud changed shape on 1 May 2026 (small uninspected claims at new partners); retrain monthly with new outcomes, and keep the post-May weight (WEIGHT_NEW_REGIME in src/train.py).
2. Raw client data is git-ignored on purpose (policy section 10): put the five CSVs in data/ and run `python src/train.py` to retrain; the service itself needs only artifacts/.
3. The flag threshold is set for about 40 flags per 750 claims; if volume or desk capacity changes, re-set it. A flag means "look first", never "refuse".

**Honest hours spent. One number.** [your honest number]

**Github Repo Link:** [your public repo URL]
