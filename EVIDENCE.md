# Evidence: does it work, and how often does it not

## Test design
- Data: 12,029 rows. After removing 681 re-submitted duplicates and 202 undecided cases: **11,146 decided claims, 141 fraud (1.3%)**.
- Honest check: train on everything **before 1 June 2026**, score **June 2026** (713 claims, 22 fraud). Random splits would flatter the model, because the claims that matter are the newest ones.
- With only 22 frauds the numbers are rough. Ranges below are 90% bootstrap ranges.

## Results on June 2026 (never seen in training)
| Measure | Result |
|---|---|
| ROC-AUC | 0.91 |
| PR-AUC (1.0 is perfect; flagging at random gives 0.03) | 0.39 (range 0.25 to 0.56) |
| Desk capacity: 38 claims in a 713-claim month (40 per 750) | caught **15 of 22 frauds** (68%, range 48% to 82%); 23 genuine claims held |
| Precision in that top 38 | 39% |
| Fraud money stopped | Rs 20,920 of Rs 45,673 that month |
| Cost of holding 23 genuine claims | Rs 8,740 (23 x Rs 380 goodwill) |
| Review cost, assumed Rs 260 per claim reviewed | Rs 9,880 (38 x Rs 260) |
| **Net** | **about +Rs 2,300** (+Rs 12,200 if review cost is ignored) |

Fraud stopped per claim checked (Farhan's question): Rs 20,920 / 38 = **Rs 550 per claim checked**, against Rs 380 goodwill lost on each wrongly held genuine claim.

## Why accuracy is the wrong KPI
- Flag nothing: **96.9%** accuracy in June (98.7% across all months). The board's 97% target is met by a model that does nothing.
- Our model flagging 38 claims a month: 95.8% accuracy, which looks "worse" while stopping Rs 20,920.
- The best accuracy we could reach was 97.2%, by flagging only 4 claims.

## What we found in the data
1. **The 1 May change reshaped fraud.** Before May, fraud was large, inspected claims (average Rs 5,433). After May it is small, uninspected claims (average Rs 1,865). A model trained only on pre-May data scores **ROC-AUC 0.22 on May and June, worse than random**: it learned the old pattern backwards. This is why the final model up-weights post-May claims.
2. **Ritu's view is half right.** Partners onboarded in the last year: 7.1% fraud, against 1.0% for others. But only **8 of 46** new partners with claims have any fraud. After 1 May, among new partners' claims under Rs 2,000, **34 of 161 were fraud (21%)**; established partners ran at 0.2%. Meenal is right too: most new partners are fine.
3. Fraud is clustered: 51 partners have any fraud, and the top 10 hold 63% of it.
4. Weak or no signal: product family, claim description, inspector note text, serial number reuse, days since purchase. We left text out.
5. Customer's earlier claims matter (0.8% fraud with none, 5% with two, 21% with three or more). Claims at 60% to 95% of list price: 8.7% fraud against under 1%.

## Where it gets it wrong
June frauds missed (7 of 22): 4 were small uninspected claims at two partners that joined in June (no history to learn from); 1 was a Rs 959 claim at a partner that joined in December 2025; 2 were at established partners (a Rs 686 uninspected claim and a Rs 17,866 inspected claim). In short: **brand-new partners with no history, and fraud from partners that had been clean.**
Wrong the other way: 23 of 38 flags in June were genuine claims, mostly new partners' small claims.

## Simple rule for comparison
"New partner AND under Rs 2,000 AND uninspected" flags 86 June claims and catches 20 of 22 frauds, but 86 is more than double the desk's capacity. The model's gain is in the ordering, so the best 38 come first.

## What is wrong with this (bugs, shortcuts, things we know are off)
- **Few frauds to learn from:** 36 since the rule change. Results could move a lot on a different month.
- **Test scores are an estimate:** we expect lower on July to September (see `PREDICTION.md`).
- **Batch vs service differ slightly:** `predictions.csv` uses partner history looking back with a 45-day lag, exactly as validated. The service uses a frozen snapshot of each partner's totals. On all 2,252 test claims the two agree closely (rank correlation 0.997; 110 of the top 120 the same), but not exactly.
- **Rs 260 is an assumption:** policy says "blended cost of a service contact"; we treated it as the cost of each review. If it only applies to genuine claims held, net improves.
- **Undecided cases:** 202 dropped. Legacy Zoho stored undecided as 0, so some old "not fraud" labels may really be undecided. We could not tell which.
- **"Newer partner" = joined since 1 Nov 2025:** our choice, and it is arbitrary.
- **Reasons are rule-based hints,** written to match what the model weighs. They are not exact model internals.
- **Partner table in the repo uses hashed IDs.** There are only about 380 partners, so the hashing is easy to reverse. Treat it as weak. Raw claims are not in the repo (policy section 10).
- **Timestamp trap (policy section 9):** resolution events from legacy Zoho are in UTC. Our data has no resolution events, so nothing needed fixing, but nobody should add them without converting.
- **Threshold:** set for about 40 flags per 750 claims. If volume changes, re-set it.
- **No monitoring or retraining job.** Fraud changed shape once already; it will again.
