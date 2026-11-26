# Kestrel Home: warranty claim risk check

Reads one warranty claim and says: hold it for the investigation desk, or pay it. Gives plain-English reasons.
No paid API, no internet needed at run time. It is a small scikit-learn model that runs on a laptop.

## Start it (clean machine, Python 3.10+)

```bash
git clone <this repo> && cd <this repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 for the screen. Two example buttons fill the form for you.
API (what the screen calls):

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{
  "partner_id":"SP3287","sku":"KH-IC-02","claim_amount_inr":1450,"days_since_purchase":200,
  "partner_inspected":"N","photo_attached":"N","customer_prior_claims":2,"submitted_at":"2026-08-20 15:10"}'
```

Returns `score` (chance of fraud), `flag_for_review`, `why_risky`, `why_reassuring`, and the rupee view.
`GET /health` says whether the model loaded. If model files are missing it answers politely (HTTP 503) instead of crashing.

## What is in here

| File | What |
|---|---|
| `predictions.csv` | Scores for all 2,252 test claims (higher = more likely fraud) |
| `MEMO.md` | One-page memo to Ritu |
| `EVIDENCE.md` | How we know it works, and how often it does not |
| `PREDICTION.md` | What we expected to score, written before submitting |
| `src/features.py`, `src/train.py` | Features and training. `python src/train.py` retrains and rewrites `predictions.csv` (needs the five client CSVs in `data/`) |
| `app/` | The service and the screen |
| `artifacts/` | Trained model, pseudonymised partner table, product list |

## How it works in four lines

1. Duplicates removed (partners re-submit bounced claims); 202 cases still under investigation are left out of training, not counted as "not fraud".
2. Features: claim size vs list price, small and uninspected, partner age, the partner's earlier confirmed-fraud rate (looking back only, with a 45-day investigation lag), customer's earlier claims, and so on. Nothing from the investigation outcome is used.
3. Gradient boosting (scikit-learn). Claims after the 1 May 2026 rule change count 10 times as much, because fraud changed shape on that day (see `EVIDENCE.md`).
4. Flag threshold is set so about 40 claims a month are flagged, which is what the investigation desk can handle.

## Things to know before trusting it

See "What is wrong with this" in `EVIDENCE.md`. The short version: only 36 frauds exist since the rule change, partner history in the API is a frozen snapshot, and the reasons are rule-based hints, not exact model internals.

## Data handling

Raw claims, partner and customer data are **not** in this repository (Kestrel policy section 10). Partner facts needed by the service are shipped under hashed keys; that is weak protection (see `EVIDENCE.md`). To drop it, delete `artifacts/partner_stats.json`; the service still starts but scores without partner history and is less accurate.
