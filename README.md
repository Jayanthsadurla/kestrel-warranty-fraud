# Kestrel Home Warranty Claim Risk Check

A FastAPI and scikit-learn demo that ranks warranty claims for a desk with capacity to investigate 40 claims per month. Each prediction includes plain-English review indicators; scores prioritize human review and are not findings of fraud.

## Run

```sh
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. `POST /predict` scores one claim and `POST /rank` sorts a batch, flagging at most 40 for investigation.

Claim payloads require `claim_id`, `claim_amount`, `product_age_months`, `prior_claims`, `invoice_matches_purchase`, `serial_matches_registration`, `duplicate_invoice`, and `has_required_documents`.

## Test

```sh
python -m unittest discover -s tests -v
```

See [the model memo](docs/risk-model-memo.md) for intended use, synthetic training data limitations, and review safeguards. Sample prediction evidence is in `evidence/sample-predictions.json`.