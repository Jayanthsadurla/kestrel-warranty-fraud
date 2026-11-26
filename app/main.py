"""Kestrel warranty-claim risk service.   Start:  uvicorn app.main:app --port 8000
No paid API, no network calls: a local scikit-learn model plus plain-English reasons."""
import json, os, sys
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import build, partner_key, POLICY_CHANGE  # noqa: E402

ART = os.path.join(ROOT, "artifacts")
try:
    BUNDLE = joblib.load(os.path.join(ART, "model.joblib"))
    MODEL, THRESHOLD = BUNDLE["model"], BUNDLE["threshold"]
    GOODWILL, REVIEW = BUNDLE["goodwill"], BUNDLE["review_cost"]
    PRODUCTS = {p["sku"]: p for p in json.load(open(os.path.join(ART, "products.json")))}
    LOAD_ERROR = None
except Exception as e:                                                     # fail politely, never crash on start
    MODEL = None; LOAD_ERROR = str(e); PRODUCTS = {}; THRESHOLD = 1.0; GOODWILL = 380; REVIEW = 260
try:                                                                       # optional: without it we score with no partner history
    PARTNERS = json.load(open(os.path.join(ART, "partner_stats.json")))
except Exception:
    PARTNERS = {}

app = FastAPI(title="Kestrel claim risk", version="1.0")


@app.exception_handler(Exception)
async def polite_errors(request, exc):
    return JSONResponse(status_code=500, content={"error": "Something went wrong scoring this claim. Please check the fields and try again.",
                                                  "detail": type(exc).__name__})


class Claim(BaseModel):
    partner_id: str = Field(..., examples=["SP3004"])
    sku: str = Field(..., examples=["KH-AF-02"])
    claim_amount_inr: float = Field(..., ge=0)
    days_since_purchase: int = Field(..., ge=0)
    partner_inspected: str = Field("N", description="Y or N")
    photo_attached: str = Field("Y", description="Y or N")
    customer_prior_claims: int = Field(0, ge=0)
    submitted_at: Optional[str] = Field(None, description="e.g. 2026-07-14 10:30. Defaults to now.")
    claim_id: Optional[str] = None
    product_serial: Optional[str] = None
    claim_description: Optional[str] = None
    inspector_note: Optional[str] = None


def reasons_for(f, row, amount, p_frauds, p_n, known):
    """Plain-English reasons. Rule-based hints that mirror what the model weighs; not exact model internals."""
    up, down = [], []
    age = int(f["partner_age_days"])
    if f["partner_is_new"] and known:
        up.append(f"Partner joined only {age} days ago. Among partners onboarded in the last year, about 7 in 100 past claims were fraud, against 1 in 100 for established partners.")
    elif known:
        down.append(f"Established partner (joined {age // 365} years ago).")
    if f["small_uninspected"]:
        up.append("Claim is under Rs 2,000 and was approved without inspection. Since the 1 May 2026 rule change most fraud sits in this group.")
    elif f["inspected"]:
        down.append("An inspector signed this claim off.")
    if not known:
        up.append("This partner is not in our partner list, so no history is available.")
    elif p_frauds > 0:
        up.append(f"{p_frauds} of this partner's {p_n} earlier decided claims were confirmed fraud.")
    elif p_n >= 10:
        down.append(f"None of this partner's {p_n} earlier decided claims were fraud.")
    elif p_n == 0:
        up.append("No investigation history for this partner yet, so we cannot vouch for them.") if f["partner_is_new"] else None
    r = f["ratio_to_list"]
    if f["ratio_high"]:
        up.append(f"Claim is {r:.0%} of the product's list price. Claims between 60% and 95% of price were fraud about 9 times in 100 in past data, against under 1 in 100 for other claims.")
    if f["prior_claims"] >= 2:
        up.append(f"Customer already has {int(f['prior_claims'])} earlier warranty claims.")
    if f["days_to_warranty_end"] < 15:
        up.append("Claim is filed within about two weeks of warranty expiry.")
    return up, down


@app.get("/health")
def health():
    return {"status": "ok" if MODEL is not None else "degraded", "model_loaded": MODEL is not None,
            "partner_history_available": bool(PARTNERS), "detail": LOAD_ERROR}


@app.post("/predict")
def predict(c: Claim):
    if MODEL is None:
        return JSONResponse(status_code=503, content={"error": "Model files are missing. Run 'python src/train.py' or restore the artifacts folder.",
                                                      "detail": LOAD_ERROR})
    sku = c.sku.strip().upper()
    if sku not in PRODUCTS:
        raise HTTPException(422, f"Unknown product code '{c.sku}'. Known codes look like KH-AF-01 (see products list).")
    try:
        ts = pd.Timestamp(c.submitted_at) if c.submitted_at else pd.Timestamp(datetime.now())
    except Exception:
        raise HTTPException(422, "submitted_at must look like 2026-07-14 10:30")
    pk = partner_key(c.partner_id)
    pinfo = PARTNERS.get(pk)
    known = pinfo is not None
    if known:
        part = pd.DataFrame([dict(partner_id=c.partner_id, city="", onboarded_date=pd.Timestamp(pinfo["onboarded"]), partner_type=pinfo["type"])])
        n, fr = pinfo["n"], pinfo["frauds"]
    else:  # unknown partner: treat as brand new with no history rather than failing
        part = pd.DataFrame([dict(partner_id=c.partner_id, city="", onboarded_date=ts, partner_type="franchise")])
        n, fr = 0, 0
    row = pd.DataFrame([dict(partner_id=c.partner_id, sku=sku, submitted_at=ts,
                             days_since_purchase=c.days_since_purchase, claim_amount_inr=c.claim_amount_inr,
                             photo_attached=c.photo_attached.strip().upper(), partner_inspected=c.partner_inspected.strip().upper(),
                             inspector_note=c.inspector_note or None, customer_prior_claims=c.customer_prior_claims)])
    X = build(row, part, pd.DataFrame(list(PRODUCTS.values())), np.array([float(n)]), np.array([float(fr)]))
    score = float(MODEL.predict_proba(X)[:, 1][0])
    f = X.iloc[0]
    up, down = reasons_for(f, row.iloc[0], c.claim_amount_inr, fr, n, known)
    flag = score >= THRESHOLD
    ev_hold = score * c.claim_amount_inr - (1 - score) * GOODWILL - REVIEW
    band = "high" if flag else ("medium" if score >= THRESHOLD / 3 else "low")
    return {
        "claim_id": c.claim_id, "score": round(score, 4), "flag_for_review": bool(flag), "risk": band,
        "action": ("Hold for the investigation desk (it can take about 40 claims a month; this claim is among the riskiest)." if flag
                   else "Pay in the normal way."),
        "why_risky": up, "why_reassuring": down,
        "money": {"claim_amount_inr": c.claim_amount_inr, "expected_fraud_loss_if_paid_inr": round(score * c.claim_amount_inr),
                  "cost_if_held_and_genuine_inr": GOODWILL, "net_value_of_reviewing_inr": round(ev_hold)},
        "note": "Decision support only. A person should decide; a flag is not proof of fraud.",
        "threshold": round(THRESHOLD, 4),
    }


@app.get("/")
def screen():
    return FileResponse(os.path.join(ROOT, "app", "index.html"))
