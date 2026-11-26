from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.risk import CAPACITY_PER_MONTH, predict_claim, rank_claims

app = FastAPI(
    title="Kestrel Home Warranty Claim Risk Check",
    description=(
        "A triage aid that ranks warranty claims for human investigation. "
        "Risk scores are not findings of fraud."
    ),
    version="1.0.0",
)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=80)
    claim_amount: float = Field(gt=0, le=25_000)
    product_age_months: int = Field(ge=0, le=240)
    prior_claims: int = Field(ge=0, le=20)
    invoice_matches_purchase: bool
    serial_matches_registration: bool
    duplicate_invoice: bool
    has_required_documents: bool


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: Annotated[list[Claim], Field(min_length=1, max_length=1_000)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(claim: Claim) -> dict[str, object]:
    return predict_claim(claim.model_dump())


@app.post("/rank")
def rank(request: RankRequest) -> dict[str, object]:
    claim_ids = [claim.claim_id for claim in request.claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise HTTPException(status_code=422, detail="claim_id values must be unique")

    ranked = rank_claims([claim.model_dump() for claim in request.claims])
    for index, result in enumerate(ranked):
        result["investigate_this_month"] = index < CAPACITY_PER_MONTH

    return {
        "monthly_investigation_capacity": CAPACITY_PER_MONTH,
        "claim_count": len(ranked),
        "investigate_count": min(len(ranked), CAPACITY_PER_MONTH),
        "ranked_claims": ranked,
    }
