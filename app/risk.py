from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CAPACITY_PER_MONTH = 40
FEATURE_NAMES = (
    "duplicate_invoice",
    "serial_mismatch",
    "purchase_mismatch",
    "missing_documents",
    "prior_claims",
    "high_claim_amount",
)


def _training_examples() -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for flags in product((0, 1), repeat=5):
        duplicate, serial_mismatch, purchase_mismatch, missing_docs, high_amount = flags
        for prior_claims in (0, 1, 3, 6, 9, 12):
            features = [
                duplicate,
                serial_mismatch,
                purchase_mismatch,
                missing_docs,
                prior_claims,
                high_amount,
            ]
            rule_score = (
                3.2 * duplicate
                + 2.4 * serial_mismatch
                + 1.8 * purchase_mismatch
                + 1.1 * missing_docs
                + 0.35 * prior_claims
                + 0.6 * high_amount
            )
            rows.append(features)
            labels.append(int(rule_score >= 3.0))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def _train_model() -> Any:
    features, labels = _training_examples()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1_000, random_state=17),
    )
    return model.fit(features, labels)


MODEL = _train_model()


def _features(claim: dict[str, Any]) -> list[float]:
    return [
        float(claim["duplicate_invoice"]),
        float(not claim["serial_matches_registration"]),
        float(not claim["invoice_matches_purchase"]),
        float(not claim["has_required_documents"]),
        float(min(claim["prior_claims"], 12)),
        float(claim["claim_amount"] > 5_000),
    ]


def _reasons(claim: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if claim["duplicate_invoice"]:
        reasons.append("The invoice appears more than once in the submitted claim.")
    if not claim["serial_matches_registration"]:
        reasons.append("The product serial number does not match the registration.")
    if not claim["invoice_matches_purchase"]:
        reasons.append("The invoice details do not match the recorded purchase.")
    if not claim["has_required_documents"]:
        reasons.append("One or more required supporting documents are missing.")
    if claim["prior_claims"] >= 3:
        reasons.append(
            f"This customer has {claim['prior_claims']} prior warranty claims."
        )
    if claim["claim_amount"] > 5_000:
        reasons.append("The requested amount is above the $5,000 review threshold.")
    if not reasons:
        reasons.append("No configured review indicators were found in the submitted fields.")
    return reasons


def predict_claim(claim: dict[str, Any]) -> dict[str, Any]:
    features = _features(claim)
    score = float(MODEL.predict_proba([features])[0, 1])
    risk_band = "high" if score >= 0.70 else "medium" if score >= 0.35 else "low"
    return {
        "claim_id": claim["claim_id"],
        "risk_score": round(score, 4),
        "risk_band": risk_band,
        "reasons": _reasons(claim),
    }


def rank_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = [predict_claim(claim) for claim in claims]
    return sorted(results, key=lambda item: (-item["risk_score"], item["claim_id"]))
