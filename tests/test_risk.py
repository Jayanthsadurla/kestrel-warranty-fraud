import unittest

from app.main import Claim, rank
from app.risk import predict_claim, rank_claims


def sample_claim(claim_id: str, **overrides: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim_id": claim_id,
        "claim_amount": 850.0,
        "product_age_months": 18,
        "prior_claims": 0,
        "invoice_matches_purchase": True,
        "serial_matches_registration": True,
        "duplicate_invoice": False,
        "has_required_documents": True,
    }
    claim.update(overrides)
    return claim


class RiskModelTests(unittest.TestCase):
    def test_prediction_includes_score_band_and_plain_english_reasons(self) -> None:
        result = predict_claim(
            sample_claim(
                "K-102",
                duplicate_invoice=True,
                serial_matches_registration=False,
                has_required_documents=False,
            )
        )

        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"], 1)
        self.assertEqual(result["risk_band"], "high")
        self.assertTrue(any("invoice appears more than once" in reason for reason in result["reasons"]))
        self.assertTrue(any("serial number does not match" in reason for reason in result["reasons"]))

    def test_rank_orders_high_risk_first_and_breaks_ties_by_claim_id(self) -> None:
        ranked = rank_claims(
            [
                sample_claim("Z-low"),
                sample_claim("B-high", duplicate_invoice=True),
                sample_claim("A-high", duplicate_invoice=True),
            ]
        )

        self.assertEqual([claim["claim_id"] for claim in ranked], ["A-high", "B-high", "Z-low"])

    def test_rank_endpoint_marks_only_first_forty_for_investigation(self) -> None:
        claims = [
            Claim(**sample_claim(f"K-{index:02}"))
            for index in range(42)
        ]

        result = rank(
            type("Request", (), {"claims": claims})()
        )

        self.assertEqual(result["monthly_investigation_capacity"], 40)
        self.assertEqual(result["investigate_count"], 40)
        flagged = [claim for claim in result["ranked_claims"] if claim["investigate_this_month"]]
        self.assertEqual(len(flagged), 40)
        self.assertFalse(result["ranked_claims"][-1]["investigate_this_month"])


if __name__ == "__main__":
    unittest.main()
