"""Feature building shared by training, batch prediction and the API.

Design rules (to avoid leakage):
  * only columns that exist at claim time are used (never inspector outcome / is_fraud);
  * partner fraud history only looks at claims submitted at least LAG_DAYS earlier,
    because investigations take time to close.
"""
import hashlib
import re
import numpy as np
import pandas as pd

LAG_DAYS = 45            # investigation lag assumed when building partner history
PRIOR_RATE = 0.012       # overall fraud base rate, used to shrink small samples
PRIOR_N = 20             # strength of the shrinkage (acts like 20 pseudo-claims)
POLICY_CHANGE = pd.Timestamp("2026-05-01")
NEW_PARTNER_CUTOFF = pd.Timestamp("2025-11-01")   # "newer partners" = onboarded in last ~year
TYPES = ["authorised_service_centre", "franchise", "freelance_technician"]
FAMILIES = ["Air Fryer", "Mixer Grinder", "Water Purifier", "Robot Vacuum",
            "Induction Cooktop", "Ceiling Fan", "Room Heater"]

FEATURES = [
    "amount", "log_amount", "ratio_to_list", "ratio_high", "small_claim",
    "days_since_purchase", "days_to_warranty_end", "prior_claims",
    "photo", "inspected", "note_missing", "post_policy_change", "small_uninspected",
    "partner_age_days", "partner_is_new",
    "p_n", "p_frauds", "p_rate",
    *[f"type_{t}" for t in TYPES], *[f"fam_{f}" for f in FAMILIES],
]


def partner_key(partner_id):
    """Pseudonymous key used in the shipped partner table (not a security boundary)."""
    return hashlib.sha256(("kestrel-v1|" + str(partner_id).strip().upper()).encode()).hexdigest()[:16]


def clean_serial(s):
    return re.sub(r"[^0-9]", "", str(s))


def load_raw(train_path, test_path, partners_path, products_path):
    tr = pd.read_csv(train_path, parse_dates=["submitted_at"])
    te = pd.read_csv(test_path, parse_dates=["submitted_at"])
    pa = pd.read_csv(partners_path, parse_dates=["onboarded_date"])
    pr = pd.read_csv(products_path)
    return tr, te, pa, pr


def dedupe_train(tr):
    """Partners re-submit bounced claims: keep the first submission of each claim number."""
    return tr.sort_values("submitted_at").drop_duplicates("claim_id", keep="first").reset_index(drop=True)


def partner_history(rows, labelled, lag_days=LAG_DAYS):
    """For each row: number of past labelled claims and frauds at that partner,
    counting only claims submitted >= lag_days before the row."""
    lab = labelled[["partner_id", "submitted_at", "is_fraud"]].sort_values("submitted_at")
    out_n = np.zeros(len(rows)); out_f = np.zeros(len(rows))
    groups = {k: g for k, g in lab.groupby("partner_id")}
    for i, (pid, ts) in enumerate(zip(rows["partner_id"].values, rows["submitted_at"].values)):
        g = groups.get(pid)
        if g is None:
            continue
        cutoff = ts - np.timedelta64(lag_days, "D")
        k = np.searchsorted(g["submitted_at"].values, cutoff, side="right")
        out_n[i] = k
        out_f[i] = g["is_fraud"].values[:k].sum()
    return out_n, out_f


def build(df, partners, products, hist_n, hist_f):
    x = df.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")
    f = pd.DataFrame(index=x.index)
    f["amount"] = x["claim_amount_inr"]
    f["log_amount"] = np.log1p(x["claim_amount_inr"])
    f["ratio_to_list"] = x["claim_amount_inr"] / x["list_price_inr"]
    f["ratio_high"] = ((f["ratio_to_list"] > 0.6) & (f["ratio_to_list"] <= 0.95)).astype(int)
    f["small_claim"] = (x["claim_amount_inr"] < 2000).astype(int)
    f["days_since_purchase"] = x["days_since_purchase"]
    f["days_to_warranty_end"] = x["warranty_months"] * 30.4 - x["days_since_purchase"]
    f["prior_claims"] = x["customer_prior_claims"]
    f["photo"] = (x["photo_attached"] == "Y").astype(int)
    f["inspected"] = (x["partner_inspected"] == "Y").astype(int)
    f["note_missing"] = x["inspector_note"].isna().astype(int)
    f["post_policy_change"] = (x["submitted_at"] >= POLICY_CHANGE).astype(int)
    f["small_uninspected"] = ((f["small_claim"] == 1) & (f["inspected"] == 0)).astype(int)
    f["partner_age_days"] = (x["submitted_at"] - x["onboarded_date"]).dt.days
    f["partner_is_new"] = (x["onboarded_date"] >= NEW_PARTNER_CUTOFF).astype(int)
    f["p_n"] = hist_n
    f["p_frauds"] = hist_f
    f["p_rate"] = (hist_f + PRIOR_RATE * PRIOR_N) / (hist_n + PRIOR_N)
    for t in TYPES:
        f[f"type_{t}"] = (x["partner_type"] == t).astype(int)
    for fam in FAMILIES:
        f[f"fam_{fam}"] = (x["family"] == fam).astype(int)
    return f[FEATURES]
