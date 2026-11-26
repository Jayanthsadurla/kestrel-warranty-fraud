# Kestrel Home warranty claim risk check — model memo

## Purpose and intended use

This service sorts submitted warranty claims to help an investigation desk prioritize up to 40 claims per month. It is a human-review aid only: a score is not a determination that a customer committed fraud, and must not by itself be used to deny or delay a claim.

## Inputs and behavior

The API accepts claim amount, product age, prior claim count, invoice-to-purchase match, serial-to-registration match, duplicate-invoice flag, and required-document completeness. `/predict` returns a 0–1 score, a broad risk band, and plain-English review indicators. `/rank` orders a batch by score and flags no more than 40 for the monthly investigation capacity. Ties are ordered by claim ID.

The model uses a scikit-learn standardized logistic regression. Its reproducible training examples are generated deterministically from synthetic combinations of the same claim-document indicators; labels are generated from a transparent review heuristic. There is no real customer data or independently measured fraud outcome in this repository. The score therefore demonstrates the service workflow, not validated predictive performance or a production-ready fraud model.

## Review indicators

Duplicate invoices and mismatches are stronger synthetic training signals; missing documentation, repeated prior claims, and requests above $5,000 are weaker signals. The API returns the underlying indicators as explanations. Product age is accepted as context but does not affect the current model score. A human investigator should verify the source records, consider reasonable explanations, and follow Kestrel Home's existing claims process.

## Limitations and safeguards

The model has not been validated against historical outcomes, calibrated, or assessed for disparate impact. The synthetic training heuristic can encode assumptions rather than actual fraud patterns. Do not use the example scores as operational thresholds. Before real deployment, replace the synthetic training process with appropriately governed and representative labeled outcomes, validate calibration and fairness, monitor drift and errors, and establish an appeal and human-review process. The service intentionally excludes sensitive personal attributes.
