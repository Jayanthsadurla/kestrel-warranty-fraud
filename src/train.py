"""Train the Kestrel warranty-fraud model, validate it on the newest month, write predictions.csv.

Run from the repo root:  python src/train.py
"""
import json, os, sys
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from features import *
from features import partner_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda f: os.path.join(ROOT, "data", f)
A = lambda f: os.path.join(ROOT, "artifacts", f)

WEIGHT_NEW_REGIME = 10      # claims after the 1 May 2026 policy change count 10x (see EVIDENCE.md)
CAPACITY_PER_MONTH = 40     # investigation desk limit (policy sec. 5)
GOODWILL = 380              # cost of holding a genuine claim (policy sec. 4)
REVIEW_COST = 260           # blended cost of a service contact (policy sec. 4) - assumed per review
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
              l2_regularization=1.0, random_state=0)


def fit(X, y, d):
    w = np.where(d >= POLICY_CHANGE, WEIGHT_NEW_REGIME, 1.0)
    m = HistGradientBoostingClassifier(**PARAMS)
    m.fit(X, y, sample_weight=w)
    return m


def main():
    tr, te, pa, pr = load_raw(D("train.csv"), D("test_unlabelled.csv"), D("partners.csv"), D("products.csv"))
    raw_n = len(tr)
    tr = dedupe_train(tr)
    n_dups = raw_n - len(tr)
    undecided = int(tr.is_fraud.isna().sum())
    lab = tr[tr.is_fraud.notna()].reset_index(drop=True)      # undecided cases are NOT treated as "not fraud"
    lab["is_fraud"] = lab["is_fraud"].astype(int)
    hn, hf = partner_history(lab, lab)
    X = build(lab, pa, pr, hn, hf); y = lab["is_fraud"].values; d = lab["submitted_at"]
    amt = lab["claim_amount_inr"].values

    # ---------- honest validation: train before June, test on June 2026 -------------
    tj = (d < "2026-06-01").values; jun = ~tj
    mv = fit(X[tj], y[tj], d[tj].values)
    p = mv.predict_proba(X[jun])[:, 1]; yv = y[jun]; av = amt[jun]
    k = int(round(CAPACITY_PER_MONTH * jun.sum() / 750))          # capacity scaled to the claims in June
    order = np.argsort(-p)
    top = order[:k]
    rng = np.random.default_rng(0); aps = []; hits = []
    for _ in range(1000):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].sum():
            aps.append(average_precision_score(yv[i], p[i]))
            o = np.argsort(-p[i])[:k]; hits.append(yv[i][o].sum() / max(yv[i].sum(), 1))
    thr = float(np.sort(p)[::-1][k - 1])                          # score of the k-th highest claim
    flag = p >= thr
    acc = float(((flag.astype(int)) == yv).mean())
    base_acc = float((yv == 0).mean())
    # accuracy-maximising threshold (to show what the "97%" KPI would reward)
    best = max(((float(((p >= t).astype(int) == yv).mean()), t) for t in np.unique(p)), key=lambda z: z[0])
    caught = (yv[top] * av[top]).sum(); ok_held = int((yv[top] == 0).sum())
    ev = dict(
        validation_window="June 2026 (model trained on claims before 1 June 2026)",
        n_claims=int(jun.sum()), n_fraud=int(yv.sum()), fraud_rate=float(yv.mean()),
        pr_auc=float(average_precision_score(yv, p)), pr_auc_ci90=[float(np.percentile(aps, 5)), float(np.percentile(aps, 95))],
        roc_auc=float(roc_auc_score(yv, p)),
        capacity_k=k, caught_in_top_k=int(yv[top].sum()), recall_at_k=float(yv[top].sum() / yv.sum()),
        recall_at_k_ci90=[float(np.percentile(hits, 5)), float(np.percentile(hits, 95))],
        precision_at_k=float(yv[top].mean()),
        fraud_rupees_stopped=float(caught), fraud_rupees_total=float((yv * av).sum()),
        genuine_held=ok_held, goodwill_cost=ok_held * GOODWILL, review_cost=k * REVIEW_COST,
        net_rupees=float(caught - ok_held * GOODWILL - k * REVIEW_COST),
        net_rupees_before_review_cost=float(caught - ok_held * GOODWILL),
        accuracy_at_capacity_threshold=acc, accuracy_flag_nothing=base_acc,
        best_accuracy=best[0], best_accuracy_threshold=float(best[1]),
        flagged_at_best_accuracy=int((p >= best[1]).sum()), threshold=thr,
        missed=[], )
    miss = lab[jun].iloc[[i for i in range(len(yv)) if yv[i] == 1 and not flag[i]]]
    ev["missed"] = miss.merge(pa, on="partner_id")[["claim_id", "partner_id", "onboarded_date", "claim_amount_inr", "partner_inspected"]] \
        .astype(str).to_dict("records")
    # old regime sanity check: model trained only before May fails on May-Jun
    pre = (d < POLICY_CHANGE).values; mj = ((d >= POLICY_CHANGE) & (d < "2026-07-01")).values
    m0 = HistGradientBoostingClassifier(**PARAMS).fit(X[pre], y[pre])
    ev["old_regime_model_on_mayjun_roc_auc"] = float(roc_auc_score(y[mj], m0.predict_proba(X[mj])[:, 1]))
    # simple rule baseline on June
    rule = ((X["partner_is_new"] == 1) & (X["small_uninspected"] == 1)).values[jun]
    ev["rule_new_partner_small_uninspected"] = dict(flagged=int(rule.sum()), caught=int(yv[rule].sum()))
    rule2 = (X["small_uninspected"] == 1).values[jun]
    ev["rule_small_uninspected"] = dict(flagged=int(rule2.sum()), caught=int(yv[rule2].sum()))

    # ---------- partner / new-partner facts for the memo --------------------------------
    x = lab.merge(pa, on="partner_id")
    x["new"] = x["onboarded_date"] >= NEW_PARTNER_CUTOFF; x["post"] = x["submitted_at"] >= POLICY_CHANGE
    newp = x[x.new].groupby("partner_id").is_fraud.agg(["size", "sum"])
    fp = x[x.is_fraud == 1].groupby("partner_id").size().sort_values(ascending=False)
    ev["facts"] = dict(
        raw_rows=raw_n, duplicate_rows_removed=n_dups, undecided_dropped=undecided, labelled_claims=len(lab),
        fraud_total=int(y.sum()), fraud_rate=float(y.mean()),
        fraud_rate_new_partners=float(x[x.new].is_fraud.mean()), fraud_rate_old_partners=float(x[~x.new].is_fraud.mean()),
        new_partners_with_claims=int(len(newp)), new_partners_with_any_fraud=int((newp["sum"] > 0).sum()),
        old_partner_fraud_rate_after_may=float(x[(~x.new) & x.post].is_fraud.mean()),
        new_partner_small_after_may=dict(n=int(((x.new) & x.post & (x.claim_amount_inr < 2000)).sum()),
                                         fraud=int(x[(x.new) & x.post & (x.claim_amount_inr < 2000)].is_fraud.sum())),
        top10_partners_share_of_fraud=float(fp.head(10).sum() / fp.sum()), partners_with_fraud=int(len(fp)),
        fraud_rate_before_may=float(x[~x.post].is_fraud.mean()), fraud_rate_after_may=float(x[x.post].is_fraud.mean()),
        mean_fraud_amount_before_may=float(x[(~x.post) & (x.is_fraud == 1)].claim_amount_inr.mean()),
        mean_fraud_amount_after_may=float(x[x.post & (x.is_fraud == 1)].claim_amount_inr.mean()),
    )

    # ---------- final model on everything we have, then score the test file -------------
    mf = fit(X, y, d.values)
    te = te.copy()
    thn, thf = partner_history(te, lab)
    Xt = build(te, pa, pr, thn, thf)
    st = mf.predict_proba(Xt)[:, 1]
    sub = pd.DataFrame({"claim_id": te["claim_id"], "score": np.round(st, 6)})
    sample = pd.read_csv(D("sample_submission.csv"))
    assert set(sub.claim_id) == set(sample.claim_id) and len(sub) == len(sample) and sub.claim_id.is_unique
    sub = sample[["claim_id"]].merge(sub, on="claim_id", how="left")
    sub.to_csv(os.path.join(ROOT, "predictions.csv"), index=False)
    ev["test"] = dict(rows=len(sub), mean_score=float(st.mean()),
                      flagged_at_threshold=int((st >= thr).sum()), flagged_share=float((st >= thr).mean()),
                      flagged_per_month=float((st >= thr).sum() / 3), expected_frauds_sum_of_scores=float(st.sum()),
                      share_from_new_partners=float(Xt["partner_is_new"].mean()),
                      top120_new_partner_share=float(Xt["partner_is_new"].values[np.argsort(-st)[:120]].mean()))

    # Deployment threshold: flag the top CAPACITY_PER_MONTH claims per ~750-claim month (policy: desk reviews 40/month).
    deploy_thr = float(np.sort(st)[::-1][3 * CAPACITY_PER_MONTH - 1])
    ev["test"]["deploy_threshold"] = deploy_thr
    ev["test"]["flagged_at_deploy_threshold"] = int((st >= deploy_thr).sum())
    joblib.dump(dict(model=mf, threshold=deploy_thr, goodwill=GOODWILL, review_cost=REVIEW_COST), A("model.joblib"))

    # Service artifacts. Raw claims are NOT shipped (policy sec. 10: no operational data in public repos).
    # Partner facts are shipped under a pseudonymous key (weak protection - see README, "What is wrong").
    allp = lab.groupby("partner_id").is_fraud.agg(n="size", frauds="sum")
    stats = {}
    for r in pa.itertuples():
        key = partner_key(r.partner_id)
        n_, f_ = (int(allp.loc[r.partner_id, "n"]), int(allp.loc[r.partner_id, "frauds"])) if r.partner_id in allp.index else (0, 0)
        stats[key] = dict(onboarded=str(r.onboarded_date.date()), type=r.partner_type, n=n_, frauds=f_)
    json.dump(stats, open(A("partner_stats.json"), "w"))
    json.dump(pr.to_dict("records"), open(A("products.json"), "w"))
    json.dump(ev, open(A("evidence.json"), "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in ev.items() if k not in ("missed", "facts")}, indent=1, default=str))
    print(json.dumps(ev["facts"], indent=1)); print("missed:", ev["missed"])


if __name__ == "__main__":
    main()
