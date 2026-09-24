"""Q1 mechanism: range restriction at a common target.
(a) book diagnostics after adoption: how compressed are the moved features, and how
    confounded is the remaining variation with management quality?
(b) jitter test: adopters land at target + a random overshoot instead of exactly on the
    target. If the p=1 recovery is a pile-up artifact it should shrink."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import harness as H
from stress import apply_actions

P_GRID = (0.5, 0.75, 0.9, 1.0)
OUT = Path(__file__).parent / "results"
JITTER_FEATURES = {"staff_training_score": (+1, 0, 1), "security_staff": (+1, 0, 3),
                   "incident_response_time": (-1, 2, 45), "camera_coverage": (+1, 0, 1),
                   "license_compliance": (+1, 0, 1)}


def book_stats(w, p):
    adopt = w.u_adopt < p
    x1 = apply_actions(w.ss, w.venues, w.chosen & adopt[:, None])
    q75 = w.venues["staff_training_score"].quantile(0.75)
    tr = x1["staff_training_score"]
    resid = tr - np.polyval(np.polyfit(x1["management_quality"], tr, 1), x1["management_quality"])
    return {"seed": w.seed, "p": p, "train_sd": tr.std(), "train_at_target": np.isclose(tr, q75).mean(),
            "corr_train_mgmt": np.corrcoef(tr, x1["management_quality"])[0, 1],
            "train_var_share_from_mgmt": 1 - resid.var() / tr.var()}


def jittered_run(w, p, scale):
    """Same as H.run but adopters overshoot their target by |N(0, scale * book sd)|."""
    rng = np.random.default_rng([w.seed, 21])
    base_apply = H.apply_actions

    def apply_jitter(ss, df, chosen):
        out = base_apply(ss, df, chosen)
        for f, (d, lo, hi) in JITTER_FEATURES.items():
            m = chosen[f].to_numpy()
            e = np.abs(rng.standard_normal(len(df))) * scale * df[f].std()
            out.loc[m, f] = np.clip(out.loc[m, f].to_numpy() + d * e[m], lo, hi)
        return out

    H.apply_actions = apply_jitter
    try:
        return H.run(w, p)
    finally:
        H.apply_actions = base_apply


def task(seed):
    with threadpool_limits(limits=1):
        w = H.make_world(seed)
        stats = pd.DataFrame([book_stats(w, p) for p in (0.0,) + P_GRID])
        runs = pd.concat([jittered_run(w, p, s).assign(jitter=s) for p in P_GRID for s in (0.0, 0.25, 0.5)])
        return stats, runs


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(task, range(20)))
    stats = pd.concat([r[0] for r in res]); runs = pd.concat([r[1] for r in res])
    stats.to_csv(OUT / "q1_book_stats.csv", index=False); runs.to_csv(OUT / "q1_jitter.csv", index=False)
    pd.set_option("display.width", 200)
    print("== post-adoption book, staff_training_score (median over 20 seeds) ==")
    print(stats.groupby("p").median().drop(columns="seed").round(3))
    runs["kept"] = runs.realized / runs.promised
    t10 = runs[runs.t == 10]
    print("\n== promise kept t=10 by jitter scale (median [min,max]) ==")
    print(t10.groupby(["p", "jitter"]).kept.apply(H.mr).unstack())
    print("\n== training per-unit coef t=10 ==")
    print(t10.groupby(["p", "jitter"]).coef_staff_training_score.apply(H.mr).unstack())
    print("\n== share of adopters at credit cap t=10 ==")
    print(t10.groupby(["p", "jitter"]).share_at_credit_cap_after.median().unstack().round(3))
