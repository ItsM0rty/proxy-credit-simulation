import numpy as np
import pandas as pd

import harness as H

per = pd.read_csv(H.Path(__file__).parent / "results" / "q1_periods.csv")
per["kept"] = per.realized / per.promised
pd.set_option("display.width", 250, "display.max_columns", 40)

print("== promise kept t=10, median [min,max] over 20 seeds ==")
t10 = per[per.t == 10]
print(t10.groupby(["p", "rebalance"])["kept"].apply(H.mr).unstack())
for col in ("kept", "noncausal_share"):
    print(f"\n== max |{col}(on) - {col}(mode)| over all seeds/p/t ==")
    piv = per.pivot_table(index=["seed", "p", "t"], columns="rebalance", values=col)
    for m in ("frozen", "never"):
        print(m, np.abs(piv["on"] - piv[m]).max())
print("\n== peak loss ratio (shows the rebalancing switch does act on the book) ==")
print(per.groupby(["p", "rebalance", "seed"]).loss_ratio.max().groupby(["p", "rebalance"]).apply(H.mr).unstack())

on = per[per.rebalance == "on"]
print("\n== promise kept by t (median), rebalance on ==")
print(on.groupby(["t", "p"]).kept.median().unstack().round(3))
print("\n== noncausal share t=10 by p, rebalance on ==")
print(on[on.t == 10].groupby("p").noncausal_share.apply(H.mr))

dcols = [c for c in on.columns if c.startswith("dlog_")]
print("\n== adopters' mean log rate change by feature (negative = credit), t=1 and t=10, median over seeds ==")
for t in (1, 10):
    print(f"-- t={t}")
    print(on[on.t == t].groupby("p")[dcols + ["promised", "realized", "share_at_credit_cap_after"]].median().round(4).T)

ccols = [c for c in on.columns if c.startswith("coef_")]
print("\n== carrier per-unit GLM coefs at t=10, median over seeds ==")
print(on[on.t == 10].groupby("p")[ccols].median().round(3).T)
print("\n== t=1 coefs ==")
print(on[on.t == 1].groupby("p")[ccols].median().round(3).T)
