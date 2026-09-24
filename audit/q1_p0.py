"""Q1 p=0 references: peak loss ratio, and non-adopter rate change vs p=0, by rebalancing mode."""
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from threadpoolctl import threadpool_limits

import harness as H


def task(args):
    seed, mode = args
    with threadpool_limits(limits=1):
        return H.run(H.make_world(seed, carrier_kw={"rebalance": mode}), 0.0).assign(rebalance=mode)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        p0 = pd.concat(list(ex.map(task, [(s, m) for s in range(20) for m in ("on", "frozen", "never")])))
    p0.to_csv(H.Path(__file__).parent / "results" / "q1_p0.csv", index=False)
    print("peak loss ratio p=0:")
    print(p0.groupby(["rebalance", "seed"]).loss_ratio.max().groupby("rebalance").apply(H.mr))

    q1 = pd.read_csv(H.Path(__file__).parent / "results" / "q1_periods.csv")
    base = p0[p0.t == 10].set_index(["rebalance", "seed"]).nonadopter_rate_change
    x = q1[(q1.t == 10) & (q1.p < 1)].copy()
    x["spill_pp"] = (x.nonadopter_rate_change - base.reindex(pd.MultiIndex.from_frame(x[["rebalance", "seed"]])).to_numpy()) * 100
    print("\nnon-adopter rate change vs p=0, pp, t=10 (median [min,max]):")
    print(x.groupby(["p", "rebalance"]).spill_pp.apply(lambda v: H.mr(v, "{:.1f}")).unstack())
