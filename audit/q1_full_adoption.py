"""Q1: is the p=1.0 recovery of promise kept a base-rate rebalancing artifact?"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

import harness as H

P_GRID = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
MODES = ("on", "frozen", "never")
OUT = Path(__file__).parent / "results"


def task(args):
    seed, mode = args
    with threadpool_limits(limits=1):
        w = H.make_world(seed, carrier_kw={"rebalance": mode})
        return pd.concat([H.run(w, p).assign(rebalance=mode) for p in P_GRID])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tasks = [(s, m) for s in range(20) for m in MODES]
    with ProcessPoolExecutor(max_workers=12) as ex:
        per = pd.concat(list(ex.map(task, tasks)), ignore_index=True)
    per.to_csv(OUT / "q1_periods.csv", index=False)
    print("saved", len(per), "rows")
