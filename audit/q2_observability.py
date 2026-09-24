"""Q2: does the carrier correct the camera credit if it observes a noisy proxy for
management quality (a loss-control inspection score)?

inspection_score = management_quality + sd * N(0,1); corr with mgmt = 1/sqrt(1+sd^2).
Variant "t1":      the carrier starts rating it at t=1 (coefficient starts at 0), starting
                   from the project's miscalibrated burn-in carrier. This is time-to-correction.
Variant "burnin":  the carrier has rated it since burn-in (how wrong is it at t=0?).
Horizon extended to 20 periods; K=3; baseline truth (camera effect 0)."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

import harness as H

NOISE = {"none": None, "high": 2.0, "medium": 1.0, "low": 0.5, "very low": 0.25, "zero": 0.0}
P_GRID = (0.0, 0.5, 1.0)
HORIZON = 20
OUT = Path(__file__).parent / "results"


def task(args):
    seed, level = args
    sd = NOISE[level]
    out = []
    with threadpool_limits(limits=1):
        w = H.make_world(seed, mgmt_proxy_sd=sd)
        feat = None if sd is None else "inspection_score"
        for p in P_GRID:
            out.append(H.run(w, p, horizon=HORIZON, add_feature=feat).assign(noise=level, variant="t1"))
        if sd is not None:
            wb = H.make_world(seed, mgmt_proxy_sd=sd, carrier_kw={"extra_exposure": ["inspection_score"]})
            for p in P_GRID:
                out.append(H.run(wb, p, horizon=HORIZON).assign(noise=level, variant="burnin"))
    return pd.concat(out)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tasks = [(s, lv) for s in range(20) for lv in NOISE]
    with ProcessPoolExecutor(max_workers=12) as ex:
        per = pd.concat(list(ex.map(task, tasks)), ignore_index=True)
    per.to_csv(OUT / "q2_periods.csv", index=False)
    print("saved", len(per), "rows")
