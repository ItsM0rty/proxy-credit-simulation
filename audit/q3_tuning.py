"""Q3: sensitivity of headline numbers to the post-hoc carrier tuning (credit cap,
tempering weight). Base-rate rebalancing stays on (Q1 covers it)."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

import harness as H
from data import effects_with
from stress import proxy_effect

CAPS = (0.15, 0.25, 0.35, 0.50, None)
WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)
P_GRID = (0.25, 0.5, 1.0)
OUT = Path(__file__).parent / "results"


def summarize(per, extra):
    per = per.assign(kept=per.realized / per.promised)
    out = []
    for p, g in per.groupby("p"):
        t1, t10 = g[g.t == 1].iloc[0], g[g.t == 10].iloc[0]
        out.append({**extra, "p": p, "kept_t1": t1.kept, "kept_t10": t10.kept,
                    "gap_proxy_heavy_t1": t1.gap_proxy_heavy * 100, "gap_proxy_heavy_t10": t10.gap_proxy_heavy * 100,
                    "gap_causal_only_t1": t1.gap_causal_only * 100, "share_proxy_heavy": t1.share_proxy_heavy,
                    "noncausal_t1": t1.noncausal_share, "noncausal_t10": t10.noncausal_share,
                    "camera_coef_t10": t10.coef_camera_coverage, "peak_lr": g.loss_ratio.max(),
                    "at_cap_t1": t1.share_at_credit_cap_after})
    return out


def task(args):
    seed, cap, weight = args
    kw = {"cap": cap, "weight": weight}
    tag = {"seed": seed, "cap": "none" if cap is None else cap, "weight": weight}
    with threadpool_limits(limits=1):
        w = H.make_world(seed, carrier_kw=kw)
        static = {**tag, **H.static_promises(w)}
        dyn = summarize(pd.concat([H.run(w, p) for p in P_GRID]), {**tag, "setting": "baseline"})
        for frac in (0.1, 0.3):
            wc = H.make_world(seed, effects=effects_with(camera_coverage=proxy_effect(frac)), carrier_kw=kw)
            dyn += summarize(H.run(wc, 0.5), {**tag, "setting": f"camera={frac}"})
    return static, dyn


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tasks = [(s, c, wt) for c in CAPS for wt in WEIGHTS for s in range(20)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(task, tasks, chunksize=2))
    pd.DataFrame([r[0] for r in res]).to_csv(OUT / "q3_static.csv", index=False)
    pd.DataFrame([d for r in res for d in r[1]]).to_csv(OUT / "q3_dynamic.csv", index=False)
    print("done", len(res))
