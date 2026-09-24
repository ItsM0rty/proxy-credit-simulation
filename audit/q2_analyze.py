import numpy as np
import pandas as pd

import harness as H
from q2_observability import NOISE

per = pd.read_csv(H.Path(__file__).parent / "results" / "q2_periods.csv")
per["kept"] = per.realized / per.promised
BANDS = {"project band 0.012": 0.012, "loose band 0.10": 0.10}
order = list(NOISE)
pd.set_option("display.width", 250, "display.max_colwidth", 40)


def ttc(g, band, horizon):
    ok = g.loc[(g.t <= horizon) & (g.coef_camera_coverage.abs() <= band), "t"]
    return ok.min() - 1 if len(ok) else np.nan


def ttc_text(x):
    x = pd.Series(x)
    n = x.notna().sum()
    return f"{n}/{len(x)} seeds; median {x.median():.0f}" if n else f"0/{len(x)} seeds"


for variant in ("t1", "burnin"):
    print(f"\n== variant {variant} ==")
    v = per[per.variant == variant]
    for p in (0.0, 0.5, 1.0):
        rows = {}
        for lv in order:
            g = v[(v.noise == lv) & (v.p == p)]
            if g.empty:
                continue
            r = {"corr(proxy,mgmt)": "-" if NOISE[lv] is None else f"{1 / np.sqrt(1 + NOISE[lv]**2):.2f}"}
            for t in (1, 10, 20):
                r[f"camera coef t={t}"] = H.mr(g[g.t == t].coef_camera_coverage)
            for name, b in BANDS.items():
                for hz in (10, 20):
                    r[f"corrected ({name}, <= {hz}p)"] = ttc_text(g.groupby("seed").apply(ttc, b, hz))
            r["noncausal share t=1"] = H.mr(g[g.t == 1].noncausal_share)
            r["noncausal share t=10"] = H.mr(g[g.t == 10].noncausal_share)
            r["noncausal share t=20"] = H.mr(g[g.t == 20].noncausal_share)
            if p > 0:
                r["promise kept t=10"] = H.mr(g[g.t == 10].kept)
            r["inspection coef t=10"] = H.mr(g[g.t == 10].get("coef_inspection_score", pd.Series(dtype=float)))
            rows[lv] = r
        print(f"\n== p={p} ==")
        print(pd.DataFrame(rows).to_string())
