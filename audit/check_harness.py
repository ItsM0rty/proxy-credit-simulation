"""Harness with default settings must reproduce results/stress_periods.csv."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import harness as H

ref = pd.read_csv(H.PROJECT / "results" / "stress_periods.csv")
with threadpool_limits(limits=4):
    for seed in (0, 7):
        w = H.make_world(seed)
        for p in (0.5, 1.0):
            got = H.run(w, p)
            exp = ref[(ref.setting == "baseline") & (ref.seed == seed) & (ref.p == p) & (ref.K == 3)].sort_values("t")
            for a, b in (("realized", "realized"), ("promised", "promised"), ("coef_camera_coverage", "camera_coef"),
                         ("noncausal_share", "noncausal_share"), ("loss_ratio", "loss_ratio")):
                d = np.abs(got[a].to_numpy() - exp[b].to_numpy()).max()
                print(f"seed {seed} p={p} {a:>22}: max abs diff {d:.2e}")
                assert d < 1e-4, a
print("harness reproduces project results")
