"""Audit figures A1-A3."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import harness as H

R, F = Path(__file__).parent / "results", Path(__file__).parent / "figures"
F.mkdir(exist_ok=True)
SYN = "synthetic simulation, not real premiums"


def band(ax, q, color, label, ls="-"):
    ax.plot(q.index, q["median"], color=color, lw=2, ls=ls, label=label)
    ax.fill_between(q.index, q["min"], q["max"], color=color, alpha=0.12)


# figure A1 (Q1)
q1 = pd.read_csv(R / "q1_periods.csv"); q1["kept"] = q1.realized / q1.promised
ref = pd.read_csv(H.PROJECT / "results" / "stress_periods.csv"); ref = ref[ref.setting == "baseline"]
ref["kept"] = ref.realized / ref.promised
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for mode, c, ls in (("on", "#08519c", "-"), ("frozen", "#e6550d", "--"), ("never", "#31a354", ":")):
    band(ax[0], q1[(q1.t == 10) & (q1.rebalance == mode)].groupby("p").kept.agg(["median", "min", "max"]),
         c, f"rebalancing {mode}", ls)
ax[0].set(xlabel="fraction of venues adopting (synthetic)", ylabel="promise kept at t=10 (synthetic)",
          title="Rebalancing on/off: curves coincide exactly")
ax[0].legend(fontsize=8)
for K, c in zip((1, 3, 5), ("#9ecae1", "#08519c", "#e6550d")):
    g = ref[(ref.K == K) & (ref.p == 1.0)].groupby("t").kept.agg(["median", "min", "max"])
    band(ax[1], g, c, f"p=100%, K={K}")
    ax[1].axvline(K + 1, color=c, lw=0.8, ls=":")
g = ref[(ref.K == 3) & (ref.p == 0.5)].groupby("t").kept.agg(["median", "min", "max"])
band(ax[1], g, "#636363", "p=50%, K=3", "--")
ax[1].set(xlabel="period (synthetic); dotted = first period with only post-adoption claims in window",
          ylabel="promise kept (synthetic)", title="p=100% recovery starts exactly at t = K+1")
ax[1].legend(fontsize=8)
fig.suptitle(f"A1. The p=100% anomaly is not rebalancing; it tracks the claims window\n{SYN}", fontsize=10)
plt.tight_layout(); plt.savefig(F / "A1_full_adoption.png", dpi=140); plt.close()

# figure A2 (Q2)
q2 = pd.read_csv(R / "q2_periods.csv")
from q2_observability import NOISE
x = [lv for lv in NOISE if NOISE[lv] is not None]
corr = [1 / np.sqrt(1 + NOISE[lv] ** 2) for lv in x]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for p, c in ((0.0, "#9ecae1"), (0.5, "#08519c"), (1.0, "#e6550d")):
    g = q2[(q2.variant == "t1") & (q2.p == p) & (q2.t == 10) & q2.noise.isin(x)]
    q = g.groupby("noise").coef_camera_coverage.agg(["median", "min", "max"]).reindex(x)
    q.index = corr
    band(ax[0], q, c, f"p={p:.0%}")
    none = q2[(q2.variant == "t1") & (q2.p == p) & (q2.t == 10) & (q2.noise == "none")].coef_camera_coverage.median()
    ax[0].scatter([0.3], [none], color=c, marker="x")
ax[0].axhspan(-0.012, 0.012, color="k", alpha=0.25, label="project correction band (+/-0.012)")
ax[0].axhspan(-0.10, 0.10, color="k", alpha=0.07, label="loose band (+/-0.10)")
ax[0].set(xlabel="corr(inspection score, management quality)  (x at 0.3 = not observed)",
          ylabel="carrier camera per-unit coef, t=10 (truth 0)", title="Camera credit vs observability (introduced at t=1)")
ax[0].legend(fontsize=7)


def n_corr(g, b, hz):
    return g.groupby("seed").apply(lambda s: ((s.t <= hz) & (s.coef_camera_coverage.abs() <= b)).any(),
                                   include_groups=False).sum()


for b, ls in ((0.012, "-"), (0.10, "--")):
    y = [n_corr(q2[(q2.variant == "t1") & (q2.p == 0.5) & (q2.noise == lv)], b, 10) for lv in x]
    ax[1].plot(corr, y, ls, marker="o", color="#08519c", label=f"band +/-{b}")
ax[1].set(xlabel="corr(inspection score, management quality)", ylabel="seeds (of 20) corrected within 10 periods",
          title="Correction needs corr >= ~0.9 (p=50%)", ylim=(0, 20.5))
ax[1].legend(fontsize=8)
fig.suptitle(f"A2. How much observability before the carrier corrects the camera credit\n{SYN}", fontsize=10)
plt.tight_layout(); plt.savefig(F / "A2_observability.png", dpi=140); plt.close()

# figure A3 (Q3)
st = pd.read_csv(R / "q3_static.csv"); st["cap"] = st.cap.astype(str)
order = ["0.15", "0.25", "0.35", "0.5", "none"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
panels = ((st.assign(v=st.proxy_share_of_promises * 100), "proxy share of per-action promises, %"),
          (st.assign(v=st["bundle_promised_%"] - st["bundle_true_%"]), "bundle promised - true loss cut, pp"),
          (st.assign(v=st.causal_promised_sum - st.causal_true_sum), "causal actions promised - true, pp"))
for a, (d, title) in zip(ax, panels):
    m = d.groupby(["cap", "weight"]).v.median().unstack().reindex(order)
    lim = np.abs(m.to_numpy()).max()
    im = a.imshow(m.to_numpy(), cmap="viridis" if "share" in title else "RdBu_r",
                  vmin=None if "share" in title else -lim, vmax=None if "share" in title else lim)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            dark = m.iat[i, j] < 24 if "share" in title else abs(m.iat[i, j]) > 0.6 * lim
            a.text(j, i, f"{m.iat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="w" if dark else "k",
                   fontweight="bold" if (order[i], m.columns[j]) == ("0.25", 0.4) else "normal")
    a.set_xticks(range(m.shape[1]), m.columns); a.set_yticks(range(m.shape[0]), order)
    a.set(xlabel="tempering weight", ylabel="schedule credit cap", title=title + " (synthetic)")
    plt.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(f"A3. Tuning grid, median of 20 seeds; project's choice (0.25, 0.4) in bold\n{SYN}", fontsize=10)
plt.tight_layout(); plt.savefig(F / "A3_tuning_grid.png", dpi=140); plt.close()
print(sorted(p.name for p in F.glob("*.png")))
