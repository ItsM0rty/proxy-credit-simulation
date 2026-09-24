"""Regenerate every number, results/*.csv and figures/*.png:  .venv\\Scripts\\python run_all.py

Deterministic: all randomness is seeded, and the stress test's worker processes are
single-threaded and their results are concatenated in task order.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import stress  # noqa: E402
from data import OBSERVABLE, TRUE_FREQ_EFFECT, TRUE_SEV_EFFECT_RESPONSE, build_world  # noqa: E402
from score import CONTROLLABLE, SavingsScore, fit_diagnostics, promised_vs_causal  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
SYN = "(synthetic simulation, not real premiums)"
COLORS = {0.1: "#9ecae1", 0.25: "#4292c6", 0.5: "#08519c", 1.0: "#e6550d"}


def band(ax, x, g, col, color, label):
    q = g.groupby(x)[col].agg(["median", "min", "max"])
    ax.plot(q.index, q["median"], color=color, label=label, lw=2)
    ax.fill_between(q.index, q["min"], q["max"], color=color, alpha=0.12)


def stage_1_2():
    _, venues, carrier, _ = build_world(n=4000, seed=0)
    truth = {"capacity": TRUE_FREQ_EFFECT["log_capacity"], "nightlife_density": TRUE_FREQ_EFFECT["log_nightlife_density"],
             "incident_response_time": TRUE_SEV_EFFECT_RESPONSE}
    print("== Stage 1-2 (seed 0) ==\ncarrier per-unit log coefficients vs truth:")
    print(pd.DataFrame({"carrier GLM": carrier.per_unit_coef(),
                        "truth": [truth.get(c, TRUE_FREQ_EFFECT.get(c, 0.0)) for c in OBSERVABLE]}).round(3))
    print("score model holdout R2 (log premium):", {k: round(v, 3) for k, v in fit_diagnostics(venues).items()})
    s = SavingsScore().fit(venues).score(venues)
    print("Savings Score %:", (s["savings_score"] * 100).describe(percentiles=[.1, .5, .9]).round(1).to_dict())

    rows = []
    for seed in stress.SEEDS:
        _, v, _, _ = build_world(seed=seed)
        rows.append(promised_vs_causal(SavingsScore(seed).fit(v), v).assign(seed=seed))
    pvc = pd.concat(rows).rename_axis("action").reset_index()
    q = pvc.groupby("action")[["promised_premium_cut_%", "true_loss_cut_%"]].agg(["median", "min", "max"])
    print("\npromised vs true loss cut per action, 20 seeds (median/min/max):\n", q.round(2))
    share = pvc[pvc.action.isin(stress.PROXIES)].groupby("seed")["promised_premium_cut_%"].sum() / \
        pvc[pvc.action.isin(list(CONTROLLABLE))].groupby("seed")["promised_premium_cut_%"].sum()
    print("share of summed per-action promises from proxies:", stress.med_range(share))

    order = list(CONTROLLABLE)[::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for off, col, color, lab in ((0.2, "promised_premium_cut_%", "#c0504d", "score's promised premium cut"),
                                 (-0.2, "true_loss_cut_%", "#4f81bd", "true (synthetic) loss cut")):
        med = q.loc[order, (col, "median")]
        err = [med - q.loc[order, (col, "min")], q.loc[order, (col, "max")] - med]
        ax.barh(y + off, med, height=0.38, color=color, xerr=err, capsize=2, label=lab)
    ax.set_yticks(y, order)
    ax.set_xlabel("mean % reduction across synthetic venues (median of 20 seeds, bars = range)")
    ax.set_title(f"Before anyone acts: cameras and paperwork are promised\nsavings with no loss behind them {SYN}",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    plt.tight_layout(); plt.savefig(FIG / "fig1_promised_vs_causal.png", dpi=150); plt.close()


def fig_promise_over_time(per):
    g = stress.derived(per)
    g = g[(g.setting == "baseline") & (g.K == stress.DEFAULT_K) & (g.p > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, color in COLORS.items():
        band(axes[0], "t", g[g.p == p], "kept", color, f"{p:.0%} of venues adopt")
    axes[0].axhline(1, color="k", lw=0.8, ls=":")
    axes[0].set(xlabel="period after adoption (synthetic)", ylabel="realized / promised premium cut (synthetic)",
                title="Calibration of the promise, adopters")
    axes[0].legend(fontsize=8)
    x = g[g.p == 0.5]
    for col, color, lab in (("gap_proxy_heavy_pp", "#c0504d", "proxy-heavy adopters (>=2 of 3 actions are proxies)"),
                            ("unbacked_pp", "#636363", "all adopters"),
                            ("gap_causal_only_pp", "#4f81bd", "causal-only adopters")):
        band(axes[1], "t", x, col, color, lab)
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set(xlabel="period after adoption (synthetic)",
                ylabel="realized premium cut - true loss cut, pp (synthetic)",
                title="Who is over-rewarded (p = 50%)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"The carrier never claws back the proxy discount; real risk reducers stay under-credited\n{SYN}",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "fig2_promise_over_time.png", dpi=150); plt.close()


def fig_effect_sweep(per, summ):
    g = per[(per.K == stress.DEFAULT_K) & (per.p == 0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    shades = {0.0: "#c0504d", 0.1: "#fd8d3c", 0.3: "#31a354"}
    for ax, feat, key in ((axes[0], "camera", "camera_ttc"), (axes[1], "license", "license_ttc")):
        for frac, color in shades.items():
            name = "baseline" if frac == 0 else f"{feat}={frac}"
            s = summ[(summ.setting == name) & (summ.K == stress.DEFAULT_K) & (summ.p == 0.5)][key]
            band(ax, "t", g[g.setting == name], "noncausal_share", color,
                 f"true effect = {frac:.0%} of training's; corrected in {s.notna().sum()}/{len(s)} seeds")
        ax.set(xlabel="period after adoption (synthetic)", title=f"{feat} true-effect sweep")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("share of carrier credit not backed by loss (synthetic)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("Non-causal share shrinks only if the proxy really works; the carrier's re-pricing barely moves it\n"
                 f"p = 50%, K = 3. {SYN}", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "fig3_proxy_effect_sweep.png", dpi=150); plt.close()


def fig_validity(per, summ, calib):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    s = summ[(summ.setting == "baseline") & (summ.K == stress.DEFAULT_K)]
    for col, color, lab in (("spearman_before", "#636363", "before adoption"),
                            ("spearman_after", "#08519c", f"after, t={stress.HORIZON} (score re-fit)")):
        band(axes[0], "p", s, col, color, lab)
    axes[0].set(xlabel="fraction of venues adopting (synthetic)", ylim=(0, 1),
                ylabel="Spearman(score, true loss cut) (synthetic)", title="Ranking survives adoption")
    axes[0].legend(fontsize=8)

    c = calib[(calib.setting == "baseline") & (calib.K == stress.DEFAULT_K) & (calib.p == 0.5)]
    m = c.groupby(["t", "bin"])[["promised", "realized", "true_cut"]].median() * 100
    lim = max(m.to_numpy().max(), 1) * 1.1
    axes[1].plot([0, lim], [0, lim], color="k", lw=0.8, ls=":")
    for t, style in ((1, "--"), (stress.HORIZON, "-")):
        axes[1].plot(m.loc[t, "promised"], m.loc[t, "realized"], style, marker="o", color="#c0504d",
                     label=f"realized premium cut, t={t}")
    axes[1].plot(m.loc[1, "promised"], m.loc[1, "true_cut"], "-", marker="s", color="#4f81bd", label="true loss cut")
    axes[1].set(xlabel="promised premium cut, % (quintiles, synthetic)", ylabel="% cut (synthetic)",
                title="Calibration curve, p = 50%")
    axes[1].legend(fontsize=8)

    g = per[per.setting == "baseline"].groupby(["K", "p", "seed"])["loss_ratio"].max().reset_index()
    for K, color in zip(stress.K_GRID, ("#9ecae1", "#08519c", "#e6550d")):
        band(axes[2], "p", g[g.K == K], "loss_ratio", color, f"carrier window K={K}")
    axes[2].set(xlabel="fraction of venues adopting (synthetic)", ylabel="peak loss ratio over 10 periods (synthetic)",
                title="The carrier's book pays for it")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Score validity vs adoption: ranks hold, calibration and carrier results do not\n{SYN}", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "fig4_validity_vs_adoption.png", dpi=150); plt.close()


def main():
    FIG.mkdir(exist_ok=True)
    for old in FIG.glob("*.png"):
        old.unlink()
    stage_1_2()
    print("\n== Stage 3: stress test (20 seeds x 5 truth settings) ==")
    per, summ, calib = stress.run_experiments()
    stress.save(per, summ, calib, RES)
    stress.print_all(per, summ)
    fig_promise_over_time(per)
    fig_effect_sweep(per, summ)
    fig_validity(per, summ, calib)
    print(f"\nfigures: {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
