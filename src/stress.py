"""Step 3: what happens after venues act on the Savings Score?

Timeline of one run:
  t = 0   Step 2 world: carrier fitted on 3 burn-in years, score fitted on t=0 quotes.
  t = 1   A random fraction p of venues ("adopters") make their top-N_ACTIONS actions
          from the published score. Their true loss changes only via TRUE_FREQ_EFFECT.
  t = 1..HORIZON
          The carrier prices with its current rule, then sees that period's claims and
          re-fits on the last K periods (its unchanged learning rule from data.Carrier).

Rates are compared within a period (new features vs. old features under the same
carrier), so market trend and quote noise cancel out of "realized premium change".
"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data import TRUE_FREQ_EFFECT, build_world, effects_with, sample_claims, true_expected_loss
from score import CONTROLLABLE, SavingsScore

PROXIES = ("camera_coverage", "license_compliance")
STRONGEST_REDUCER = "staff_training_score"  # -0.60 per unit, same 0-1 scale as the proxies
P_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
K_GRID = (1, 3, 5)
DEFAULT_K = 3
EFFECT_FRACS = (0.0, 0.1, 0.3)
HORIZON = 10
N_ACTIONS = 3
SEEDS = tuple(range(20))
CORRECTION_TOL = 0.20


def proxy_effect(frac: float) -> float:
    return frac * TRUE_FREQ_EFFECT[STRONGEST_REDUCER]


def correction_band(true_effect: float) -> float:
    # A22: "within 20% of the true effect" is empty for a true effect of 0, so floor
    # it at 20% of the smallest non-zero effect in the sweep.
    floor = CORRECTION_TOL * abs(proxy_effect(min(f for f in EFFECT_FRACS if f > 0)))
    return max(CORRECTION_TOL * abs(true_effect), floor)


def recommend(ss: SavingsScore, df: pd.DataFrame, n_actions: int = N_ACTIONS) -> pd.DataFrame:
    """Each venue's top-n actions by predicted single-action saving (skipping ones that do nothing)."""
    save = ss.score(df)[[f"save_{f}" for f in CONTROLLABLE]].to_numpy()
    top = np.argsort(-save, axis=1)[:, :n_actions]
    chosen = np.zeros_like(save, dtype=bool)
    chosen[np.arange(len(df))[:, None], top] = True
    chosen &= save > 0.005
    return pd.DataFrame(chosen, columns=list(CONTROLLABLE), index=df.index)


def apply_actions(ss: SavingsScore, df: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for f in CONTROLLABLE:
        m = chosen[f].to_numpy()
        out.loc[m, f] = ss.moves[f](df.loc[m, f].to_numpy())
    return out


def noncausal_share(price, df: pd.DataFrame, moves: dict, effects: dict) -> dict:
    """Share of the summed per-action premium cuts not backed by a true loss cut, counting
    only the proxy actions (A21). price: callable df -> premium-like array."""
    base_p, base_l = price(df), true_expected_loss(df, effects)
    prem, unbacked = {}, {}
    for f in CONTROLLABLE:
        m = df.copy()
        m[f] = moves[f](df[f].to_numpy())
        prem[f] = np.mean(1 - price(m) / base_p)
        loss_cut = np.mean(1 - true_expected_loss(m, effects) / base_l)
        unbacked[f] = max(prem[f] - loss_cut, 0.0) if f in PROXIES else 0.0
    total = sum(prem.values())
    out = {f"noncausal_{f}": unbacked[f] / total for f in PROXIES}
    out["noncausal_share"] = sum(unbacked.values()) / total
    return out


def spearman(a, b) -> float:
    return pd.Series(a).corr(pd.Series(b), method="spearman")


@dataclass
class World:
    setting: str
    seed: int
    effects: dict
    venues: pd.DataFrame
    carrier: object
    ss: SavingsScore
    chosen: pd.DataFrame          # what every venue would do if it adopted
    u_adopt: np.ndarray           # adopt iff u < p, so adopter sets are nested across p


def make_world(setting: str, effects: dict, seed: int) -> World:
    _, venues, carrier, _ = build_world(seed=seed, effects=effects)
    ss = SavingsScore(seed).fit(venues)
    u = np.random.default_rng([seed, 3]).random(len(venues))
    return World(setting, seed, effects, venues, carrier, ss, recommend(ss, venues), u)


def _score_validity(ss: SavingsScore, df: pd.DataFrame, effects: dict, seed: int):
    """Spearman(score's promised cut, true loss cut) over venues with >=1 recommended action."""
    chosen = recommend(ss, df)
    moved = apply_actions(ss, df, chosen)
    promised = 1 - ss.predict_premium(moved) / ss.predict_premium(df)
    true_cut = 1 - true_expected_loss(moved, effects) / true_expected_loss(df, effects)
    m = chosen.any(axis=1).to_numpy()
    return spearman(promised[m], true_cut[m])


def run_one(w: World, p: float, K: int, horizon: int = HORIZON, validity: bool = False):
    carrier = copy.deepcopy(w.carrier)
    carrier.window = K
    adopt = w.u_adopt < p
    x0 = w.venues
    x1 = apply_actions(w.ss, x0, w.chosen & adopt[:, None])
    promised = 1 - w.ss.predict_premium(x1) / w.ss.predict_premium(x0)
    loss0, loss1 = true_expected_loss(x0, w.effects), true_expected_loss(x1, w.effects)
    true_cut = 1 - loss1 / loss0
    rate0 = carrier.expected_cost(x0)
    a = adopt & w.chosen.any(axis=1).to_numpy()
    n_proxy = w.chosen[list(PROXIES)].sum(axis=1).to_numpy()
    groups = {"proxy_heavy": a & (n_proxy >= 2), "causal_only": a & (n_proxy == 0)}

    rows, calib = [], None
    for t in range(1, horizon + 1):
        r_old, r_new = carrier.expected_cost(x0), carrier.expected_cost(x1)
        realized = 1 - r_new / r_old
        coef = carrier.per_unit_coef()
        rec = {"setting": w.setting, "seed": w.seed, "p": p, "K": K, "t": t,
               "camera_coef": coef["camera_coverage"], "license_coef": coef["license_compliance"],
               "loss_ratio": loss1.sum() / (r_new.sum() * carrier.load),
               "nonadopter_rate_change": (r_old[~adopt] / rate0[~adopt]).mean() - 1 if (~adopt).any() else np.nan,
               **noncausal_share(carrier.expected_cost, x0, w.ss.moves, w.effects)}
        if a.any():
            rec.update(promised=promised[a].mean(), realized=realized[a].mean(), true_cut=true_cut[a].mean())
            for g, m in groups.items():
                rec[f"share_{g}"] = m.sum() / a.sum()
                rec[f"gap_{g}"] = (realized[m] - true_cut[m]).mean() if m.any() else np.nan
            if t in (1, horizon):
                bins = pd.qcut(promised[a], 5, labels=False, duplicates="drop")
                cb = pd.DataFrame({"bin": bins, "promised": promised[a], "realized": realized[a],
                                   "true_cut": true_cut[a]}).groupby("bin").mean().assign(t=t)
                calib = cb if calib is None else pd.concat([calib, cb])
        rows.append(rec)
        claims = sample_claims(x1, np.random.default_rng([w.seed, 7, t]), w.effects)
        carrier.observe_year(claims.assign(year=t))

    summary = {"setting": w.setting, "seed": w.seed, "p": p, "K": K}
    if validity:
        summary["spearman_before"] = _score_validity(w.ss, x0, w.effects, w.seed)
        book = x1.assign(premium=carrier.quote(x1, year=horizon))
        summary["spearman_after"] = _score_validity(SavingsScore(w.seed).fit(book), x1, w.effects, w.seed)
    if calib is not None:
        calib = calib.reset_index().assign(setting=w.setting, seed=w.seed, p=p, K=K)
    return pd.DataFrame(rows), summary, calib


def settings() -> dict:
    s = {"baseline": effects_with()}
    for frac in EFFECT_FRACS[1:]:
        s[f"camera={frac}"] = effects_with(camera_coverage=proxy_effect(frac))
        s[f"license={frac}"] = effects_with(license_compliance=proxy_effect(frac))
    return s


def _world_task(args):
    name, seed = args
    with threadpool_limits(limits=1):  # one thread per worker process; avoids oversubscription
        w = make_world(name, settings()[name], seed)
        out = []
        for p in P_GRID:
            for K in (K_GRID if name == "baseline" else (DEFAULT_K,)):
                out.append(run_one(w, p, K, validity=(K == DEFAULT_K)))
    return out


def run_experiments(seeds=SEEDS, workers: int | None = None):
    """Every (setting, seed) world is independent and seeded, so running them in parallel
    and concatenating in task order gives identical results to a serial run."""
    tasks = [(name, seed) for name in settings() for seed in seeds]
    t0 = time.time()
    if workers == 1:
        results = [_world_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_world_task, tasks))
    print(f"  {len(tasks)} worlds, {sum(len(r) for r in results)} runs in {time.time() - t0:.0f}s")
    runs = [r for rs in results for r in rs]
    per = pd.concat([r[0] for r in runs], ignore_index=True)
    summ = add_time_to_correction(per, pd.DataFrame([r[1] for r in runs]))
    calib = pd.concat([r[2] for r in runs if r[2] is not None], ignore_index=True)
    return per, summ, calib


def add_time_to_correction(per: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """Periods of carrier learning until its per-unit GLM estimate for the proxy is within
    the correction band of the true effect (NaN = never within horizon)."""
    eff = settings()
    out = []
    for key, g in per.groupby(["setting", "seed", "p", "K"]):
        row = list(key)
        for f, col in (("camera_coverage", "camera_coef"), ("license_compliance", "license_coef")):
            tr = eff[key[0]][f]
            ok = g.loc[(g[col] - tr).abs() <= correction_band(tr), "t"]
            row.append(int(ok.min()) - 1 if len(ok) else np.nan)
        out.append(row)
    ttc = pd.DataFrame(out, columns=["setting", "seed", "p", "K", "camera_ttc", "license_ttc"])
    return summ.merge(ttc, on=["setting", "seed", "p", "K"])


def med_range(x) -> str:
    x = pd.Series(x).dropna()
    if x.empty:
        return "n/a"
    return f"{x.median():.3f} [{x.min():.3f}, {x.max():.3f}]"


def ttc_text(x) -> str:
    x = pd.Series(x)
    if x.notna().sum() == 0:
        return f"never within horizon ({len(x)}/{len(x)} seeds)"
    return f"median {x.median():.0f} periods; never in {x.isna().sum()}/{len(x)} seeds"


def derived(per: pd.DataFrame) -> pd.DataFrame:
    g = per.copy()
    g["kept"] = g["realized"] / g["promised"]
    g["unbacked_pp"] = (g["realized"] - g["true_cut"]) * 100
    for c in ("gap_proxy_heavy", "gap_causal_only"):
        g[c + "_pp"] = g[c] * 100
    base = g[g.p == 0].set_index(["setting", "seed", "K", "t"])["nonadopter_rate_change"]
    g["spillover_pp"] = (g["nonadopter_rate_change"]
                         - base.reindex(pd.MultiIndex.from_frame(g[["setting", "seed", "K", "t"]])).to_numpy()) * 100
    return g


def headline(per, summ, setting="baseline", K=DEFAULT_K) -> pd.DataFrame:
    g = derived(per)
    g = g[(g.setting == setting) & (g.K == K)]
    rows = {}
    for p in P_GRID:
        r = {}
        gp = g[g.p == p]
        for t in (1, HORIZON):
            x = gp[gp.t == t]
            r[f"promise kept (realized/promised) t={t}"] = med_range(x["kept"])
            r[f"realized - true cut, all adopters, pp t={t}"] = med_range(x["unbacked_pp"])
            r[f"realized - true cut, proxy-heavy adopters, pp t={t}"] = med_range(x["gap_proxy_heavy_pp"])
            r[f"noncausal share of carrier credit t={t}"] = med_range(x["noncausal_share"])
        r["share of adopters proxy-heavy"] = med_range(gp[gp.t == 1]["share_proxy_heavy"])
        r["realized - true cut, causal-only adopters, pp t=1"] = med_range(gp[gp.t == 1]["gap_causal_only_pp"])
        r[f"camera per-unit credit t={HORIZON}"] = med_range(gp[gp.t == HORIZON]["camera_coef"])
        r["peak loss ratio over horizon"] = med_range(gp.groupby("seed")["loss_ratio"].max())
        r[f"non-adopter rate change vs p=0, pp t={HORIZON}"] = med_range(gp[gp.t == HORIZON]["spillover_pp"])
        s = summ[(summ.setting == setting) & (summ.K == K) & (summ.p == p)]
        if s["spearman_after"].notna().any():
            r["spearman(score, true cut) before"] = med_range(s["spearman_before"])
            r[f"spearman(score, true cut) after, t={HORIZON}"] = med_range(s["spearman_after"])
        r["camera time-to-correction"] = ttc_text(s["camera_ttc"])
        rows[f"p={p}"] = r
    return pd.DataFrame(rows)


def effect_sweep_table(per, summ, p=0.5) -> pd.DataFrame:
    g = derived(per)
    g = g[(g.K == DEFAULT_K) & (g.p == p)]
    rows = {}
    for name in settings():
        x = g[g.setting == name]
        s = summ[(summ.setting == name) & (summ.K == DEFAULT_K) & (summ.p == p)]
        feat = "license" if name.startswith("license") else "camera"
        rows[name] = {
            "true per-unit effect": f"{settings()[name][feat + ('_compliance' if feat == 'license' else '_coverage')]:.3f}",
            f"{feat} credit t=1": med_range(x[x.t == 1][f"{feat}_coef"]),
            f"{feat} credit t={HORIZON}": med_range(x[x.t == HORIZON][f"{feat}_coef"]),
            "noncausal share t=1": med_range(x[x.t == 1]["noncausal_share"]),
            f"noncausal share t={HORIZON}": med_range(x[x.t == HORIZON]["noncausal_share"]),
            f"realized-true, proxy-heavy, pp t={HORIZON}": med_range(x[x.t == HORIZON]["gap_proxy_heavy_pp"]),
            f"{feat} time-to-correction": ttc_text(s[f"{feat}_ttc"]),
        }
    return pd.DataFrame(rows)


def k_sweep_table(per, summ) -> pd.DataFrame:
    g = derived(per)
    g = g[g.setting == "baseline"]
    rows = {}
    for K in K_GRID:
        for p in (0.0, 0.5, 1.0):
            x = g[(g.K == K) & (g.p == p)]
            s = summ[(summ.setting == "baseline") & (summ.K == K) & (summ.p == p)]
            rows[f"K={K} p={p}"] = {
                f"promise kept t={HORIZON}": med_range(x[x.t == HORIZON]["kept"]),
                f"noncausal share t={HORIZON}": med_range(x[x.t == HORIZON]["noncausal_share"]),
                f"camera credit t={HORIZON}": med_range(x[x.t == HORIZON]["camera_coef"]),
                "peak loss ratio": med_range(x.groupby("seed")["loss_ratio"].max()),
                "camera time-to-correction": ttc_text(s["camera_ttc"]),
            }
    return pd.DataFrame(rows)


def too_clean_checks(per: pd.DataFrame, summ: pd.DataFrame) -> list[str]:
    """Flags: any Spearman > 0.9; a headline metric with ~zero seed spread; or seed 0
    (the Step 2 world) lying outside the range of the other 19 seeds."""
    flags = []
    for c in ("spearman_before", "spearman_after"):
        if (summ[c] > 0.9).any():
            flags.append(f"{c} > 0.9 in {(summ[c] > 0.9).sum()} runs")
    g = derived(per)
    g = g[(g.setting == "baseline") & (g.K == DEFAULT_K) & (g.p > 0) & g.t.isin([1, HORIZON])]
    for (p, t), x in g.groupby(["p", "t"]):
        for m in ("kept", "unbacked_pp", "gap_proxy_heavy_pp", "noncausal_share"):
            v = x.set_index("seed")[m]
            others = v.drop(index=0, errors="ignore")
            if v.max() - v.min() < 1e-6:
                flags.append(f"{m} p={p} t={t}: zero spread across seeds")
            elif 0 in v.index and not (others.min() <= v[0] <= others.max()):
                flags.append(f"{m} p={p} t={t}: seed 0 = {v[0]:.3f} outside other seeds "
                             f"[{others.min():.3f}, {others.max():.3f}]")
    return flags


def print_all(per, summ):
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print("\n== Stage 3a: baseline (proxies have zero true effect), K=3; median [min, max] over seeds ==")
    print(headline(per, summ).to_string())
    print("\n== Stage 3b: carrier window K sweep (baseline) ==")
    print(k_sweep_table(per, summ).T.to_string())
    print("\n== Stage 3c: proxy true-effect sweep, p=0.5, K=3 (fraction of the training effect, -0.60/unit) ==")
    print(effect_sweep_table(per, summ).to_string())
    flags = too_clean_checks(per, summ)
    print("\ntoo-clean flags:", "\n  ".join([""] + flags) if flags else "none")


def save(per, summ, calib, out_dir):
    out_dir.mkdir(exist_ok=True)
    per.to_csv(out_dir / "stress_periods.csv", index=False, float_format="%.6g")
    summ.to_csv(out_dir / "stress_runs.csv", index=False, float_format="%.6g")
    calib.to_csv(out_dir / "stress_calibration.csv", index=False, float_format="%.6g")


def load(out_dir):
    return tuple(pd.read_csv(out_dir / f) for f in
                 ("stress_periods.csv", "stress_runs.csv", "stress_calibration.csv"))


if __name__ == "__main__":
    from pathlib import Path
    per, summ, calib = run_experiments()
    save(per, summ, calib, Path(__file__).resolve().parents[1] / "results")
    print_all(per, summ)
