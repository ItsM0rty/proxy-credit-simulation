"""Audit harness: imports the project's src/ unchanged and swaps in a parameterized carrier.
With default arguments it reproduces the project exactly (check_harness.py)."""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import data  # noqa: E402
from data import EXPOSURE, SCHEDULE, Carrier, sample_claims, true_expected_loss  # noqa: E402
from score import CONTROLLABLE, SavingsScore, promised_vs_causal  # noqa: E402
from stress import PROXIES, apply_actions, noncausal_share, recommend  # noqa: E402

LOGGED = ("capacity", "nightlife_density")


class AuditCarrier(Carrier):
    """data.Carrier with the post-hoc choices exposed:
    cap        schedule credit/debit cap (None = uncapped)
    weight     share of the GLM schedule indication applied ("tempering")
    rebalance  "on" (project default), "frozen" (off-balance fixed at its burn-in value),
               "never" (off-balance always 0)
    Also supports adding an exposure feature mid-run (add_exposure_feature)."""

    def __init__(self, cap=data.SCHEDULE_CAP, weight=data.SCHEDULE_WEIGHT, rebalance="on", extra_exposure=(), **kw):
        super().__init__(**kw)
        self.cap, self.weight, self.rebalance_mode = cap, weight, rebalance
        self.exposure = list(EXPOSURE) + list(extra_exposure)
        self.schedule = list(SCHEDULE)
        self._initialized = False

    @property
    def features(self):
        return self.exposure + self.schedule

    def design(self, df):  # instance method shadows the staticmethod
        X = df[self.features].copy()
        for c in LOGGED:
            X[c] = np.log(X[c])
        return X.to_numpy(float)

    def initialize(self, burn_in):
        super().initialize(burn_in)
        self._initialized = True

    def _rebalance(self, book):
        if self.rebalance_mode == "never":
            self.off_balance = 0.0
            return
        if self.rebalance_mode == "frozen" and self._initialized:
            return
        super()._rebalance(book)

    def schedule_factor(self, df):
        Z = self.scaler.transform(self.design(df))
        k = len(self.exposure)
        f = np.exp(self.weight * Z[:, k:] @ self.coef[k:])
        if self.cap is not None:
            f = np.clip(f, 1 - self.cap, 1 + self.cap)
        return f

    def expected_cost(self, df):
        Z = self.scaler.transform(self.design(df))
        k = len(self.exposure)
        return np.exp(Z[:, :k] @ self.coef[:k] + self.intercept + self.off_balance) * self.schedule_factor(df)

    def per_unit_coef(self):
        return pd.Series(self.coef / self.scaler.scale_, index=self.features)

    def add_exposure_feature(self, name):
        """Start rating a new directly-rated feature. Its coefficient starts at 0 (not yet
        priced) and is learned by the carrier's unchanged partial-update rule."""
        k = len(self.exposure)
        old_mean, old_scale = self.scaler.mean_, self.scaler.scale_
        self.exposure.append(name)
        burn = pd.concat(self.history[:3])  # the same burn-in rows the frozen scaler was fit on
        self.scaler = StandardScaler().fit(self.design(burn))
        keep = np.r_[0:k, k + 1:len(self.features)]
        assert np.allclose(self.scaler.mean_[keep], old_mean) and np.allclose(self.scaler.scale_[keep], old_scale)
        self.coef = np.insert(self.coef, k, 0.0)


@dataclass
class World:
    seed: int
    effects: dict
    venues: pd.DataFrame
    carrier: AuditCarrier
    ss: SavingsScore
    chosen: pd.DataFrame
    u_adopt: np.ndarray


def make_world(seed, effects=None, carrier_kw=None, mgmt_proxy_sd=None, n=4000, burn_in_years=3):
    """Mirror of data.build_world + stress.make_world with AuditCarrier. mgmt_proxy_sd adds an
    'inspection_score' column = management_quality + sd * N(0,1) (never used unless the carrier
    is told to rate it); it uses its own RNG stream so nothing else changes."""
    effects = effects or data.effects_with()
    if not data._REAL_CACHE:
        data._REAL_CACHE.append(data.load_real_bars())
    venues = data.make_venues(data._REAL_CACHE[0], n=n, seed=seed, effects=effects)
    eps = np.random.default_rng([seed, 11]).standard_normal(n)
    venues["inspection_score"] = venues["management_quality"] + (mgmt_proxy_sd or 0.0) * eps
    rng = np.random.default_rng(seed + 100)
    carrier = AuditCarrier(seed=seed + 200, **(carrier_kw or {}))
    carrier.initialize([sample_claims(venues, rng, effects).assign(year=y) for y in range(-burn_in_years, 0)])
    venues["premium"] = carrier.quote(venues, year=0)
    ss = SavingsScore(seed).fit(venues)
    u = np.random.default_rng([seed, 3]).random(len(venues))
    return World(seed, effects, venues, carrier, ss, recommend(ss, venues), u)


def run(w: World, p: float, K: int = 3, horizon: int = 10, add_feature: str | None = None):
    """Mirror of stress.run_one (same metrics, same claim RNG), plus per-feature diagnostics."""
    carrier = copy.deepcopy(w.carrier)
    carrier.window = K
    if add_feature:
        carrier.add_exposure_feature(add_feature)
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

    rows = []
    for t in range(1, horizon + 1):
        r_old, r_new = carrier.expected_cost(x0), carrier.expected_cost(x1)
        realized = 1 - r_new / r_old
        coef = carrier.per_unit_coef()
        rec = {"seed": w.seed, "p": p, "K": K, "t": t,
               "loss_ratio": loss1.sum() / (r_new.sum() * carrier.load),
               "off_balance": carrier.off_balance,
               "nonadopter_rate_change": (r_old[~adopt] / rate0[~adopt]).mean() - 1 if (~adopt).any() else np.nan,
               **{f"coef_{c}": coef[c] for c in coef.index},
               **noncausal_share(carrier.expected_cost, x0, w.ss.moves, w.effects)}
        if a.any():
            rec.update(promised=promised[a].mean(), realized=realized[a].mean(), true_cut=true_cut[a].mean())
            # log-scale decomposition of adopters' realized rate ratio by feature (pre-cap), plus cap effect
            Z0 = carrier.scaler.transform(carrier.design(x0[a]))
            Z1 = carrier.scaler.transform(carrier.design(x1[a]))
            k = len(carrier.exposure)
            dZ = (Z1 - Z0) * carrier.coef
            dZ[:, k:] *= carrier.weight
            for j, c in enumerate(carrier.features):
                rec[f"dlog_{c}"] = dZ[:, j].mean()
            sf0 = np.exp(carrier.weight * Z0[:, k:] @ carrier.coef[k:])
            sf1 = np.exp(carrier.weight * Z1[:, k:] @ carrier.coef[k:])
            cap = carrier.cap if carrier.cap is not None else np.inf
            rec["dlog_cap_effect"] = (np.log(np.clip(sf1, 1 - cap, 1 + cap) / np.clip(sf0, 1 - cap, 1 + cap))
                                      - np.log(sf1 / sf0)).mean()
            rec["share_at_credit_cap_after"] = (sf1 <= 1 - cap + 1e-12).mean()
            for g, m in groups.items():
                rec[f"share_{g}"] = m.sum() / a.sum()
                rec[f"gap_{g}"] = (realized[m] - true_cut[m]).mean() if m.any() else np.nan
        rows.append(rec)
        claims = sample_claims(x1, np.random.default_rng([w.seed, 7, t]), w.effects)
        carrier.observe_year(claims.assign(year=t))
    return pd.DataFrame(rows)


def static_promises(w: World) -> dict:
    """The t=0 headline: share of summed per-action promises from proxies; bundle promised vs true."""
    pvc = promised_vs_causal(w.ss, w.venues, w.effects)
    acts = list(CONTROLLABLE)
    out = {"proxy_share_of_promises": pvc.loc[list(PROXIES), "promised_premium_cut_%"].sum()
           / pvc.loc[acts, "promised_premium_cut_%"].sum(),
           "bundle_promised_%": pvc.loc["ALL", "promised_premium_cut_%"],
           "bundle_true_%": pvc.loc["ALL", "true_loss_cut_%"]}
    for f in acts:
        out[f"promised_{f}"] = pvc.loc[f, "promised_premium_cut_%"]
        out[f"true_{f}"] = pvc.loc[f, "true_loss_cut_%"]
    causal = [f for f in acts if f not in PROXIES]
    out["causal_promised_sum"] = pvc.loc[causal, "promised_premium_cut_%"].sum()
    out["causal_true_sum"] = pvc.loc[causal, "true_loss_cut_%"].sum()
    return out


def mr(x, fmt="{:.3f}") -> str:
    x = pd.Series(x).dropna()
    if x.empty:
        return "n/a"
    return f"{fmt.format(x.median())} [{fmt.format(x.min())}, {fmt.format(x.max())}]"
