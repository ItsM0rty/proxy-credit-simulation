"""Step 2: the Savings Score.

The score says: if you moved each thing you control to a good but realistic value,
a model of how carriers have been pricing venues like yours predicts your premium
would be X% lower. Per-feature scores split X by action.

It assumes that
  1. carriers will keep pricing the way the quote model learned. This is a model of
     carrier behaviour, not of risk: if carriers reward a feature because it
     correlates with safe venues, the score rewards it too;
  2. changing one feature leaves everything else about the venue unchanged;
  3. carriers do not re-price when many venues make the same change;
  4. the targets in `targets()` are realistic for the venue.
stress.py tests assumptions 1 and 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from data import OBSERVABLE

# direction: -1 = lower premium when feature is higher, +1 = higher premium when higher.
CONTROLLABLE = {
    "staff_training_score": -1,
    "security_staff": -1,
    "incident_response_time": +1,
    "camera_coverage": -1,
    "license_compliance": -1,
    "late_night_hours": +1,
}


def targets(book: pd.DataFrame, good: float = 0.75) -> dict:
    """Target per controllable feature (A15): the book's 75th percentile, or one hour
    earlier. A venue already past the target keeps its value."""
    hi, lo = book[list(CONTROLLABLE)].quantile(good), book[list(CONTROLLABLE)].quantile(1 - good)
    return {
        "staff_training_score": lambda x: np.maximum(x, hi["staff_training_score"]),
        "security_staff": lambda x: np.maximum(x, hi["security_staff"]),
        "incident_response_time": lambda x: np.minimum(x, lo["incident_response_time"]),
        "camera_coverage": lambda x: np.maximum(x, hi["camera_coverage"]),
        "license_compliance": lambda x: np.maximum(x, hi["license_compliance"]),
        "late_night_hours": lambda x: np.maximum(x - 1, 0),
    }


class SavingsScore:
    def __init__(self, seed: int = 0):
        # Monotone constraints keep per-feature explanations sane (more training never
        # raises the predicted premium). Non-controllable features are unconstrained.
        mono = [CONTROLLABLE.get(c, 0) for c in OBSERVABLE]
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            monotonic_cst=mono, random_state=seed)

    def fit(self, book: pd.DataFrame):
        self.book = book
        self.model.fit(book[OBSERVABLE], np.log(book["premium"]))
        self.moves = targets(book)
        return self

    def predict_premium(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(df[OBSERVABLE]))

    def moved(self, df: pd.DataFrame, features=None) -> pd.DataFrame:
        out = df.copy()
        for f in features or CONTROLLABLE:
            out[f] = self.moves[f](df[f].to_numpy())
        return out

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-venue predicted % premium reduction, total and per feature."""
        base = self.predict_premium(df)
        res = {f"save_{f}": 1 - self.predict_premium(self.moved(df, [f])) / base for f in CONTROLLABLE}
        res["savings_score"] = 1 - self.predict_premium(self.moved(df)) / base
        res["predicted_premium"] = base
        return pd.DataFrame(res, index=df.index)


def promised_vs_causal(ss: SavingsScore, df: pd.DataFrame, effects: dict | None = None) -> pd.DataFrame:
    """Per action: mean promised premium cut vs mean cut in true expected loss, holding
    the hidden management_quality fixed."""
    from data import true_expected_loss
    s = ss.score(df)
    base = true_expected_loss(df, effects)
    rows = {}
    for f in list(CONTROLLABLE) + ["ALL"]:
        moved = ss.moved(df, None if f == "ALL" else [f])
        promised = s["savings_score"] if f == "ALL" else s[f"save_{f}"]
        rows[f] = {"promised_premium_cut_%": promised.mean() * 100,
                   "true_loss_cut_%": (1 - true_expected_loss(moved, effects) / base).mean() * 100}
    out = pd.DataFrame(rows).T
    out["causal_share"] = out["true_loss_cut_%"] / out["promised_premium_cut_%"]
    return out


def fit_diagnostics(book: pd.DataFrame, seed: int = 0) -> dict:
    tr, te = train_test_split(book, test_size=0.25, random_state=seed)
    y_te = np.log(te["premium"])
    gbm = SavingsScore(seed).fit(tr).model
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(tr[OBSERVABLE], np.log(tr["premium"]))
    return {"gbm_holdout_r2_log_premium": gbm.score(te[OBSERVABLE], y_te),
            "ridge_holdout_r2_log_premium": ridge.score(te[OBSERVABLE], y_te)}


if __name__ == "__main__":
    from data import build_world

    pd.set_option("display.width", 160, "display.precision", 3)
    _, venues, _, _ = build_world()
    print("== model fit (predicting log premium from observable features) ==")
    for k, v in fit_diagnostics(venues).items():
        print(f"  {k}: {v:.3f}")

    ss = SavingsScore().fit(venues)
    s = ss.score(venues)
    print("\n== Savings Score (predicted % premium reduction if all controllable moves are made) ==")
    print((s["savings_score"] * 100).describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    print(f"share of venues with score > 20%: {(s['savings_score'] > 0.20).mean():.1%}")
    print(f"mean $ saving at year-0 premium: ${(s['savings_score'] * venues['premium']).mean():,.0f}")

    print("\n== per-feature mean predicted saving (% of premium) and share of total ==")
    per = s[[f"save_{f}" for f in CONTROLLABLE]].mean() * 100
    print(pd.DataFrame({"mean_%": per, "share_of_sum": per / per.sum()}).round(3))
    # % savings compound multiplicatively, so compare single vs joint moves in log space
    log_single = -np.log(1 - s[[f"save_{f}" for f in CONTROLLABLE]]).sum(axis=1)
    log_joint = -np.log(1 - s["savings_score"])
    print(f"interaction gap (log scale, sum of single moves minus joint): {(log_single - log_joint).mean():.3f}")

    print("\n== static check: promised premium cut vs true (synthetic) loss cut per action ==")
    print(promised_vs_causal(ss, venues).round(2))
