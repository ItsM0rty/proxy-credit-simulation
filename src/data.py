"""Step 1: data.

Real public data (cached under data/raw/) supplies only two feature marginals:
NYC DOHMH inspections (43nn-pn8j), filtered to bar-like names, give
prior_violations; NYS Liquor Authority active licenses (9s3h-dpkz), counted per
ZIP, give nightlife_density. No public dataset has venue-level premiums or
losses, so everything else is synthetic and seeded. COLUMN_SOURCES labels every
column.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

NYC_INSPECTIONS_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv"
NYS_LICENSES_URL = "https://data.ny.gov/resource/9s3h-dpkz.csv"
INSPECTION_WINDOW = ("2023-01-01", "2026-09-01")  # fixed so re-fetches are comparable
BAR_NAME = re.compile(r"\b(?:BAR|PUB|LOUNGE|TAVERN|SALOON|TAPROOM|BIERGARTEN|NIGHTCLUB)\b")
NOT_A_BAR = re.compile(r"\b(?:JUICE|SALAD|MILK|BAGEL|CANDY|SNACK|SMOOTHIE|ACAI|PROTEIN|COFFEE|ESPRESSO|"
                       r"DESSERT|CHOCOLATE|YOGURT|CEREAL|OXYGEN|NAIL|POKE)\b")
NYC_COUNTIES = ("New York", "Kings", "Queens", "Bronx", "Richmond")
ON_PREMISES_TYPES = ("Restaurant", "Additional Bar", "Club", "Food & Beverage Business",
                     "Hotel", "Catering Establishment")

COLUMN_SOURCES = {
    "prior_violations": "real marginal: NYC DOHMH critical violations 2023-01..2026-08 of a real bar",
    "nightlife_density": "real marginal: NYS SLA active on-premises licenses in that bar's ZIP",
    "capacity": "synthetic",
    "late_night_hours": "synthetic",
    "staff_training_score": "synthetic",
    "security_staff": "synthetic",
    "incident_response_time": "synthetic",
    "camera_coverage": "synthetic",
    "license_compliance": "synthetic",
    "management_quality": "synthetic latent (never shown to the carrier or the score)",
    "unobserved_risk": "synthetic latent (clientele, block, layout...; independent of everything)",
    "true_expected_loss": "synthetic ground truth",
    "claim_count": "synthetic (sampled)",
    "loss": "synthetic (sampled)",
    "premium": "synthetic (simulated carrier)",
}

# What the carrier (and therefore the score) can see. The carrier rates the first
# group directly; the second group only moves a capped schedule credit/debit.
EXPOSURE = ["capacity", "nightlife_density", "prior_violations", "late_night_hours"]
SCHEDULE = ["staff_training_score", "security_staff", "incident_response_time",
            "camera_coverage", "license_compliance"]
OBSERVABLE = EXPOSURE + SCHEDULE
SCHEDULE_CAP = 0.25     # A11; a common cap on schedule credits in small-commercial rating plans
SCHEDULE_WEIGHT = 0.40  # A11; share of the GLM-indicated schedule effect underwriters actually apply

# Ground-truth log-effects on claim frequency, per unit of the feature (A5-A7).
# camera_coverage, license_compliance and prior_violations are deliberately 0:
# they are proxies, correlated with low loss only through management_quality.
TRUE_FREQ_EFFECT = {
    "log_capacity": 0.70,
    "log_nightlife_density": 0.15,
    "late_night_hours": 0.08,
    "staff_training_score": -0.60,
    "security_staff": -0.25,
    "management_quality": -0.45,
    "unobserved_risk": 1.0,
    "camera_coverage": 0.0,
    "license_compliance": 0.0,
    "prior_violations": 0.0,
}
# log-effect on claim severity per minute of incident response time
TRUE_SEV_EFFECT_RESPONSE = 0.02
BASE_FREQ = 0.25          # claims/yr for a 150-cap venue at all-zero features
BASE_SEVERITY = 30_000    # $ mean per claim at 10-minute response
SEVERITY_SIGMA = 1.0      # lognormal spread of individual claim sizes


def _fetch_csv(url: str, params: dict, path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, params=params, timeout=300)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return pd.read_csv(path, dtype=str)


def load_nyc_bar_inspections() -> pd.DataFrame:
    """One row per violation line for bar-like venues, from the cached CSV."""
    like = " OR ".join(f"upper(dba) like '%{w}%'" for w in
                       ("BAR", "PUB", "LOUNGE", "TAVERN", "SALOON", "TAPROOM", "BIERGARTEN", "NIGHTCLUB"))
    params = {
        "$select": "camis,dba,boro,zipcode,inspection_date,violation_code,critical_flag",
        "$where": f"inspection_date >= '{INSPECTION_WINDOW[0]}' AND "
                  f"inspection_date < '{INSPECTION_WINDOW[1]}' AND ({like})",
        "$limit": 500_000,
    }
    df = _fetch_csv(NYC_INSPECTIONS_URL, params, RAW / "nyc_bar_inspections.csv")
    # SQL LIKE '%BAR%' also matches BARBECUE, BARN...; keep whole-word matches only.
    name = df["dba"].str.upper()
    return df[name.str.contains(BAR_NAME, na=False) & ~name.str.contains(NOT_A_BAR, na=False)]


def load_nys_license_density() -> pd.DataFrame:
    params = {
        "$select": "zipcode, count(*) AS n_licenses",
        "$where": "premisescounty in ({}) AND description in ({})".format(
            ",".join(f"'{c}'" for c in NYC_COUNTIES), ",".join(f"'{d}'" for d in ON_PREMISES_TYPES)),
        "$group": "zipcode",
        "$limit": 50_000,
    }
    df = _fetch_csv(NYS_LICENSES_URL, params, RAW / "nys_onprem_licenses_by_zip.csv")
    df["zipcode"] = df["zipcode"].str[:5]
    df["n_licenses"] = df["n_licenses"].astype(int)
    return df.groupby("zipcode", as_index=False)["n_licenses"].sum()


def load_real_bars() -> pd.DataFrame:
    """One row per real NYC bar: critical violations in the window + ZIP license density."""
    insp = load_nyc_bar_inspections()
    insp = insp[insp["zipcode"].notna()]
    insp["zipcode"] = insp["zipcode"].str[:5]
    bars = insp.groupby("camis").agg(
        dba=("dba", "first"), zipcode=("zipcode", "first"),
        n_inspections=("inspection_date", "nunique"),
        critical_violations=("critical_flag", lambda s: int((s == "Critical").sum())),
    ).reset_index()
    dens = load_nys_license_density()
    bars = bars.merge(dens, on="zipcode", how="inner")
    return bars.rename(columns={"n_licenses": "nightlife_density"})


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_venues(real_bars: pd.DataFrame, n: int = 4000, seed: int = 0, effects: dict | None = None) -> pd.DataFrame:
    """Synthetic venues, each anchored to a bootstrapped real bar for the two real columns."""
    rng = np.random.default_rng(seed)
    anchor = real_bars.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    viol = anchor["critical_violations"].to_numpy(float)
    density = anchor["nightlife_density"].to_numpy(float)

    v_z = (np.log1p(viol) - np.log1p(viol).mean()) / np.log1p(viol).std()
    d_z = (np.log(density) - np.log(density).mean()) / np.log(density).std()
    # Latent: well-run venues have fewer health violations, but most variation is independent.
    mgmt = -0.4 * v_z + np.sqrt(1 - 0.4**2) * rng.normal(size=n)

    df = pd.DataFrame({
        "venue_id": np.arange(n),
        "real_anchor_camis": anchor["camis"],
        "prior_violations": viol,
        "nightlife_density": density,
        "capacity": np.clip(rng.lognormal(np.log(150), 0.5, n), 30, 1000).round(),
        "late_night_hours": np.clip(np.round(2 * rng.normal(1.8 - 0.3 * mgmt + 0.4 * d_z, 1.0)) / 2, 0, 4),
        "staff_training_score": _sigmoid(0.8 * mgmt + rng.normal(0, 1.1, n)),
        "security_staff": np.clip(rng.gamma(2.0, 0.35, n) * np.exp(0.25 * mgmt), 0, 3),
        "incident_response_time": np.clip(rng.lognormal(np.log(10) - 0.2 * mgmt, 0.45, n), 2, 45),
        "camera_coverage": _sigmoid(0.3 + 1.0 * mgmt + rng.normal(0, 1.2, n)),
        "license_compliance": _sigmoid(1.0 + 0.9 * mgmt + rng.normal(0, 1.2, n)),
        "management_quality": mgmt,
        # log-frailty; mean-corrected so it does not shift average loss
        "unobserved_risk": rng.normal(-0.6**2 / 2, 0.6, n),
    })
    df["true_expected_loss"] = true_expected_loss(df, effects)
    df.attrs["column_sources"] = COLUMN_SOURCES
    return df


def effects_with(**overrides) -> dict:
    """Ground truth with some per-unit effects replaced, e.g. effects_with(camera_coverage=-0.06)."""
    unknown = set(overrides) - set(TRUE_FREQ_EFFECT)
    if unknown:
        raise KeyError(unknown)
    return {**TRUE_FREQ_EFFECT, **overrides}


def true_freq(df: pd.DataFrame, effects: dict | None = None) -> np.ndarray:
    e = effects or TRUE_FREQ_EFFECT
    log_lam = (np.log(BASE_FREQ)
               + e["log_capacity"] * np.log(df["capacity"] / 150)
               + e["log_nightlife_density"] * np.log(df["nightlife_density"] / 100)
               + sum(e[c] * df[c] for c in ("late_night_hours", "staff_training_score", "security_staff",
                                             "management_quality", "unobserved_risk", "camera_coverage",
                                             "license_compliance", "prior_violations")))
    return np.exp(log_lam).to_numpy()


def true_severity(df: pd.DataFrame) -> np.ndarray:
    return (BASE_SEVERITY * np.exp(TRUE_SEV_EFFECT_RESPONSE * (df["incident_response_time"] - 10))).to_numpy()


def true_expected_loss(df: pd.DataFrame, effects: dict | None = None) -> np.ndarray:
    return true_freq(df, effects) * true_severity(df)


MAX_CLAIMS = 30  # per venue-year; P(more) is negligible at these frequencies


def sample_claims(df: pd.DataFrame, rng: np.random.Generator, effects: dict | None = None) -> pd.DataFrame:
    # Fixed-size draws per venue (inverse-CDF Poisson + a fixed block of normals), so two
    # scenarios that differ only in features see the same luck ("common random numbers").
    lam = true_freq(df, effects)
    u = rng.random(len(df))
    z = rng.standard_normal((len(df), MAX_CLAIMS))
    counts = np.zeros(len(df), dtype=int)
    pmf = np.exp(-lam)
    cdf = pmf.copy()
    for k in range(1, MAX_CLAIMS + 1):
        counts += u > cdf
        pmf = pmf * lam / k
        cdf = cdf + pmf
    # lognormal with the given mean: mu = log(mean) - sigma^2/2
    mu = np.log(true_severity(df)) - SEVERITY_SIGMA**2 / 2
    sizes = np.exp(mu[:, None] + SEVERITY_SIGMA * z)
    loss = (sizes * (np.arange(MAX_CLAIMS) < counts[:, None])).sum(axis=1)
    return df.assign(claim_count=counts, loss=loss)


class Carrier:
    """Prices on the observable features only (A9-A12).

    Each renewal year it refits a Tweedie GLM to the last `window` years of claims
    and moves its coefficients `update_rate` of the way toward the new fit, standing
    in for rate filings and underwriting inertia. EXPOSURE features are rated
    directly; SCHEDULE features only set a capped credit/debit relative to the
    burn-in book average. Tempering and capping change the book's total premium, so
    the base rate is then off-balanced to the GLM's indicated level on the latest
    book. Quotes also carry a market trend and per-quote noise.
    """

    def __init__(self, load=1.35, update_rate=0.35, window=3, trend=0.06, noise_sd=0.20, seed=1):
        self.load, self.update_rate, self.window = load, update_rate, window
        self.trend, self.noise_sd = trend, noise_sd
        self.rng = np.random.default_rng(seed)
        self.scaler = None
        self.coef = self.intercept = None
        self.off_balance = 0.0
        self.history: list[pd.DataFrame] = []

    @staticmethod
    def design(df: pd.DataFrame) -> np.ndarray:
        X = df[OBSERVABLE].copy()
        X["capacity"] = np.log(X["capacity"])
        X["nightlife_density"] = np.log(X["nightlife_density"])
        return X.to_numpy(float)

    def _fit(self):
        hist = pd.concat(self.history[-self.window:])
        X = self.scaler.transform(self.design(hist))
        glm = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=2000)
        glm.fit(X, hist["loss"] / 1000)
        return glm.coef_, glm.intercept_ + np.log(1000)

    def observe_year(self, claims: pd.DataFrame):
        self.history.append(claims)
        if self.scaler is None:
            return
        c, b = self._fit()
        a = self.update_rate
        self.coef = (1 - a) * self.coef + a * c
        self.intercept = (1 - a) * self.intercept + a * b
        self._rebalance(claims)

    def initialize(self, burn_in: list[pd.DataFrame]):
        self.history = list(burn_in)
        self.scaler = StandardScaler().fit(self.design(pd.concat(burn_in)))  # frozen afterwards
        self.coef, self.intercept = self._fit()
        self._rebalance(burn_in[-1])

    def _rebalance(self, book: pd.DataFrame):
        self.off_balance = 0.0
        indicated = np.exp(self.scaler.transform(self.design(book)) @ self.coef + self.intercept).mean()
        self.off_balance = np.log(indicated / self.expected_cost(book).mean())

    def schedule_factor(self, df: pd.DataFrame) -> np.ndarray:
        # standardized features are 0 at the burn-in book mean, so 1.0 = average venue
        Z = self.scaler.transform(self.design(df))
        k = len(EXPOSURE)
        return np.clip(np.exp(SCHEDULE_WEIGHT * Z[:, k:] @ self.coef[k:]), 1 - SCHEDULE_CAP, 1 + SCHEDULE_CAP)

    def expected_cost(self, df: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(self.design(df))
        k = len(EXPOSURE)
        return np.exp(Z[:, :k] @ self.coef[:k] + self.intercept + self.off_balance) * self.schedule_factor(df)

    def quote(self, df: pd.DataFrame, year: int, noise: bool = True) -> np.ndarray:
        p = self.expected_cost(df) * self.load * (1 + self.trend) ** year
        if noise:
            p = p * self.rng.lognormal(-self.noise_sd**2 / 2, self.noise_sd, len(df))
        return p

    def per_unit_coef(self) -> pd.Series:
        """Coefficients on the raw (unstandardized) scale, comparable to TRUE_FREQ_EFFECT."""
        return pd.Series(self.coef / self.scaler.scale_, index=OBSERVABLE)


_REAL_CACHE: list = []


def build_world(n=4000, seed=0, burn_in_years=3, effects: dict | None = None):
    """Real bars -> synthetic venues -> burn-in claims -> calibrated carrier -> year-0 quotes."""
    if not _REAL_CACHE:
        _REAL_CACHE.append(load_real_bars())
    real = _REAL_CACHE[0]
    venues = make_venues(real, n=n, seed=seed, effects=effects)
    rng = np.random.default_rng(seed + 100)
    carrier = Carrier(seed=seed + 200)
    carrier.initialize([sample_claims(venues, rng, effects).assign(year=y) for y in range(-burn_in_years, 0)])
    venues["premium"] = carrier.quote(venues, year=0)
    return real, venues, carrier, rng


if __name__ == "__main__":
    pd.set_option("display.width", 160, "display.precision", 3)
    real, venues, carrier, _ = build_world()

    print("== real public data (cached in data/raw/) ==")
    print(f"bar-like NYC venues with a ZIP match: {len(real):,}")
    print("critical violations per bar, 2023-01..2026-08:",
          real["critical_violations"].describe(percentiles=[.25, .5, .75, .95]).round(1).to_dict())
    print("on-premises licenses in bar's ZIP:",
          real["nightlife_density"].describe(percentiles=[.25, .5, .75]).round(0).to_dict())
    print("most common names:", real["dba"].value_counts().head(5).to_dict())

    print("\n== synthetic venues ==")
    print(venues[OBSERVABLE + ["management_quality", "true_expected_loss", "premium"]].describe().T
          [["mean", "std", "min", "50%", "max"]])
    print("\ncorrelation with latent management_quality and with true expected loss (log):")
    corr = pd.DataFrame({
        "corr_mgmt": venues[OBSERVABLE].corrwith(venues["management_quality"]),
        "corr_log_true_loss": venues[OBSERVABLE].corrwith(np.log(venues["true_expected_loss"])),
    })
    print(corr.round(2))

    print("\n== carrier after burn-in: per-unit log coefficients vs ground-truth frequency effects ==")
    truth = {"capacity": TRUE_FREQ_EFFECT["log_capacity"],
             "nightlife_density": TRUE_FREQ_EFFECT["log_nightlife_density"],
             "incident_response_time": TRUE_SEV_EFFECT_RESPONSE}
    cmp_ = pd.DataFrame({"carrier": carrier.per_unit_coef(),
                         "truth": [truth.get(c, TRUE_FREQ_EFFECT.get(c, 0.0)) for c in OBSERVABLE]})
    print(cmp_.round(3))
    tl, prem = venues["true_expected_loss"], venues["premium"]
    print(f"\nmean true expected loss ${tl.mean():,.0f}; mean premium ${prem.mean():,.0f}; "
          f"book loss ratio {tl.sum() / prem.sum():.2f}")
    print(f"corr(log premium, log true loss) = {np.corrcoef(np.log(prem), np.log(tl))[0, 1]:.3f}")
    sf = carrier.schedule_factor(venues)
    print(f"schedule factor: mean {sf.mean():.3f}, at credit cap {(sf <= 1 - SCHEDULE_CAP + 1e-9).mean():.1%}, "
          f"at debit cap {(sf >= 1 + SCHEDULE_CAP - 1e-9).mean():.1%}")
    print(f"mean claims/venue/yr {true_freq(venues).mean():.3f}")
    from sklearn.linear_model import LinearRegression
    X = Carrier.design(venues)
    for target in ("management_quality", "true_expected_loss"):
        y = venues[target] if target == "management_quality" else np.log(venues[target])
        print(f"too-clean check: linear R2 of {target} from observables = "
              f"{LinearRegression().fit(X, y).score(X, y):.3f}")
