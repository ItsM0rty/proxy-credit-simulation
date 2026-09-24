# Does a savings score survive venues optimizing toward it?

Everything below comes from a synthetic simulation. Nothing here is a claim about Third
Space's product, any carrier, or real NYC premiums.

## Question

Third Space (thirdspacerisk.com, read 2026-09-24) describes its Savings Score as
carrier pricing "reverse-engineered … into a transparent score. See exactly how
your actions move your rate". It says it has "lowered premiums up to 20%". If a
score is defined as the predicted premium reduction from changing operational
features, how much of its signal survives once venues act on it?

## Method

The world (`src/data.py`) has 4,000 synthetic venues, each anchored on a real NYC bar
(1,359 from DOHMH inspections, joined to NY SLA licenses by ZIP). A hidden management
quality drives true loss and also makes well-run venues install cameras and keep their
paperwork tidy. Cameras and paperwork are therefore proxies, with zero causal effect in
the baseline. Training, security staff, response time and late-night hours are causal.

The carrier fits a Tweedie GLM to 3 years of claims on observable features only. It
moves 35% of the way toward each refit, prices operational features through a ±25%
schedule credit (applied at 40% strength) and off-balances the base rate.

The score (`src/score.py`) is a monotone gradient-boosting model fit to observed quotes.
The Savings Score is the predicted premium cut from moving each controllable feature to
the book's 75th percentile, or closing one hour earlier.

In the stress test (`src/stress.py`), a random fraction p ∈ {0, 10, 25, 50, 100%} of
venues make their top-3 recommended actions at t=1. True loss changes only through the
causal channel. The carrier keeps its learning rule for 10 periods, with a claims window
K ∈ {1, 3, 5}. The true effect of cameras, and separately of paperwork, is swept at
{0, 10, 30%} of training's effect. All results are over 20 seeds, reported as
median [min-max]. There are 9 unit tests, including one showing the non-causal metric
reads about 0 when the camera's true effect equals the carrier's credit.

## What is real vs synthetic

| real (public, cached in `data/raw/`) | synthetic |
|---|---|
| critical health violations per bar (NYC DOHMH); on-premises liquor licenses per ZIP (NYS SLA) | every other feature, both latent variables, true loss, claims, carrier, premiums, adoption |

## Findings (baseline, K = 3)

The robustness checks for these findings are in [`audit/AUDIT.md`](audit/AUDIT.md). "The grid"
below is the audit's sweep of 25 settings: credit cap {0.15, 0.25, 0.35, 0.50, none} ×
tempering weight {0.2, 0.4, 0.6, 0.8, 1.0}. The baseline is cap 0.25, weight 0.4.

1. About a quarter of the score's advice has no loss behind it. Before anyone acts,
   24.5% [18.6-36.2%] of the summed per-action promises come from cameras and paperwork,
   whose true effect is 0 (fig 1). Across the grid this share stays between 20% and 30%.
   The bundle as a whole is not reliably calibrated. Its promised-minus-true gap runs from
   about −8 to +25 pp across the grid and is within 1 pp only at the chosen setting.
   Whether the schedule under-credits the real risk reducers also depends on the tuning. It
   does only for tight schedules (cap ≤ 0.25 or weight ≤ 0.4). With looser ones, causal
   features are over-credited too, because the carrier's causal coefficients are also
   confounded.
2. The promise decays, and it decays most at partial adoption. At p = 50%, adopters are
   promised 27.2% and realize 24.9% at t=1. By t=10 they realize 20.0% [17.2-24.7%], while
   their true loss fell 25.9% (promise kept: 0.91 → 0.73 [0.54-1.03], and 0.64 to 0.83
   across the grid). At p = 100% promise kept returns to 0.96 (fig 2, left). That does not
   mean the score has become honest. The rise starts at t = K + 1, when the carrier's claims
   window holds only post-adoption claims. With no unmoved venue-years left, the refit
   re-learns the confounded slope: the training coefficient reaches about −1.54 at t=10,
   against a truth of −0.60. So 0.96 reflects over-crediting, and it coincides with the
   worst loss ratio. Base-rate rebalancing plays no part, because it cancels out of the
   metric. The 20% landing near Third Space's "up to 20%" is a coincidence of my
   calibration and says nothing about their claim.
3. Adopters who took mostly proxy actions are over-rewarded relative to those who took
   causal ones. Adopters whose recommended actions were mostly proxies (7.3% of adopters)
   get a premium cut 9.8 pp [5.3-16.8] larger than their true loss cut at t=1, and 8.5 pp
   [2.5-16.0] larger at t=10. Adopters who took only causal actions get 7.2 pp [3.2-12.1]
   less than they earned (fig 2, right). The gap between the two groups is about 17 pp here
   and 17-24 pp across the grid. The separate magnitudes are fragile. The proxy-heavy
   over-reward is positive at every setting but ranges from 6 to 36 pp, and the causal-only
   under-reward flips sign under looser schedules.
4. The carrier does not correct the proxy credit unless it observes management quality
   closely. Its per-unit camera estimate is −0.50 at t=1 and −0.46 [−0.77, −0.16] at t=10,
   against a truth of 0. The non-causal share of the carrier's credit is 0.22 at t=1 and
   0.20-0.25 at t=10, and 0.22 to 0.27 across the grid. The problem is identification: a
   cross-sectional refit cannot separate the proxy from the unobserved management quality
   it stands in for. The audit tests this by giving the carrier a noisy proxy for
   management quality. With a weak proxy, correction fails: at correlation 0.71 the camera
   credit stays at −0.35 to −0.43. Correction needs a correlation of about 0.89 or higher,
   and then takes about 4 to 6 periods, a lag set by the 35% partial-update rule. My
   original tolerance band (±0.012) was tighter than the estimator's noise. A carrier that
   observes management quality exactly meets it within 10 periods in only 7 of 20 seeds,
   and from −0.50 it takes at least 9 partial updates to reach. The window length barely
   changes the level of the camera credit, but it sets when the p = 100% rebound in
   Finding 2 starts.
5. Proxy strength sweep (fig 3). If cameras truly cut loss by 10% or 30% of training's
   effect, the non-causal share at t=10 falls from 0.25 to 0.18 and 0.13. Paperwork gives
   0.25, 0.20 and 0.14. The carrier over-credits by about 0.5 per unit regardless of the truth
   (camera credit at t=1: −0.50, −0.58, −0.70 against a truth of 0, −0.06, −0.18).
6. Ranking holds up. Who pays for the mispricing depends on a modeling choice (fig 4).
   Spearman(score, true loss cut) is 0.69 [0.58-0.76] before and 0.69-0.77 after adoption
   at every p; none exceeds 0.9. The audit did not test this. The rest of this finding
   depends on base-rate rebalancing (A12). With rebalancing on, as in the baseline, the
   carrier's peak loss ratio rises from 0.76 (p = 0) to 0.84 (50%) and 1.06 [0.96-1.11]
   (100%), against a target of 0.74. That rise holds across the grid. Non-adopters' rates
   fall 5.4 pp relative to p = 0, so the carrier's book absorbs the mispricing. With the
   off-balance frozen after t=0, this reverses. The peak loss ratio falls with adoption
   (0.76 → 0.69), and non-adopters' rates rise 14.1 pp at p = 50%, so non-adopters pay. The
   direction of the transfer is set by the rebalancing rule, not by the score.

In short: the score's ranking holds up (Spearman about 0.7), but its magnitudes and its
attribution to specific actions do not. About 20-30% of its per-action promises come from
actions with no effect on loss, at every cap and tempering setting tested. About 20-27% of
the carrier's credit is unbacked, and it is not corrected unless the carrier observes a
close proxy (correlation about 0.9 or better) of the hidden management quality. Whether the
bundle looks calibrated, and whether the carrier or the non-adopters pay, depend on tuning
and rebalancing choices.

## Limitations

- The effect sizes and the data-generating process are my choices (appendix). The
  non-correction result follows from a cross-sectional carrier plus an unobserved confounder.
  That is realistic, but it is built in.
- There is one carrier, actions are costless, adopters are random, adoption is one-shot, and
  the horizon is 10 periods. A real venue can switch carriers, which Third Space's
  multi-market shopping makes likely.
- The non-causal metric is one-sided and counts proxies only. Over-credit of late-night
  hours is not counted.
- Seed 0 (the Step 2 world) is the least-calibrated seed at t=1: promise kept 0.83, while
  the other 19 seeds range 0.85-0.95, so single-seed Step 2 numbers are not representative
  on that metric. The too-clean check raised 5 such seed-0 flags out of 32 checks (about 3
  expected by chance, and the four t=1 flags share one carrier), and none for Spearman > 0.9
  or zero spread.

## What I'd do with real data

1. Compare within venues. For clients who changed a feature (installed camera backups,
   added training), compare renewal quotes and loss runs before vs after against
   matched non-changers (difference-in-differences). This is the real "promise kept" test.
2. Use staggered rollouts. The order in which the broker rolls out camera backups or
   incident files is close to a natural experiment on severity.
3. Track per-action calibration at every renewal. For each action, compare the promised
   rate change with the realized rate change and the loss-run change, and flag actions
   whose promises are consistently unbacked, as the proxies are here.
4. Add public outcome proxies for incident frequency (NYPD complaints near the venue,
   311 noise complaints, SLA disciplinary actions) to test which levers track incidents.

---

## Appendix: every assumption

### Data (step 1)
- A1. Bars identified by name (BAR/PUB/LOUNGE/TAVERN/SALOON/TAPROOM/BIERGARTEN/NIGHTCLUB minus JUICE/MILK/BAGEL/…).
- A2. Health-code critical violations stand in for "prior violations".
- A3. Nightlife density = currently active on-premises licenses in the ZIP.
- A4. No public venue-level premium/loss data exists; all premiums and losses are synthetic.

### Ground truth
- A5. Per-unit frequency effects: training −0.60, security −0.25, late night +0.08/h, management −0.45/SD, unobserved risk σ=0.6. Severity: response time +2%/min.
- A6. Baseline camera and paperwork effects are 0. The sweep uses 10% and 30% of training's effect (−0.06, −0.18 per unit; same 0-1 scale), applied to frequency and to the whole world including burn-in.
- A7. prior_violations has no direct effect.
- A8. 0.25 claims/yr at capacity 150, $30k mean claim (lognormal σ=1). Order-of-magnitude choices.

### Carrier
- A9. It sees all nine observable features (via an enriched submission).
- A10. Tweedie GLM (p=1.5); coefficients move 35% toward each refit; loading 1.35; trend 6%/yr; quote noise σ=0.20.
- A11. Operational features enter only via a schedule credit capped at ±25%, applied at 40% of the GLM indication.
- A12. (Added after the first run.) The base rate is off-balanced to the GLM's indicated level on the latest book. It was added because, in an earlier version of the code, capping plus tempering made premiums drift to about 2× losses after adoption without it. That does not reproduce in the current code: with the off-balance frozen, premiums are about 1.45× true loss at p = 100%, against the 1.35 load. Rebalancing does decide who pays for the mispricing (Finding 6). It moved the Step 1 book loss ratio from 0.90 to 0.74.
- A13. (Added after the first run.) Claims use common random numbers: fixed draws per venue-period, so p-scenarios share luck. This shifted Step 2 single-seed numbers slightly.
- A14. (Added after the first run.) The window K applies from t=1. The initial carrier always uses 3 burn-in years, so every K starts from the same carrier. The carrier gets no adoption flag and does not use within-venue changes.

### Score
- A15. Targets: 75th percentile of the book (25th for response time), or close one hour earlier.
- A16. Monotone constraints on controllable features.
- A17. The score is fit on all 4,000 t=0 quotes.

### Stress test
- A18. A random cohort (nested across p) adopts once at t=1, following the t=0 published score. There is no self-selection and no re-optimization.
- A19. Each adopter takes its top-3 actions by predicted single-action saving (skipping any < 0.5%). Actions cost nothing, hit their targets exactly, and persist. The latent variables do not change.
- A20. Realized premium change = the carrier's same-period rate for new vs old features (noise-free, trend-free).
- A21. Non-causal share = Σ over proxies of max(premium cut − true loss cut, 0) ÷ Σ over all actions of premium cut. It is evaluated on the t=0 population with the carrier's period-t pricing, i.e. what a perfectly re-fit score would show.
- A22. Time to correction compares the carrier's GLM per-unit estimate (not the 40%-tempered credit) to the truth. The band is 20% of |truth|, floored at 0.012 (20% of the smallest non-zero swept effect) because a band around 0 is otherwise empty. Horizon: 10 periods.
- A23. "Spearman after" re-fits the score on period-10 quotes of the shifted population and evaluates its new recommendations.
- A24. Proxy-heavy = at least 2 of 3 chosen actions are proxies; causal-only = none.
- A25. The 20 seeds are 20 different synthetic worlds (venues, claims, carrier noise); the real anchors are shared.
