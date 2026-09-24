"""Sanity tests for the synthetic world and the stress-test metrics.

Run:  .venv\\Scripts\\python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import SCHEDULE_WEIGHT, effects_with, true_expected_loss  # noqa: E402
from score import CONTROLLABLE, promised_vs_causal  # noqa: E402
from stress import PROXIES, apply_actions, make_world, noncausal_share, run_one  # noqa: E402

W = make_world("baseline", effects_with(), seed=0)


class GroundTruth(unittest.TestCase):
    def test_causal_actions_rank_above_proxies_before_optimization(self):
        truth = promised_vs_causal(W.ss, W.venues)["true_loss_cut_%"]
        causal = [f for f in CONTROLLABLE if f not in PROXIES]
        self.assertGreater(truth[causal].min(), truth[list(PROXIES)].max())

    def test_proxy_moves_leave_true_loss_unchanged(self):
        moved = W.ss.moved(W.venues, list(PROXIES))
        np.testing.assert_allclose(true_expected_loss(moved), true_expected_loss(W.venues))

    def test_score_prices_proxies_despite_zero_effect(self):
        # the premise of the whole exercise: the score does credit the proxies
        s = W.ss.score(W.venues)
        for f in PROXIES:
            self.assertGreater(s[f"save_{f}"].mean(), 0.01)


class NonCausalMetric(unittest.TestCase):
    def _camera_credit_as_truth(self):
        credit = SCHEDULE_WEIGHT * W.carrier.per_unit_coef()["camera_coverage"]
        return effects_with(camera_coverage=credit)

    def test_near_zero_when_camera_truth_equals_carrier_credit(self):
        eff = self._camera_credit_as_truth()
        share = noncausal_share(W.carrier.expected_cost, W.venues, W.ss.moves, eff)
        self.assertLess(share["noncausal_camera_coverage"], 0.005)

    def test_near_zero_for_published_score_too(self):
        eff = self._camera_credit_as_truth()
        share = noncausal_share(W.ss.predict_premium, W.venues, W.ss.moves, eff)
        self.assertLess(share["noncausal_camera_coverage"], 0.02)

    def test_clearly_positive_when_camera_truth_is_zero(self):
        share = noncausal_share(W.carrier.expected_cost, W.venues, W.ss.moves, W.effects)
        self.assertGreater(share["noncausal_camera_coverage"], 0.05)


class Dynamics(unittest.TestCase):
    def test_no_adopters_means_no_promise_and_same_first_period_pricing(self):
        per0, _, cal0 = run_one(W, p=0.0, K=3, horizon=2)
        per1, _, _ = run_one(W, p=1.0, K=3, horizon=2)
        self.assertNotIn("promised", per0.columns)
        self.assertIsNone(cal0)
        # before the carrier has seen any post-adoption claims, its credit is identical
        self.assertAlmostEqual(per0.loc[0, "noncausal_share"], per1.loc[0, "noncausal_share"])

    def test_run_is_deterministic(self):
        a, _, _ = run_one(W, p=0.5, K=3, horizon=3)
        b, _, _ = run_one(W, p=0.5, K=3, horizon=3)
        pd.testing.assert_frame_equal(a, b)

    def test_adoption_does_not_touch_non_adopters(self):
        chosen = W.chosen & (W.u_adopt < 0.25)[:, None]
        moved = apply_actions(W.ss, W.venues, chosen)
        untouched = ~chosen.any(axis=1).to_numpy()
        pd.testing.assert_frame_equal(moved[untouched], W.venues[untouched])


if __name__ == "__main__":
    unittest.main()
