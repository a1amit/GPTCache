import math
import unittest

from gptcache.manager.eviction.ewma_cost_tracker import EWMACostTracker


class TestEWMACostTracker(unittest.TestCase):

    def test_warmup_returns_neutral_score(self):
        tracker = EWMACostTracker(warmup=20)
        for i in range(1, 15):
            tracker.observe(float(i * 100))
        # Still in warmup — should return neutral mid-range (7)
        self.assertEqual(tracker.score(500.0), 7)
        self.assertFalse(tracker.is_warmed_up)

    def test_after_warmup_normalizes(self):
        tracker = EWMACostTracker(alpha=0.1, warmup=10)
        # Feed costs centered around 100
        for _ in range(30):
            tracker.observe(100.0)
        self.assertTrue(tracker.is_warmed_up)
        # A cost of 100 (the mean) should score near the midpoint
        score = tracker.score(100.0)
        self.assertGreaterEqual(score, 6)
        self.assertLessEqual(score, 9)

    def test_high_cost_scores_high(self):
        tracker = EWMACostTracker(alpha=0.1, warmup=10)
        # Establish distribution: mostly 100, some 200
        for _ in range(50):
            tracker.observe(100.0)
        for _ in range(10):
            tracker.observe(200.0)
        # A cost of 4000 (very expensive) should score high
        high_score = tracker.score(4000.0)
        low_score = tracker.score(10.0)
        self.assertGreater(high_score, low_score)
        self.assertGreaterEqual(high_score, 12)
        self.assertLessEqual(low_score, 3)

    def test_log_transform_compresses_range(self):
        tracker = EWMACostTracker(alpha=0.1, warmup=5)
        # Feed wide range of costs
        for cost in [10, 50, 100, 500, 1000, 2000, 4000]:
            tracker.observe(cost)
            tracker.observe(cost)
            tracker.observe(cost)
        # Scores should be spread across levels, not clustered
        scores = [tracker.score(c) for c in [10, 100, 1000, 4000]]
        self.assertGreater(scores[-1], scores[0])
        # Should span at least 5 levels
        self.assertGreaterEqual(scores[-1] - scores[0], 5)

    def test_zero_cost_handled(self):
        tracker = EWMACostTracker(warmup=5)
        for _ in range(10):
            tracker.observe(100.0)
        # cost=0 should not crash and should score low
        score = tracker.score(0.0)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 15)

    def test_uniform_costs_give_neutral_scores(self):
        tracker = EWMACostTracker(alpha=0.1, warmup=10)
        # All same cost -> variance ≈ 0 -> neutral scores
        for _ in range(30):
            tracker.observe(500.0)
        score = tracker.score(500.0)
        self.assertEqual(score, 7)  # neutral mid-range

    def test_score_range_is_bounded(self):
        tracker = EWMACostTracker(alpha=0.1, warmup=5)
        for _ in range(20):
            tracker.observe(100.0)
        # Even extreme values should be clamped to [0, 15]
        self.assertGreaterEqual(tracker.score(0.001), 0)
        self.assertLessEqual(tracker.score(0.001), 15)
        self.assertGreaterEqual(tracker.score(1_000_000.0), 0)
        self.assertLessEqual(tracker.score(1_000_000.0), 15)

    def test_ewma_properties(self):
        tracker = EWMACostTracker(alpha=0.1)
        self.assertEqual(tracker.count, 0)
        tracker.observe(100.0)
        self.assertEqual(tracker.count, 1)
        self.assertAlmostEqual(tracker.mean, math.log(100.0))

    def test_first_observation_sets_mean(self):
        tracker = EWMACostTracker()
        tracker.observe(200.0)
        self.assertAlmostEqual(tracker.mean, math.log(200.0))
        self.assertAlmostEqual(tracker.variance, 0.0)


class TestEWMAIntegrationWithLLMCosts(unittest.TestCase):
    """Test with realistic LLM token count distributions."""

    def test_bimodal_distribution(self):
        """LLM responses are often bimodal: short (50-200) and long (1000-3000)."""
        tracker = EWMACostTracker(alpha=0.05, warmup=20)
        # Warmup with mixed costs
        for _ in range(30):
            tracker.observe(100.0)
        for _ in range(30):
            tracker.observe(2000.0)

        short_score = tracker.score(100.0)
        long_score = tracker.score(2000.0)
        self.assertGreater(long_score, short_score)

    def test_drifting_workload(self):
        """Costs shift over time — EWMA should adapt."""
        tracker = EWMACostTracker(alpha=0.1, warmup=10)
        # Phase 1: cheap responses
        for _ in range(50):
            tracker.observe(50.0)
        cheap_score_at_50 = tracker.score(50.0)

        # Phase 2: expensive responses
        for _ in range(50):
            tracker.observe(3000.0)
        cheap_score_after_shift = tracker.score(50.0)

        # After shift, 50 tokens should score lower (it's now below average)
        self.assertGreater(cheap_score_at_50, cheap_score_after_shift)


if __name__ == "__main__":
    unittest.main()
