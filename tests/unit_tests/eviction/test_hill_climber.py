import unittest

from gptcache.manager.eviction.hill_climber import HillClimber


class TestHillClimber(unittest.TestCase):

    def test_initial_step_is_negative(self):
        """Caffeine starts by trying to shrink the window."""
        climber = HillClimber(maximum=1000)
        self.assertLess(climber.step_size, 0)
        self.assertAlmostEqual(climber.step_size, -62.5)

    def test_no_adjust_before_threshold(self):
        climber = HillClimber(maximum=1000)
        for _ in range(50):
            climber.record_hit()
        result = climber.adjust(sample_threshold=100)
        self.assertEqual(result, 0.0)

    def test_positive_hit_rate_continues_direction(self):
        climber = HillClimber(maximum=1000)
        # First sample: establish baseline (low hit rate)
        for _ in range(80):
            climber.record_miss()
        for _ in range(20):
            climber.record_hit()
        climber.adjust(sample_threshold=100)

        # Second sample: better hit rate -> keep direction
        for _ in range(50):
            climber.record_miss()
        for _ in range(50):
            climber.record_hit()
        step = climber.adjust(sample_threshold=100)
        # Initial direction was negative (shrink window), hit rate improved,
        # so we continue shrinking
        self.assertLess(step, 0)

    def test_negative_hit_rate_reverses_direction(self):
        climber = HillClimber(maximum=1000)
        # First sample: high hit rate baseline
        for _ in range(20):
            climber.record_miss()
        for _ in range(80):
            climber.record_hit()
        climber.adjust(sample_threshold=100)

        # Second sample: much worse hit rate -> reverse direction
        for _ in range(90):
            climber.record_miss()
        for _ in range(10):
            climber.record_hit()
        step = climber.adjust(sample_threshold=100)
        # Was shrinking (negative), hit rate dropped, should reverse to positive
        self.assertGreater(step, 0)

    def test_large_change_triggers_restart(self):
        climber = HillClimber(maximum=1000, restart_threshold=0.05)
        # First sample
        for _ in range(50):
            climber.record_hit()
            climber.record_miss()
        climber.adjust(sample_threshold=100)

        # Second sample: large hit rate change (workload shift)
        for _ in range(95):
            climber.record_hit()
        for _ in range(5):
            climber.record_miss()
        climber.adjust(sample_threshold=100)

        # After restart, step_size should be at full magnitude
        self.assertAlmostEqual(abs(climber.step_size), 62.5)

    def test_decay_reduces_step_size(self):
        climber = HillClimber(maximum=1000, decay_rate=0.98)
        initial_magnitude = abs(climber.step_size)

        # Small positive change: should decay
        for _ in range(50):
            climber.record_hit()
            climber.record_miss()
        climber.adjust(sample_threshold=100)

        for _ in range(52):
            climber.record_hit()
        for _ in range(48):
            climber.record_miss()
        climber.adjust(sample_threshold=100)

        self.assertLess(abs(climber.step_size), initial_magnitude)

    def test_cost_aware_mode(self):
        climber = HillClimber(maximum=1000, cost_aware=True)
        # Hits on expensive items should drive cost-hit-ratio
        for _ in range(50):
            climber.record_hit(cost=1000.0)
        for _ in range(50):
            climber.record_miss(cost=10.0)
        step = climber.adjust(sample_threshold=100)
        # Should have a very high cost-hit-ratio
        self.assertNotEqual(step, 0.0)

    def test_counters_reset_after_adjust(self):
        climber = HillClimber(maximum=1000)
        for _ in range(100):
            climber.record_hit()
        climber.adjust(sample_threshold=100)
        # Internal counters should be reset
        self.assertEqual(climber._hits_in_sample, 0)
        self.assertEqual(climber._misses_in_sample, 0)


if __name__ == "__main__":
    unittest.main()
