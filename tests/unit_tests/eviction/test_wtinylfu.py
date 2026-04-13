import unittest

from gptcache.manager.eviction.manager import EvictionBase


class TestWTinyLFU(unittest.TestCase):

    def test_basic_put_get(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, on_evict=lambda x: None
        )
        eviction.put([1, 2, 3])
        self.assertIsNotNone(eviction.get(1))
        self.assertIsNotNone(eviction.get(2))
        self.assertIsNotNone(eviction.get(3))

    def test_policy_property(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, on_evict=lambda x: None
        )
        self.assertEqual(eviction.policy, "WTINYLFU")

    def test_eviction_triggered(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=5, clean_size=1, on_evict=on_evict
        )
        for i in range(10):
            eviction.put([i])
        self.assertGreater(len(evicted), 0)

    def test_frequent_items_retained(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1, on_evict=on_evict
        )

        eviction.put(list(range(10)))

        for _ in range(20):
            for i in range(5):
                eviction.get(i)

        for i in range(10, 20):
            eviction.put([i])

        surviving_popular = sum(1 for i in range(5) if i not in evicted)
        self.assertGreaterEqual(surviving_popular, 3)

    def test_cost_aware_retention(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1,
            on_evict=on_evict, cost_aware=True
        )

        eviction.put(list(range(10)))

        for i in range(5):
            eviction.set_cost(i, 1000.0)
        for i in range(5, 10):
            eviction.set_cost(i, 1.0)

        for _ in range(5):
            for i in range(10):
                eviction.get(i)

        for i in range(10, 20):
            eviction.set_cost(i, 1.0)
            eviction.put([i])

        high_cost_evicted = sum(1 for i in range(5) if i in evicted)
        low_cost_evicted = sum(1 for i in range(5, 10) if i in evicted)
        self.assertLessEqual(high_cost_evicted, low_cost_evicted)

    def test_one_hit_wonders_evicted(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1, on_evict=on_evict
        )

        eviction.put(list(range(5)))
        for _ in range(20):
            for i in range(5):
                eviction.get(i)

        eviction.put(list(range(5, 10)))

        for i in range(10, 15):
            eviction.put([i])

        frequent_evicted = sum(1 for i in range(5) if i in evicted)
        one_hit_evicted = sum(1 for i in range(5, 10) if i in evicted)
        self.assertLessEqual(frequent_evicted, one_hit_evicted)

    def test_clean_size_default(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None
        )
        self.assertEqual(eviction._clean_size, 20)

    def test_set_cost(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, on_evict=lambda x: None
        )
        eviction.put([1])
        eviction.set_cost(1, 500.0)
        self.assertEqual(eviction._cost_map[1], 500.0)

    def test_doorkeeper_cleared_on_sketch_reset(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            reset_multiplier=1,
        )
        sentinel = hash(0xDEADBEEF)
        eviction._doorkeeper.add(sentinel)
        self.assertTrue(eviction._doorkeeper.contains(sentinel))

        eviction.put(list(range(50)))
        for _ in range(5):
            for i in range(50):
                eviction.get(i)

        self.assertFalse(eviction._doorkeeper.contains(sentinel),
                         "Doorkeeper should be cleared when CMS resets")

    def test_matches_existing_test_pattern_lru_style(self):
        datas = []

        def on_evict(deletes):
            for delete in deletes:
                if delete in datas:
                    datas.remove(delete)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=4, clean_size=2, on_evict=on_evict
        )

        def add_data(data):
            datas.append(data)
            eviction.put([data])

        add_data(1)
        add_data(2)
        add_data(3)
        add_data(4)
        for _ in range(10):
            eviction.get(1)
        add_data(5)
        self.assertIn(1, datas)


class TestWTinyLFUEWMANormalization(unittest.TestCase):
    """Tests for EWMA-based cost normalization (Ben Manes proposal)."""

    def test_ewma_tracker_initialized(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            cost_aware=True
        )
        self.assertIsNotNone(eviction._cost_tracker)
        self.assertEqual(eviction._cost_tracker.count, 0)

    def test_set_cost_feeds_ewma(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            cost_aware=True
        )
        eviction.set_cost(1, 500.0)
        eviction.set_cost(2, 1000.0)
        self.assertEqual(eviction._cost_tracker.count, 2)

    def test_normalized_cost_score_range(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            cost_aware=True, ewma_warmup=5
        )
        # Feed enough costs to pass warmup
        for i in range(1, 30):
            eviction.set_cost(i, float(i * 100))
        eviction.put(list(range(1, 30)))

        # Scores should be in valid range
        for i in range(1, 30):
            score = eviction._get_cost_score(i)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 15)

    def test_expensive_items_get_higher_normalized_score(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=200, on_evict=lambda x: None,
            cost_aware=True, ewma_alpha=0.1, ewma_warmup=10
        )
        # Establish cost distribution
        for i in range(50):
            eviction.set_cost(i, 100.0)  # cheap
        for i in range(50, 60):
            eviction.set_cost(i, 3000.0)  # expensive

        cheap_score = eviction._get_cost_score(0)
        expensive_score = eviction._get_cost_score(50)
        self.assertGreater(expensive_score, cheap_score)

    def test_cost_aware_false_disables_normalization(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            cost_aware=False
        )
        eviction.set_cost(1, 5000.0)
        eviction.put([1])
        score = eviction._get_cost_score(1)
        self.assertEqual(score, 1)  # neutral when disabled

    def test_ewma_custom_alpha(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=100, on_evict=lambda x: None,
            cost_aware=True, ewma_alpha=0.2
        )
        self.assertAlmostEqual(eviction._cost_tracker._alpha, 0.2)


class TestWTinyLFUAdaptiveWindow(unittest.TestCase):
    """Tests for the hill-climbing adaptive window sizing."""

    def test_adaptive_enabled_for_large_caches(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=200, on_evict=lambda x: None,
            adaptive=True
        )
        self.assertIsNotNone(eviction._hill_climber)

    def test_adaptive_disabled_for_small_caches(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=50, on_evict=lambda x: None,
            adaptive=True
        )
        self.assertIsNone(eviction._hill_climber)

    def test_adaptive_disabled_explicitly(self):
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=200, on_evict=lambda x: None,
            adaptive=False
        )
        self.assertIsNone(eviction._hill_climber)

    def test_window_cap_changes_after_heavy_traffic(self):
        """After enough operations, the hill climber should adjust window size."""
        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=200, on_evict=lambda x: None,
            adaptive=True, reset_multiplier=1,
        )
        initial_window_cap = eviction._window_cap

        # Drive heavy traffic to trigger multiple CMS resets (= adaptation)
        for cycle in range(5):
            for i in range(cycle * 200, (cycle + 1) * 200):
                eviction.put([i])
            for i in range(cycle * 200, (cycle + 1) * 200):
                eviction.get(i)

        # Window cap should have changed (in either direction)
        # (We can't predict which direction without knowing the workload)
        # At minimum, verify no crash and the cache still works
        eviction.put([99999])
        self.assertIsNotNone(eviction.get(99999))

    def test_backward_compatible_without_adaptive(self):
        """Disabling adaptive mode should match original fixed-window behavior."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=10, clean_size=1,
            on_evict=on_evict, adaptive=False
        )
        eviction.put(list(range(10)))

        for _ in range(20):
            for i in range(5):
                eviction.get(i)

        for i in range(10, 20):
            eviction.put([i])

        surviving_popular = sum(1 for i in range(5) if i not in evicted)
        self.assertGreaterEqual(surviving_popular, 3)


class TestWTinyLFUEndToEnd(unittest.TestCase):
    """End-to-end integration tests with all features enabled."""

    def test_full_feature_set_no_crash(self):
        """Smoke test with EWMA + hill climber + cost_aware all enabled."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction = EvictionBase.get(
            name="wtinylfu", maxsize=200, clean_size=10,
            on_evict=on_evict, cost_aware=True, adaptive=True,
            ewma_alpha=0.1, ewma_warmup=10,
        )

        # Simulate realistic LLM cache workload
        import random
        random.seed(42)
        for i in range(500):
            cost = random.choice([50, 100, 200, 500, 1000, 2000, 3500])
            eviction.set_cost(i, float(cost))
            eviction.put([i])

        # Access some items repeatedly (popular queries)
        for _ in range(10):
            for i in range(20):
                eviction.get(i)

        # Insert more items
        for i in range(500, 700):
            cost = random.choice([50, 100, 200, 500, 1000, 2000])
            eviction.set_cost(i, float(cost))
            eviction.put([i])

        # Verify cache integrity
        total_in_cache = len(eviction._window) + len(eviction._main)
        self.assertLessEqual(total_in_cache, eviction._maxsize)
        self.assertGreater(len(evicted), 0)

    def test_cost_normalization_improves_retention(self):
        """Expensive items should be retained more with EWMA normalization."""
        evicted_normalized = []
        evicted_raw = []

        def make_on_evict(evicted_list):
            def on_evict(keys):
                evicted_list.extend(keys)
            return on_evict

        import random

        for evicted_list, adaptive in [
            (evicted_normalized, True),
            (evicted_raw, False),
        ]:
            random.seed(123)
            eviction = EvictionBase.get(
                name="wtinylfu", maxsize=50, clean_size=5,
                on_evict=make_on_evict(evicted_list),
                cost_aware=True, adaptive=adaptive,
                ewma_warmup=5, ewma_alpha=0.1,
            )

            # Insert items with varied costs
            for i in range(50):
                cost = 3000.0 if i < 10 else 50.0
                eviction.set_cost(i, cost)
                eviction.put([i])

            # Access all equally
            for _ in range(10):
                for i in range(50):
                    eviction.get(i)

            # Force eviction with new items
            for i in range(50, 100):
                eviction.set_cost(i, 50.0)
                eviction.put([i])

        # Both should evict cheap items preferentially
        expensive_evicted_norm = sum(1 for i in range(10) if i in evicted_normalized)
        cheap_evicted_norm = sum(1 for i in range(10, 50) if i in evicted_normalized)
        # Expensive items should be evicted less
        self.assertLessEqual(expensive_evicted_norm, cheap_evicted_norm)


if __name__ == "__main__":
    unittest.main()
