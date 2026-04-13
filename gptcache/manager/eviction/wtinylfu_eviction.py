"""Token-Cost-Aware W-TinyLFU eviction policy for GPTCache.

Combines a frequency-based admission filter (TinyLFU with Count-Min Sketch
and Bloom filter doorkeeper) with a segmented LRU main cache and an LRU
window cache.  Weighs eviction decisions by the normalized regeneration
cost of cached entries (e.g., response token count) using EWMA-based
z-score normalization, and adaptively tunes the window/main split via
hill climbing.

Architecture:
    Window LRU (adaptive %) --> TinyLFU admission gate --> Main SLRU
                                      |                       |
                               Count-Min Sketch         Probation (20%)
                               + Bloom doorkeeper       Protected (80%)
                               + EWMA cost normalizer

Improvements over vanilla W-TinyLFU:
    - EWMA cost normalization: z-score maps costs to 0-15 discrete levels
      matching the CMS frequency range (Ben Manes, Caffeine discussion #1744)
    - Adaptive window sizing: hill climber auto-tunes window/main split
      based on cost-hit-ratio (Caffeine's BoundedLocalCache algorithm)
    - Cost temporal decay: EWMA inherently decays old cost observations
    - Similarity-score weighting: optional embedding distance factor

References:
    - TinyLFU: Gil Einziger, Roy Friedman, Ben Manes (arXiv:1512.00727)
    - Caffeine: github.com/ben-manes/caffeine
    - Theine: github.com/Yiling-J/theine
    - Lightweight Robust Size Aware Cache Management (arXiv:2105.08770)
    - Keren et al., NSDI'26: Adaptive Pipeline Cache
"""

import random
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from gptcache.manager.eviction.base import EvictionBase as EvictionBaseABC
from gptcache.manager.eviction.count_min_sketch import CountMinSketch
from gptcache.manager.eviction.doorkeeper import Doorkeeper
from gptcache.manager.eviction.ewma_cost_tracker import EWMACostTracker
from gptcache.manager.eviction.hill_climber import HillClimber
from gptcache.manager.eviction.segmented_lru import SegmentedLRU


class WTinyLFUEviction(EvictionBaseABC):
    """W-TinyLFU eviction policy with EWMA cost normalization and adaptive sizing.

    :param maxsize: total cache capacity (entry count)
    :param clean_size: number of entries to evict per batch (default 20% of maxsize)
    :param on_evict: callback receiving list of evicted entry IDs
    :param window_pct: initial window cache percentage (default 1.0)
    :param probation_pct: probation as percentage of main cache (default 20.0)
    :param cost_aware: enable cost-weighted eviction decisions (default True)
    :param adaptive: enable hill-climbing window adaptation (default True)
    :param cms_width_multiplier: CMS width = next_power_of_2(maxsize * this)
    :param reset_multiplier: CMS resets every maxsize * this increments
    :param ewma_alpha: EWMA smoothing factor for cost normalization (default 0.05)
    :param ewma_warmup: observations before EWMA normalization activates
    """

    def __init__(
        self,
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Optional[Callable[[List[Any]], None]] = None,
        window_pct: float = 1.0,
        probation_pct: float = 20.0,
        cost_aware: bool = True,
        adaptive: bool = True,
        cms_width_multiplier: int = 1,
        reset_multiplier: int = 10,
        ewma_alpha: float = 0.05,
        ewma_warmup: int = 20,
        **kwargs,
    ):
        self._maxsize = max(maxsize, 4)
        self._clean_size = clean_size if clean_size else int(self._maxsize * 0.2)
        self._on_evict = on_evict
        self._cost_aware = cost_aware

        # Segment sizes — ensure minimum viable sizes for small caches
        window_size = max(int(self._maxsize * window_pct / 100.0), 1)
        main_size = max(self._maxsize - window_size, 2)
        probation_size = max(int(main_size * probation_pct / 100.0), 1)
        protected_size = max(main_size - probation_size, 1)
        self._maxsize = window_size + probation_size + protected_size
        self._probation_pct = probation_pct

        # Data structures
        self._window = OrderedDict()
        self._window_cap = window_size
        self._main = SegmentedLRU(probation_size, protected_size)
        self._sketch = CountMinSketch(
            self._maxsize,
            width_multiplier=cms_width_multiplier,
            sample_size_multiplier=reset_multiplier,
        )
        self._doorkeeper = Doorkeeper(
            capacity=self._maxsize * reset_multiplier
        )

        # Cost metadata: entry_id -> raw cost
        self._cost_map: Dict[int, float] = {}

        # EWMA cost normalization (Ben Manes, Caffeine discussion #1744)
        self._cost_tracker = EWMACostTracker(
            alpha=ewma_alpha,
            num_levels=15,  # match CMS 4-bit range
            warmup=ewma_warmup,
        )

        # Adaptive window sizing (Caffeine's hill climber)
        self._adaptive = adaptive and self._maxsize >= 100
        self._hill_climber: Optional[HillClimber] = None
        if self._adaptive:
            self._hill_climber = HillClimber(
                maximum=self._maxsize,
                cost_aware=cost_aware,
            )

        # Track which segment each key is in for fast lookup
        self._key_location: Dict[Any, str] = {}

    def put(self, objs: List[Any]):
        """Register entry IDs after insertion into scalar/vector stores.

        Triggers eviction via the W-TinyLFU admission pipeline when the
        cache is full.
        """
        evicted = []

        for obj in objs:
            if obj in self._key_location:
                self.get(obj)
                continue

            key_hash = hash(obj)
            self._increment_sketch(key_hash)

            self._window[obj] = True
            self._window.move_to_end(obj)
            self._key_location[obj] = "window"

            while len(self._window) > self._window_cap:
                cand_key, _ = self._window.popitem(last=False)
                del self._key_location[cand_key]

                main_cap = self._maxsize - self._window_cap
                if len(self._main) < main_cap:
                    self._main.put(cand_key, True)
                    self._key_location[cand_key] = "main"
                else:
                    victim_key = self._main.peek_victim()
                    if victim_key is not None and self._admit(cand_key, victim_key):
                        self._main.evict()
                        del self._key_location[victim_key]
                        self._cost_map.pop(victim_key, None)
                        evicted.append(victim_key)
                        self._main.put(cand_key, True)
                        self._key_location[cand_key] = "main"
                        if self._hill_climber:
                            self._hill_climber.record_miss()
                    else:
                        self._cost_map.pop(cand_key, None)
                        evicted.append(cand_key)
                        if self._hill_climber:
                            self._hill_climber.record_miss()

        if evicted and self._on_evict:
            self._on_evict(evicted)

    def get(self, obj: Any):
        """Touch an entry on cache hit, updating frequency and LRU position."""
        key_hash = hash(obj)
        self._increment_sketch(key_hash)

        loc = self._key_location.get(obj)
        if loc == "window":
            if obj in self._window:
                self._window.move_to_end(obj)
                if self._hill_climber:
                    cost = self._cost_map.get(obj, 1.0)
                    self._hill_climber.record_hit(cost)
                return True
        elif loc == "main":
            result = self._main.get(obj)
            if result is not None:
                if self._hill_climber:
                    cost = self._cost_map.get(obj, 1.0)
                    self._hill_climber.record_hit(cost)
                return result

        return None

    @property
    def policy(self) -> str:
        return "WTINYLFU"

    def set_cost(self, obj_id: Any, cost: float):
        """Set the regeneration cost for a cache entry.

        The raw cost is stored per-entry and fed to the EWMA tracker
        for distribution estimation.  The EWMA normalizes costs into
        z-score-based discrete levels (0-15) for the admission decision.
        """
        self._cost_map[obj_id] = cost
        self._cost_tracker.observe(cost)

    def _increment_sketch(self, key_hash: int):
        """Doorkeeper-gated sketch increment with adaptive climb trigger."""
        if self._doorkeeper.allow(key_hash):
            self._sketch.increment(key_hash)
        if self._sketch.additions >= self._sketch._sample_size:
            # Trigger hill climber adaptation before CMS reset
            if self._hill_climber:
                self._adapt_window()
            self._sketch.reset()
            self._doorkeeper.clear()

    def _estimate_frequency(self, key: Any) -> int:
        """Estimate access frequency for a key."""
        key_hash = hash(key)
        if self._doorkeeper.contains(key_hash):
            return self._sketch.estimate(key_hash) + 1
        return 0

    def _get_cost_score(self, key: Any) -> int:
        """Get the normalized cost score for a key (0-15 discrete levels).

        When cost_aware is disabled or no cost is set, returns a neutral
        mid-range score so admission falls back to frequency-only.
        """
        if not self._cost_aware:
            return 1
        raw_cost = self._cost_map.get(key, 1.0)
        return max(self._cost_tracker.score(raw_cost), 1)

    _ADMIT_HASHDOS_THRESHOLD = 6

    def _admit(self, candidate_key: Any, victim_key: Any) -> bool:
        """W-TinyLFU admission decision: should candidate replace victim?

        Scoring uses frequency * normalized_cost_score.  Both terms are
        on the 0-15 scale (CMS frequency, EWMA z-score cost), giving a
        composite range of 0-225 with good discrimination.

        Follows Caffeine's admission policy:
        1. Candidate wins if its estimated value exceeds the victim's.
        2. When cost-aware, value = frequency * cost_score (EWMA-normalized).
        3. At high candidate frequencies (>= 6), admit with ~1/128 probability
           as a hash-DoS defence (Caffeine's ADMIT_HASHDOS_THRESHOLD).
        4. Otherwise reject — favour cache stability at low frequencies.
        """
        freq_c = self._estimate_frequency(candidate_key)
        freq_v = self._estimate_frequency(victim_key)

        if self._cost_aware:
            cost_c = self._get_cost_score(candidate_key)
            cost_v = self._get_cost_score(victim_key)
            value_c = freq_c * cost_c
            value_v = freq_v * cost_v
        else:
            value_c = freq_c
            value_v = freq_v

        if value_c > value_v:
            return True

        if freq_c >= self._ADMIT_HASHDOS_THRESHOLD:
            return random.randint(0, 127) == 0

        return False

    # -- Adaptive window sizing (Caffeine hill climber) --

    def _adapt_window(self) -> None:
        """Run one hill-climbing step to adjust window/main split."""
        adjustment = self._hill_climber.adjust(self._sketch._sample_size)
        if adjustment == 0.0:
            return

        amount = int(adjustment)
        if amount == 0:
            return

        if amount > 0:
            self._grow_window(min(amount, self._maxsize - self._window_cap - 2))
        else:
            self._shrink_window(min(-amount, self._window_cap - 1))

    def _grow_window(self, amount: int) -> None:
        """Transfer capacity from main to window."""
        if amount <= 0:
            return

        main_cap = self._maxsize - self._window_cap
        new_main_cap = max(main_cap - amount, 2)
        actual = main_cap - new_main_cap

        self._window_cap += actual
        new_prob = max(int(new_main_cap * self._probation_pct / 100.0), 1)
        new_prot = max(new_main_cap - new_prob, 1)
        self._main._probation_cap = new_prob
        self._main._protected_cap = new_prot

        # Demote excess entries from main to window if main overflows
        evicted = []
        while len(self._main) > new_main_cap:
            result = self._main.evict()
            if result is None:
                break
            demoted_key, _ = result
            if len(self._window) < self._window_cap:
                self._window[demoted_key] = True
                self._window.move_to_end(demoted_key)
                self._key_location[demoted_key] = "window"
            else:
                del self._key_location[demoted_key]
                self._cost_map.pop(demoted_key, None)
                evicted.append(demoted_key)

        if evicted and self._on_evict:
            self._on_evict(evicted)

    def _shrink_window(self, amount: int) -> None:
        """Transfer capacity from window to main."""
        if amount <= 0:
            return

        new_window_cap = max(self._window_cap - amount, 1)
        actual = self._window_cap - new_window_cap

        self._window_cap = new_window_cap
        new_main_cap = self._maxsize - new_window_cap
        new_prob = max(int(new_main_cap * self._probation_pct / 100.0), 1)
        new_prot = max(new_main_cap - new_prob, 1)
        self._main._probation_cap = new_prob
        self._main._protected_cap = new_prot

        # Move excess window entries into main probation
        evicted = []
        while len(self._window) > self._window_cap:
            moved_key, _ = self._window.popitem(last=False)
            main_cap = self._maxsize - self._window_cap
            if len(self._main) < main_cap:
                self._main.put(moved_key, True)
                self._key_location[moved_key] = "main"
            else:
                victim_key = self._main.peek_victim()
                if victim_key is not None and self._admit(moved_key, victim_key):
                    self._main.evict()
                    del self._key_location[victim_key]
                    self._cost_map.pop(victim_key, None)
                    evicted.append(victim_key)
                    self._main.put(moved_key, True)
                    self._key_location[moved_key] = "main"
                else:
                    del self._key_location[moved_key]
                    self._cost_map.pop(moved_key, None)
                    evicted.append(moved_key)

        if evicted and self._on_evict:
            self._on_evict(evicted)
