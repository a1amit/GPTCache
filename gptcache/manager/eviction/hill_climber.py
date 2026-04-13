"""Adaptive window sizing via hill climbing for W-TinyLFU.

Implements the exact algorithm from Caffeine's BoundedLocalCache:
    1. Accumulate hit/miss statistics over a sample period
    2. Compute hit-rate (or cost-hit-ratio) change from previous sample
    3. If positive change: continue in the current direction (with decay)
    4. If negative change: reverse direction
    5. If large change (>= restart threshold): restart with full step size
    6. Otherwise: decay step size toward convergence

The hill climber adjusts the window-vs-main cache split to adapt to
workload characteristics without manual configuration.

References:
    - Caffeine: BoundedLocalCache.java, determineAdjustment()
    - Keren et al., NSDI'26: Sampled Hill Climber variant
"""


class HillClimber:
    """Adaptive window sizing for W-TinyLFU.

    :param maximum: total cache capacity (determines step granularity)
    :param cost_aware: optimize cost-hit-ratio instead of hit rate
    :param step_percent: initial step as fraction of capacity (default 6.25%)
    :param decay_rate: step decay per adaptation cycle (default 0.98)
    :param restart_threshold: hit-rate change that triggers restart (default 5%)
    """

    def __init__(
        self,
        maximum: int,
        cost_aware: bool = False,
        step_percent: float = 0.0625,
        decay_rate: float = 0.98,
        restart_threshold: float = 0.05,
    ):
        self._maximum = maximum
        self._cost_aware = cost_aware
        self._step_percent = step_percent
        self._decay_rate = decay_rate
        self._restart_threshold = restart_threshold

        # Caffeine starts by trying to shrink the window (negative step)
        self._step_size = -step_percent * maximum

        self._previous_hit_rate = 0.0
        self._hits_in_sample = 0
        self._misses_in_sample = 0
        self._cost_hits = 0.0
        self._cost_total = 0.0

    def record_hit(self, cost: float = 1.0) -> None:
        """Record a cache hit with optional cost weight."""
        self._hits_in_sample += 1
        if self._cost_aware:
            self._cost_hits += cost
            self._cost_total += cost

    def record_miss(self, cost: float = 1.0) -> None:
        """Record a cache miss with optional cost weight."""
        self._misses_in_sample += 1
        if self._cost_aware:
            self._cost_total += cost

    def adjust(self, sample_threshold: int) -> float:
        """Compute window adjustment after a sample period.

        Returns the signed adjustment (positive = grow window,
        negative = shrink window).  Returns 0.0 if the sample period
        has not yet completed.

        :param sample_threshold: minimum operations before adapting
        """
        total = self._hits_in_sample + self._misses_in_sample
        if total < sample_threshold:
            return 0.0

        # Compute objective: cost-hit-ratio or plain hit rate
        if self._cost_aware and self._cost_total > 0:
            hit_rate = self._cost_hits / self._cost_total
        elif total > 0:
            hit_rate = self._hits_in_sample / total
        else:
            hit_rate = 0.0

        hit_rate_change = hit_rate - self._previous_hit_rate
        amount = self._step_size

        # Determine direction: keep going or reverse
        if hit_rate_change >= 0:
            step = amount
        else:
            step = -amount

        # Restart or decay
        if abs(hit_rate_change) >= self._restart_threshold:
            # Workload shift: restart with full step, preserving direction
            self._step_size = (
                self._step_percent * self._maximum
                * (1.0 if step >= 0 else -1.0)
            )
        else:
            # Converging: decay step
            self._step_size = self._decay_rate * step

        self._previous_hit_rate = hit_rate
        self._reset_counters()

        return step

    def _reset_counters(self) -> None:
        self._hits_in_sample = 0
        self._misses_in_sample = 0
        self._cost_hits = 0.0
        self._cost_total = 0.0

    @property
    def step_size(self) -> float:
        return self._step_size
