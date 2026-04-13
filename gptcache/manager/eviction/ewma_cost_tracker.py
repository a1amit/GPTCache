"""EWMA-based cost normalization for W-TinyLFU admission decisions.

Tracks the running cost distribution using Exponential Weighted Moving
Average (EWMA) and maps individual costs to discrete integer scores via
z-score normalization.  This ensures the cost term is on the same 0-15
scale as the Count-Min Sketch frequency estimate, preventing raw cost
magnitudes from dominating the admission comparison.

Design follows Ben Manes' proposal (Caffeine discussion #1744):
    1. Log-transform costs to compress heavy-tailed token-count distributions
    2. Track mean and variance via EWMA
    3. Compute z-score, clamp to [-1, +1]
    4. Map to discrete integer in [0, num_levels]

References:
    - Caffeine discussion #1744: github.com/ben-manes/caffeine/discussions/1744
    - EWMA models: Boyd & Vandenberghe, Stanford
"""

import math


class EWMACostTracker:
    """Tracks cost distribution and provides normalized discrete scores.

    :param alpha: EWMA smoothing factor (default 0.05 ≈ 20-sample window)
    :param num_levels: number of discrete score levels (default 15, matching CMS)
    :param warmup: observations before normalization activates (default 20)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        num_levels: int = 15,
        warmup: int = 20,
    ):
        self._alpha = alpha
        self._num_levels = num_levels
        self._warmup = warmup
        self._count = 0
        self._mean = 0.0
        self._var = 0.0

    def observe(self, cost: float) -> None:
        """Update running statistics with a new cost observation.

        Costs are log-transformed before EWMA to handle the heavy-tailed
        distribution of LLM token counts (typically 10-4000+).
        """
        x = math.log(max(cost, 1.0))
        if self._count == 0:
            self._mean = x
            self._var = 0.0
        else:
            delta = x - self._mean
            self._mean += self._alpha * delta
            # Welford-style EWMA variance update
            self._var = (1.0 - self._alpha) * (self._var + self._alpha * delta * delta)
        self._count += 1

    def score(self, cost: float) -> int:
        """Return a discrete score in [0, num_levels] for the given cost.

        Higher cost -> higher score -> more valuable to retain in cache.
        During warmup (insufficient observations), returns a neutral
        mid-range score so all entries are treated equally.
        """
        if self._count < self._warmup or self._var <= 1e-12:
            return self._num_levels >> 1  # neutral

        x = math.log(max(cost, 1.0))
        sigma = math.sqrt(self._var)
        z = (x - self._mean) / sigma
        z = max(-1.0, min(1.0, z))  # clamp per Ben Manes' spec
        return round((z + 1.0) * self._num_levels / 2.0)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        """Current EWMA mean (in log-space)."""
        return self._mean

    @property
    def variance(self) -> float:
        """Current EWMA variance (in log-space)."""
        return self._var

    @property
    def is_warmed_up(self) -> bool:
        return self._count >= self._warmup
