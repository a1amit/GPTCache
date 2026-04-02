"""Metrics collection and result models for cache benchmarking."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RequestMetrics:
    """Per-request metrics recorded during simulation."""
    query_idx: int
    hit: bool
    embedding_time_ms: float = 0.0
    lookup_time_ms: float = 0.0
    insert_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_saved: int = 0
    similarity_score: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a single run."""
    policy: str
    cache_size: int
    similarity_threshold: float
    dataset: str
    n_queries: int = 0
    n_hits: int = 0
    n_misses: int = 0
    total_tokens_saved: int = 0
    total_tokens_possible: int = 0
    # Latency stats (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    embedding_time_mean: float = 0.0
    lookup_time_mean: float = 0.0
    # Cache stats
    total_evictions: int = 0
    wall_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_qps: float = 0.0
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    # Per-request log
    request_log: List[RequestMetrics] = field(default_factory=list)
    # Extra params
    extra_params: Dict = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_queries if self.n_queries > 0 else 0.0

    @property
    def token_saving_ratio(self) -> float:
        return (self.total_tokens_saved / self.total_tokens_possible
                if self.total_tokens_possible > 0 else 0.0)

    def finalize(self):
        """Compute aggregate stats from the request log."""
        if not self.request_log:
            return
        self.n_queries = len(self.request_log)
        self.n_hits = sum(1 for r in self.request_log if r.hit)
        self.n_misses = self.n_queries - self.n_hits

        latencies = [r.total_time_ms for r in self.request_log]
        self.latency_p50, self.latency_p95, self.latency_p99 = percentiles(latencies)
        self.latency_mean = sum(latencies) / len(latencies) if latencies else 0

        embed_times = [r.embedding_time_ms for r in self.request_log]
        self.embedding_time_mean = sum(embed_times) / len(embed_times) if embed_times else 0

        lookup_times = [r.lookup_time_ms for r in self.request_log]
        self.lookup_time_mean = sum(lookup_times) / len(lookup_times) if lookup_times else 0

        self.total_tokens_saved = sum(r.tokens_saved for r in self.request_log if r.hit)
        self.total_tokens_possible = sum(r.tokens_saved for r in self.request_log)

        if self.wall_time_seconds > 0:
            self.throughput_qps = self.n_queries / self.wall_time_seconds

    def to_dict(self, include_log: bool = False) -> dict:
        d = asdict(self)
        d["hit_rate"] = self.hit_rate
        d["token_saving_ratio"] = self.token_saving_ratio
        if not include_log:
            d.pop("request_log", None)
        return d

    def save(self, path: Path, include_log: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(include_log=include_log), f, indent=2)

    def save_log(self, path: Path):
        """Save per-request log separately (for CDF/curve plots)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        log = [asdict(r) for r in self.request_log]
        with open(path, "w") as f:
            json.dump(log, f)


def percentiles(values: List[float]) -> tuple:
    """Compute p50, p95, p99 from a list of values."""
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    p50 = s[int(n * 0.50)]
    p95 = s[min(int(n * 0.95), n - 1)]
    p99 = s[min(int(n * 0.99), n - 1)]
    return p50, p95, p99


def sliding_window_hit_rate(request_log: List[RequestMetrics],
                            window: int = 1000) -> List[float]:
    """Compute sliding-window hit rate over the request log."""
    rates = []
    hits_in_window = 0
    for i, req in enumerate(request_log):
        hits_in_window += int(req.hit)
        if i >= window:
            hits_in_window -= int(request_log[i - window].hit)
        current_window = min(i + 1, window)
        rates.append(hits_in_window / current_window)
    return rates


class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
