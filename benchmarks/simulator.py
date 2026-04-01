"""Cache workload simulator.

Replays a sequence of (prompt, response) pairs against a GPTCache instance,
recording per-request metrics for hit rate, latency, and cost analysis.
"""

import time
import tracemalloc
from typing import Dict, List, Optional

import numpy as np

from benchmarks.data_loader import CacheEntry
from benchmarks.metrics import BenchmarkResult, RequestMetrics, Timer

# Pre-defined workload profiles for reproducible benchmarking
WORKLOAD_PROFILES: Dict[str, Dict] = {
    "repetitive_short": {
        "description": "Repetitive short prompts",
        "vocabulary_size": 100,
        "zipf_param": 1.5,
        "max_prompt_tokens": 128,
        "n_samples": 10000,
    },
    "novel_long": {
        "description": "Novel long prompts",
        "vocabulary_size": 5000,
        "zipf_param": 0.5,
        "max_prompt_tokens": 512,
        "n_samples": 10000,
    },
}

# Number of FAISS results to retrieve per search.
# Must be > 1 to handle ghost entries (soft-deleted but still in FAISS).
_SEARCH_TOP_K = 10


class CacheSimulator:
    """Simulate a cache workload by replaying dataset entries.

    :param cache_size: max number of entries in the cache
    :param eviction_policy: "lru", "fifo", "lfu", "rr", "wtinylfu", or "wtinylfu_nocost"
    :param similarity_threshold: cosine similarity threshold for cache hits (0-1)
    :param embedding_model: sentence-transformers model name for encoding prompts
    :param dataset_name: name of the dataset (for logging)
    :param extra_params: additional W-TinyLFU params (window_pct, etc.)
    """

    def __init__(
        self,
        cache_size: int = 1000,
        eviction_policy: str = "lru",
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2",
        dataset_name: str = "unknown",
        extra_params: Optional[dict] = None,
    ):
        self.cache_size = cache_size
        self.eviction_policy = eviction_policy
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.dataset_name = dataset_name
        self.extra_params = extra_params or {}
        self._encoder = None
        self._dimension = None

        # Eviction tracking
        self._eviction_count = 0

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model_name)
            test_emb = self._encoder.encode(["test"])
            self._dimension = test_emb.shape[1]
        return self._encoder

    def _encode(self, text: str) -> np.ndarray:
        encoder = self._get_encoder()
        emb = encoder.encode([text], normalize_embeddings=True)
        return emb[0].astype("float32")

    def _batch_encode(self, texts: list) -> np.ndarray:
        """Encode all texts at once — orders of magnitude faster than one-by-one."""
        encoder = self._get_encoder()
        return encoder.encode(texts, normalize_embeddings=True,
                              batch_size=256, show_progress_bar=True).astype("float32")

    def _build_cache(self):
        """Construct GPTCache with the specified eviction policy.

        CRITICAL: We let SSDataManager create the eviction object internally
        so that on_evict is wired to _clear() (which actually soft-deletes
        from SQLite and triggers FAISS cleanup). If we pre-build the eviction
        object and pass it in, _clear() is never connected and eviction
        has no effect on the data stores.
        """
        from gptcache.manager import CacheBase, VectorBase
        from gptcache.manager.data_manager import SSDataManager

        dimension = self._dimension
        policy_name = self.eviction_policy.lower()
        is_wtinylfu = policy_name.startswith("wtinylfu")

        scalar = CacheBase("sqlite", sql_url="sqlite:///:memory:")
        vector = VectorBase("faiss", dimension=dimension)

        if is_wtinylfu:
            # For W-TinyLFU, we must pre-build (GPTCache doesn't know about it)
            # but we need to wire _clear as the on_evict callback.
            # We'll create the data manager first, then build eviction with
            # its _clear method, then attach it.
            cost_aware = policy_name != "wtinylfu_nocost"

            # Create data manager with a dummy eviction first
            data_manager = SSDataManager(
                s=scalar, v=vector, o=None, e=None,
                max_size=self.cache_size, clean_size=0,
                policy="LRU",  # temporary, will be replaced
            )

            # Now create W-TinyLFU with _clear as the eviction callback
            from gptcache.manager.eviction.wtinylfu_eviction import WTinyLFUEviction
            eviction = WTinyLFUEviction(
                maxsize=self.cache_size,
                on_evict=data_manager._clear,  # THE KEY FIX
                cost_aware=cost_aware,
                **self.extra_params,
            )
            data_manager.eviction_base = eviction

            # Also lower the physical delete threshold for small caches
            data_manager.eviction_manager.MAX_MARK_COUNT = max(self.cache_size, 10)
            data_manager.eviction_manager.MAX_MARK_RATE = 0.05
        else:
            # For built-in policies, let SSDataManager wire everything
            data_manager = SSDataManager(
                s=scalar, v=vector, o=None, e=None,
                max_size=self.cache_size, clean_size=0,
                policy=policy_name.upper(),
            )
            eviction = data_manager.eviction_base

            # Lower physical delete threshold for small caches
            data_manager.eviction_manager.MAX_MARK_COUNT = max(self.cache_size, 10)
            data_manager.eviction_manager.MAX_MARK_RATE = 0.05

        return data_manager, eviction

    def run(self, entries: List[CacheEntry],
            warmup_queries: Optional[int] = None,
            verbose: bool = True,
            precomputed_embeddings: Optional[np.ndarray] = None) -> BenchmarkResult:
        """Replay entries against the cache and collect metrics.

        :param precomputed_embeddings: optional (n, dim) array of pre-encoded
            prompts — skips the encoding step entirely.
        """
        if warmup_queries is None:
            warmup_queries = 2 * self.cache_size

        # Pre-compute all embeddings in one batch (>> faster than one-by-one)
        if precomputed_embeddings is not None:
            all_embeddings = precomputed_embeddings
            self._dimension = all_embeddings.shape[1]
        else:
            if verbose:
                print(f"  Encoding {len(entries)} prompts (batch)...")
            all_embeddings = self._batch_encode([e.prompt for e in entries])

        data_manager, eviction = self._build_cache()
        is_wtinylfu = self.eviction_policy.lower().startswith("wtinylfu")

        result = BenchmarkResult(
            policy=self.eviction_policy,
            cache_size=self.cache_size,
            similarity_threshold=self.similarity_threshold,
            dataset=self.dataset_name,
            extra_params=self.extra_params,
        )

        start_wall = time.perf_counter()
        n = len(entries)
        all_request_metrics: List[RequestMetrics] = []
        next_id = 1  # track insert IDs locally instead of querying SQLite

        tracemalloc.start()

        for i, entry in enumerate(entries):
            if verbose and i > 0 and i % 1000 == 0:
                measured = all_request_metrics[warmup_queries:]
                current_hr = (sum(1 for r in measured if r.hit)
                              / len(measured)) if measured else 0
                print(f"    [{i}/{n}] hit_rate={current_hr:.3f}")

            embedding = all_embeddings[i]

            # Search cache — use top_k > 1 to skip FAISS ghost entries
            with Timer() as t_lookup:
                search_results = data_manager.search(
                    embedding, top_k=_SEARCH_TOP_K)

            hit = False
            sim_score = 0.0

            if search_results:
                for dist, entry_id in search_results:
                    if int(entry_id) == -1:
                        continue
                    l2_dist = float(dist)
                    cosine_sim = 1.0 - l2_dist / 2.0
                    if cosine_sim < self.similarity_threshold:
                        break
                    sim_score = cosine_sim
                    scalar_data = data_manager.get_scalar_data(
                        (dist, entry_id))
                    if scalar_data is not None:
                        hit = True
                        data_manager.hit_cache_callback((dist, entry_id))
                        break

            with Timer() as t_insert:
                if not hit:
                    data_manager.save(
                        entry.prompt,
                        entry.response,
                        embedding,
                    )
                    if is_wtinylfu and hasattr(eviction, "set_cost"):
                        eviction.set_cost(next_id, float(entry.response_tokens))
                    next_id += 1

            req = RequestMetrics(
                query_idx=i,
                hit=hit,
                embedding_time_ms=0.0,  # encoding done upfront
                lookup_time_ms=t_lookup.elapsed_ms,
                insert_time_ms=t_insert.elapsed_ms,
                total_time_ms=t_lookup.elapsed_ms + t_insert.elapsed_ms,
                tokens_saved=entry.response_tokens,
                similarity_score=sim_score,
            )
            all_request_metrics.append(req)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.request_log = all_request_metrics[warmup_queries:]
        result.peak_memory_mb = peak / (1024 * 1024)
        result.wall_time_seconds = time.perf_counter() - start_wall
        result.total_evictions = self._eviction_count
        result.finalize()

        data_manager.close()

        if verbose:
            print(f"  Done: hit_rate={result.hit_rate:.4f}, "
                  f"token_saving={result.token_saving_ratio:.4f}, "
                  f"wall_time={result.wall_time_seconds:.1f}s")

        return result
