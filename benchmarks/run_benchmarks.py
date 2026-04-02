#!/usr/bin/env python3
"""CLI benchmark runner for eviction policy comparison.

Usage:
    python benchmarks/run_benchmarks.py \
        --dataset synthetic \
        --n_samples 5000 \
        --cache_sizes 100,500 \
        --policies lru,fifo,wtinylfu,wtinylfu_nocost \
        --thresholds 0.85 \
        --output results/
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.data_loader import LOADERS
from benchmarks.simulator import CacheSimulator


def _run_one_config(args_tuple):
    """Worker function for parallel benchmark execution.

    Runs in a separate process — must be a top-level function for pickling.
    """
    (cache_size, policy, threshold, entries, all_embeddings,
     embedding_model, dataset_name, warmup_factor, output_dir,
     shuffle_seed) = args_tuple

    # Shuffle entry order per repeat for statistical variation
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        perm = rng.permutation(len(entries))
        entries = [entries[i] for i in perm]
        all_embeddings = all_embeddings[perm]

    sim = CacheSimulator(
        cache_size=cache_size,
        eviction_policy=policy,
        similarity_threshold=threshold,
        embedding_model=embedding_model,
        dataset_name=dataset_name,
    )
    sim._dimension = all_embeddings.shape[1]

    warmup = warmup_factor * cache_size
    result = sim.run(entries, warmup_queries=warmup, verbose=False,
                     precomputed_embeddings=all_embeddings)

    # Save results in worker
    output_dir = Path(output_dir)
    fname = f"{dataset_name}_{policy}_cs{cache_size}_t{threshold}.json"
    result.save(output_dir / fname)
    log_fname = f"{dataset_name}_{policy}_cs{cache_size}_t{threshold}_log.json"
    result.save_log(output_dir / log_fname)

    d = result.to_dict()
    print(f"  Done: policy={policy}, size={cache_size} -> "
          f"hit_rate={d['hit_rate']:.4f}, token_save={d['token_saving_ratio']:.4f}, "
          f"wall_time={d['wall_time_seconds']:.1f}s",
          flush=True)
    return d


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",")]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",")]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="GPTCache eviction policy benchmark")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=list(LOADERS.keys()),
                        help="Dataset to use (default: synthetic)")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of samples to load (default: 5000)")
    parser.add_argument("--cache_sizes", type=str, default="100,500,1000",
                        help="Comma-separated cache sizes (default: 100,500,1000)")
    parser.add_argument("--policies", type=str,
                        default="lru,fifo,lfu,wtinylfu,wtinylfu_nocost",
                        help="Comma-separated eviction policies")
    parser.add_argument("--thresholds", type=str, default="0.85",
                        help="Comma-separated similarity thresholds (default: 0.85)")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("--warmup_factor", type=int, default=2,
                        help="Warmup queries = factor * cache_size (default: 2)")
    parser.add_argument("--profile", type=str, default=None,
                        choices=["repetitive_short", "novel_long"],
                        help="Use a predefined workload profile (overrides dataset/n_samples)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: 0 = sequential, "
                             "-1 = all CPUs)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeated trials per config for statistical "
                             "significance (default: 1)")

    args = parser.parse_args()

    cache_sizes = parse_int_list(args.cache_sizes)
    policies = parse_str_list(args.policies)
    thresholds = parse_float_list(args.thresholds)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset — profile overrides dataset/n_samples
    if args.profile:
        from benchmarks.simulator import WORKLOAD_PROFILES
        profile = WORKLOAD_PROFILES[args.profile]
        print(f"Using workload profile: {args.profile} — {profile['description']}")
        entries = LOADERS["synthetic"](
            n_samples=profile["n_samples"],
            vocabulary_size=profile["vocabulary_size"],
            zipf_param=profile["zipf_param"],
        )
        args.dataset = f"synthetic_{args.profile}"
    else:
        loader = LOADERS[args.dataset]
        entries = loader(n_samples=args.n_samples)

    # Run all combinations
    configs = list(product(cache_sizes, policies, thresholds))
    total = len(configs)
    n_repeats = args.repeats
    all_results = []

    print(f"\nRunning {total} benchmark configurations"
          f"{f' x {n_repeats} repeats' if n_repeats > 1 else ''}...")
    print(f"  Dataset: {args.dataset} ({len(entries)} entries)")
    print(f"  Cache sizes: {cache_sizes}")
    print(f"  Policies: {policies}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Embedding: {args.embedding_model}")
    print()

    # Pre-compute all embeddings once — this is the main speedup.
    print("Pre-computing embeddings (one-time batch)...")
    pre_sim = CacheSimulator(
        cache_size=1, eviction_policy="lru",
        embedding_model=args.embedding_model, dataset_name=args.dataset,
    )
    all_embeddings = pre_sim._batch_encode([e.prompt for e in entries])
    print(f"  Encoded {len(entries)} prompts -> shape {all_embeddings.shape}\n")

    n_workers = args.workers
    total_runs = total * n_repeats
    if n_workers == -1:
        n_workers = min(os.cpu_count() or 4, total_runs)
    elif n_workers == 0:
        n_workers = 1  # sequential

    # Build work items: (config, repeat_index)
    work_items = [(cs, pol, thr, rep)
                  for cs, pol, thr in configs
                  for rep in range(n_repeats)]

    raw_results = []  # all individual run results

    if n_workers > 1:
        print(f"Running {total_runs} runs across {n_workers} parallel workers...\n")
        worker_args = [
            (cs, pol, thr, entries, all_embeddings,
             args.embedding_model, args.dataset, args.warmup_factor,
             str(output_dir),
             (rep * 1000 + hash((cs, pol, thr)) % 1000) if n_repeats > 1 else None)
            for cs, pol, thr, rep in work_items
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one_config, wa): (wa, work_items[i])
                       for i, wa in enumerate(worker_args)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                d = future.result()
                _, (cs, pol, thr, rep) = futures[future]
                d["_repeat"] = rep
                raw_results.append(d)
                print(f"  [{done_count}/{total_runs}] policy={d['policy']}, "
                      f"size={d['cache_size']}"
                      f"{f', repeat={rep}' if n_repeats > 1 else ''}"
                      f" -> hit_rate={d['hit_rate']:.4f}, "
                      f"token_save={d['token_saving_ratio']:.4f}",
                      flush=True)
    else:
        for idx, (cache_size, policy, threshold, rep) in enumerate(work_items):
            print(f"[{idx+1}/{total_runs}] policy={policy}, "
                  f"cache_size={cache_size}, threshold={threshold}"
                  f"{f', repeat={rep}' if n_repeats > 1 else ''}")

            # Shuffle entry order per repeat for statistical variation
            if n_repeats > 1:
                seed = rep * 1000 + hash((cache_size, policy, threshold)) % 1000
                rng = np.random.RandomState(seed)
                perm = rng.permutation(len(entries))
                run_entries = [entries[i] for i in perm]
                run_embeddings = all_embeddings[perm]
            else:
                run_entries = entries
                run_embeddings = all_embeddings

            sim = CacheSimulator(
                cache_size=cache_size,
                eviction_policy=policy,
                similarity_threshold=threshold,
                embedding_model=args.embedding_model,
                dataset_name=args.dataset,
            )
            sim._encoder = pre_sim._encoder
            sim._dimension = all_embeddings.shape[1]

            warmup = args.warmup_factor * cache_size
            result = sim.run(run_entries, warmup_queries=warmup, verbose=not args.quiet,
                             precomputed_embeddings=run_embeddings)

            fname = f"{args.dataset}_{policy}_cs{cache_size}_t{threshold}.json"
            result.save(output_dir / fname)
            if rep == 0:
                log_fname = f"{args.dataset}_{policy}_cs{cache_size}_t{threshold}_log.json"
                result.save_log(output_dir / log_fname)
            d = result.to_dict()
            d["_repeat"] = rep
            raw_results.append(d)

            print()

    # Aggregate results: if repeats > 1, compute mean +/- std
    policy_order = {p: i for i, p in enumerate(policies)}

    if n_repeats > 1:
        grouped = defaultdict(list)
        for r in raw_results:
            key = (r["policy"], r["cache_size"], r["similarity_threshold"])
            grouped[key].append(r)

        for key, runs in grouped.items():
            agg = dict(runs[0])  # copy first run as template
            agg.pop("_repeat", None)
            metrics = ["hit_rate", "token_saving_ratio", "latency_p50",
                       "latency_p95", "latency_p99", "latency_mean",
                       "throughput_qps", "peak_memory_mb",
                       "cpu_user_seconds", "cpu_system_seconds"]
            for m in metrics:
                vals = [r[m] for r in runs if m in r]
                if vals:
                    agg[m] = float(np.mean(vals))
                    agg[f"{m}_std"] = float(np.std(vals))
                    agg[f"{m}_min"] = float(np.min(vals))
                    agg[f"{m}_max"] = float(np.max(vals))
            agg["n_repeats"] = n_repeats
            all_results.append(agg)

        all_results.sort(key=lambda r: (
            r["cache_size"], policy_order.get(r["policy"], 99),
            r["similarity_threshold"]))
    else:
        for r in raw_results:
            r.pop("_repeat", None)
        all_results = sorted(raw_results, key=lambda r: (
            r["cache_size"], policy_order.get(r["policy"], 99),
            r["similarity_threshold"]))

    # Save combined summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_dir}/")
    print(f"Summary: {summary_path}")

    # Print comparison table
    has_std = n_repeats > 1
    print("\n" + "=" * 130)
    header = (f"{'Policy':<20} {'Size':>6} {'Thresh':>7} "
              f"{'Hit Rate':>9} {'TokSave':>8} "
              f"{'p95 (ms)':>9} {'p99 (ms)':>9} {'Mean(ms)':>9} "
              f"{'QPS':>8} {'Mem(MB)':>8} {'Time(s)':>8}")
    print(header)
    print("-" * 130)
    for r in all_results:
        hr = f"{r['hit_rate']:.4f}"
        ts = f"{r['token_saving_ratio']:.4f}"
        if has_std:
            hr += f"\u00b1{r.get('hit_rate_std', 0):.4f}"
            ts += f"\u00b1{r.get('token_saving_ratio_std', 0):.4f}"
        print(f"{r['policy']:<20} {r['cache_size']:>6} {r['similarity_threshold']:>7.2f} "
              f"{hr:>16} {ts:>15} "
              f"{r['latency_p95']:>9.1f} {r['latency_p99']:>9.1f} "
              f"{r['latency_mean']:>9.2f} "
              f"{r['throughput_qps']:>8.1f} {r['peak_memory_mb']:>8.1f} "
              f"{r.get('wall_time_seconds', 0):>8.1f}")
    print("=" * 130)


if __name__ == "__main__":
    main()
