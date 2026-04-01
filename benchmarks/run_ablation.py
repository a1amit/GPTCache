#!/usr/bin/env python3
"""Ablation study and parameter sweep for W-TinyLFU.

Runs three sets of experiments:
1. Workload profiles: repetitive_short vs novel_long
2. Component ablation: LRU vs W-TinyLFU vs W-TinyLFU+Cost
3. Parameter sweep: window_pct effect

Usage:
    python benchmarks/run_ablation.py --output results_ablation/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.data_loader import load_synthetic_zipf
from benchmarks.simulator import CacheSimulator, WORKLOAD_PROFILES


def run_experiment(name, entries, cache_size, policy, threshold,
                   extra_params=None, dataset_name="ablation",
                   precomputed_embeddings=None, shared_encoder=None):
    """Run a single experiment and return the result dict."""
    sim = CacheSimulator(
        cache_size=cache_size,
        eviction_policy=policy,
        similarity_threshold=threshold,
        dataset_name=dataset_name,
        extra_params=extra_params or {},
    )
    if shared_encoder is not None:
        sim._encoder = shared_encoder
        sim._dimension = shared_encoder.get_sentence_embedding_dimension()
    result = sim.run(entries, verbose=False,
                     precomputed_embeddings=precomputed_embeddings)
    d = result.to_dict()
    d["experiment"] = name
    return d, sim._encoder


def main():
    parser = argparse.ArgumentParser(description="Ablation study for W-TinyLFU")
    parser.add_argument("--output", type=str, default="results_ablation",
                        help="Output directory")
    parser.add_argument("--cache_size", type=int, default=20,
                        help="Cache size for ablation (default: 20)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cs = args.cache_size
    all_results = []
    shared_encoder = None

    # =========================================================
    # Experiment 1: Workload Profiles
    # =========================================================
    print("=" * 60)
    print("Experiment 1: Workload Profiles")
    print("=" * 60)

    for profile_name, profile in WORKLOAD_PROFILES.items():
        print(f"\n  Profile: {profile_name} — {profile['description']}")
        entries = load_synthetic_zipf(
            n_samples=profile["n_samples"],
            vocabulary_size=profile["vocabulary_size"],
            zipf_param=profile["zipf_param"],
        )
        # Pre-compute embeddings for this profile
        _tmp = CacheSimulator(cache_size=1, eviction_policy="lru",
                              dataset_name=f"profile_{profile_name}")
        if shared_encoder:
            _tmp._encoder = shared_encoder
        embeddings = _tmp._batch_encode([e.prompt for e in entries])
        if not shared_encoder:
            shared_encoder = _tmp._encoder

        for policy in ["lru", "wtinylfu", "wtinylfu_nocost"]:
            print(f"    Running {policy}...", end=" ", flush=True)
            r, shared_encoder = run_experiment(
                name=f"profile_{profile_name}",
                entries=entries,
                cache_size=cs,
                policy=policy,
                threshold=0.85,
                dataset_name=f"profile_{profile_name}",
                precomputed_embeddings=embeddings,
                shared_encoder=shared_encoder,
            )
            print(f"hit_rate={r['hit_rate']:.4f} token_save={r['token_saving_ratio']:.4f}")
            all_results.append(r)

    # =========================================================
    # Experiment 2: Component Ablation
    # =========================================================
    print("\n" + "=" * 60)
    print("Experiment 2: Component Ablation (isolating cost-awareness)")
    print("=" * 60)

    entries = load_synthetic_zipf(n_samples=5000, vocabulary_size=500, zipf_param=0.7)
    _tmp = CacheSimulator(cache_size=1, eviction_policy="lru",
                          dataset_name="ablation_component")
    _tmp._encoder = shared_encoder
    embeddings = _tmp._batch_encode([e.prompt for e in entries])

    for policy in ["lru", "fifo", "lfu", "wtinylfu_nocost", "wtinylfu"]:
        print(f"  Running {policy}...", end=" ", flush=True)
        r, shared_encoder = run_experiment(
            name="ablation_component",
            entries=entries,
            cache_size=cs,
            policy=policy,
            threshold=0.85,
            dataset_name="ablation_component",
            precomputed_embeddings=embeddings,
            shared_encoder=shared_encoder,
        )
        print(f"hit_rate={r['hit_rate']:.4f} token_save={r['token_saving_ratio']:.4f}")
        all_results.append(r)

    # =========================================================
    # Experiment 3: Parameter Sweep — window_pct
    # =========================================================
    print("\n" + "=" * 60)
    print("Experiment 3: Window Size Sweep")
    print("=" * 60)

    # Reuse embeddings from experiment 2 (same entries)
    for window_pct in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        print(f"  window_pct={window_pct}%...", end=" ", flush=True)
        r, shared_encoder = run_experiment(
            name=f"sweep_window_{window_pct}",
            entries=entries,
            cache_size=cs,
            policy="wtinylfu",
            threshold=0.85,
            extra_params={"window_pct": window_pct},
            dataset_name="sweep_window",
            precomputed_embeddings=embeddings,
            shared_encoder=shared_encoder,
        )
        r["window_pct"] = window_pct
        print(f"hit_rate={r['hit_rate']:.4f} token_save={r['token_saving_ratio']:.4f}")
        all_results.append(r)

    # =========================================================
    # Experiment 4: Cache Size Sweep (for ablation context)
    # =========================================================
    print("\n" + "=" * 60)
    print("Experiment 4: Cache Size Sweep (W-TinyLFU+Cost vs LRU)")
    print("=" * 60)

    for cache_sz in [10, 20, 50, 100, 200]:
        for policy in ["lru", "wtinylfu"]:
            print(f"  cache_size={cache_sz}, {policy}...", end=" ", flush=True)
            r = run_experiment(
                name=f"sweep_cachesize_{cache_sz}",
                entries=entries,
                cache_size=cache_sz,
                policy=policy,
                threshold=0.85,
                dataset_name="sweep_cachesize",
            )
            r["sweep_cache_size"] = cache_sz
            print(f"hit_rate={r['hit_rate']:.4f} token_save={r['token_saving_ratio']:.4f}")
            all_results.append(r)

    # =========================================================
    # Save all results
    # =========================================================
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll ablation results saved to {summary_path}")

    # Print summary tables
    print("\n" + "=" * 60)
    print("WORKLOAD PROFILES")
    print("=" * 60)
    print(f"{'Profile':<25} {'Policy':<20} {'HitRate':>9} {'TokenSave':>10}")
    print("-" * 65)
    for r in all_results:
        if r["experiment"].startswith("profile_"):
            profile = r["experiment"].replace("profile_", "")
            print(f"{profile:<25} {r['policy']:<20} {r['hit_rate']:>9.4f} {r['token_saving_ratio']:>10.4f}")

    print("\n" + "=" * 60)
    print("COMPONENT ABLATION")
    print("=" * 60)
    print(f"{'Policy':<20} {'HitRate':>9} {'TokenSave':>10} {'vs LRU Hit':>12}")
    print("-" * 55)
    lru_hr = next((r["hit_rate"] for r in all_results
                   if r["experiment"] == "ablation_component" and r["policy"] == "lru"), 0)
    for r in all_results:
        if r["experiment"] == "ablation_component":
            delta = ((r["hit_rate"] - lru_hr) / lru_hr * 100) if lru_hr > 0 else 0
            print(f"{r['policy']:<20} {r['hit_rate']:>9.4f} {r['token_saving_ratio']:>10.4f} {delta:>+11.1f}%")

    print("\n" + "=" * 60)
    print("WINDOW SIZE SWEEP")
    print("=" * 60)
    print(f"{'Window %':<12} {'HitRate':>9} {'TokenSave':>10}")
    print("-" * 35)
    for r in all_results:
        if r["experiment"].startswith("sweep_window_"):
            print(f"{r.get('window_pct', '?'):<12} {r['hit_rate']:>9.4f} {r['token_saving_ratio']:>10.4f}")

    print("\n" + "=" * 60)
    print("CACHE SIZE SWEEP")
    print("=" * 60)
    print(f"{'CacheSize':<12} {'Policy':<20} {'HitRate':>9} {'TokenSave':>10}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: (x.get("sweep_cache_size", 0), x.get("policy", ""))):
        if r["experiment"].startswith("sweep_cachesize_"):
            print(f"{r.get('sweep_cache_size', '?'):<12} {r['policy']:<20} "
                  f"{r['hit_rate']:>9.4f} {r['token_saving_ratio']:>10.4f}")


if __name__ == "__main__":
    main()
