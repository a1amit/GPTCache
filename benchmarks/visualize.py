#!/usr/bin/env python3
"""Publication-quality visualization for benchmark results.

Usage:
    python benchmarks/visualize.py --input results/ --output figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Readable defaults — large enough for reports and presentations
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
})

POLICY_COLORS = {
    "lru": "#4C72B0",
    "fifo": "#DD8452",
    "lfu": "#55A868",
    "rr": "#C44E52",
    "wtinylfu_nocost": "#8172B3",
    "wtinylfu": "#E04040",
}

POLICY_LABELS = {
    "lru": "LRU",
    "fifo": "FIFO",
    "lfu": "LFU",
    "rr": "Random",
    "wtinylfu_nocost": "W-TinyLFU",
    "wtinylfu": "W-TinyLFU + Cost",
}


def load_results(results_dir: Path):
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    results = []
    for p in results_dir.glob("*.json"):
        if p.name == "summary.json":
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results


def _get_policies(filtered):
    order = list(POLICY_COLORS.keys())
    policies = sorted(
        set(r["policy"] for r in filtered),
        key=lambda p: order.index(p) if p in order else 99,
    )
    return policies


def _save(fig, output_dir, name):
    for ext in ("pdf", "png"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}.pdf + {name}.png")


def plot_hit_rate_by_cache_size(results, output_dir: Path, threshold: float = 0.85):
    fig, ax = plt.subplots(figsize=(8, 5))

    filtered = [r for r in results if abs(r["similarity_threshold"] - threshold) < 0.01]
    if not filtered:
        return

    cache_sizes = sorted(set(r["cache_size"] for r in filtered))
    policies = _get_policies(filtered)
    x = np.arange(len(cache_sizes))
    width = 0.75 / len(policies)

    for i, policy in enumerate(policies):
        rates = []
        for cs in cache_sizes:
            match = [r for r in filtered
                     if r["policy"] == policy and r["cache_size"] == cs]
            rates.append(match[0]["hit_rate"] if match else 0)

        color = POLICY_COLORS.get(policy, "#999999")
        label = POLICY_LABELS.get(policy, policy)
        bars = ax.bar(x + i * width - (len(policies) - 1) * width / 2,
                       rates, width, label=label, color=color, edgecolor="white")
        # Add value labels on bars
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Cache Size (entries)")
    ax.set_ylabel("Hit Rate")
    ax.set_title(f"Cache Hit Rate by Eviction Policy\n(similarity threshold = {threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels(cache_sizes)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, min(max(r["hit_rate"] for r in filtered) * 1.15, 1.05))

    _save(fig, output_dir, "hit_rate_by_cache_size")


def plot_token_saving_by_cache_size(results, output_dir: Path, threshold: float = 0.85):
    fig, ax = plt.subplots(figsize=(8, 5))

    filtered = [r for r in results if abs(r["similarity_threshold"] - threshold) < 0.01]
    if not filtered:
        return

    cache_sizes = sorted(set(r["cache_size"] for r in filtered))
    policies = _get_policies(filtered)
    x = np.arange(len(cache_sizes))
    width = 0.75 / len(policies)

    for i, policy in enumerate(policies):
        ratios = []
        for cs in cache_sizes:
            match = [r for r in filtered
                     if r["policy"] == policy and r["cache_size"] == cs]
            ratios.append(match[0]["token_saving_ratio"] if match else 0)

        color = POLICY_COLORS.get(policy, "#999999")
        label = POLICY_LABELS.get(policy, policy)
        bars = ax.bar(x + i * width - (len(policies) - 1) * width / 2,
                       ratios, width, label=label, color=color, edgecolor="white")
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Cache Size (entries)")
    ax.set_ylabel("Token Saving Ratio")
    ax.set_title(f"Token Saving Ratio by Eviction Policy\n"
                 f"(higher = more expensive responses retained in cache)")
    ax.set_xticks(x)
    ax.set_xticklabels(cache_sizes)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, min(max(r["token_saving_ratio"] for r in filtered) * 1.15, 1.05))

    _save(fig, output_dir, "token_saving_by_cache_size")


def plot_latency_comparison(results, output_dir: Path, threshold: float = 0.85):
    fig, ax = plt.subplots(figsize=(8, 5))

    max_cs = max(r["cache_size"] for r in results)
    filtered = [r for r in results
                if r["cache_size"] == max_cs
                and abs(r["similarity_threshold"] - threshold) < 0.01]
    if not filtered:
        return

    policies = _get_policies(filtered)
    x = np.arange(len(policies))
    width = 0.25

    p50s, p95s, p99s = [], [], []
    for policy in policies:
        match = [r for r in filtered if r["policy"] == policy]
        if match:
            p50s.append(match[0]["latency_p50"])
            p95s.append(match[0]["latency_p95"])
            p99s.append(match[0]["latency_p99"])
        else:
            p50s.append(0)
            p95s.append(0)
            p99s.append(0)

    ax.bar(x - width, p50s, width, label="p50 (median)", color="#4C72B0")
    ax.bar(x, p95s, width, label="p95", color="#DD8452")
    ax.bar(x + width, p99s, width, label="p99", color="#C44E52")

    labels = [POLICY_LABELS.get(p, p) for p in policies]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Per-Request Latency Percentiles\n(cache_size = {max_cs})")
    ax.legend(frameon=True, fancybox=True, shadow=True)

    _save(fig, output_dir, "latency_comparison")


def plot_improvement_summary(results, output_dir: Path, threshold: float = 0.85):
    """Side-by-side: hit rate improvement AND token saving improvement vs LRU."""
    filtered = [r for r in results if abs(r["similarity_threshold"] - threshold) < 0.01]
    if not filtered:
        return

    cache_sizes = sorted(set(r["cache_size"] for r in filtered))
    policies = [p for p in _get_policies(filtered) if p != "lru"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (ax1, "hit_rate", "Hit Rate Improvement vs LRU (%)"),
        (ax2, "token_saving_ratio", "Token Saving Improvement vs LRU (%)"),
    ]:
        x = np.arange(len(cache_sizes))
        width = 0.7 / len(policies)

        for i, policy in enumerate(policies):
            improvements = []
            for cs in cache_sizes:
                lru = [r for r in filtered if r["policy"] == "lru" and r["cache_size"] == cs]
                pol = [r for r in filtered if r["policy"] == policy and r["cache_size"] == cs]
                if lru and pol:
                    lru_val = lru[0][metric]
                    pol_val = pol[0][metric]
                    if lru_val > 0:
                        pct = (pol_val - lru_val) / lru_val * 100
                    else:
                        pct = 0
                    improvements.append(pct)
                else:
                    improvements.append(0)

            color = POLICY_COLORS.get(policy, "#999999")
            label = POLICY_LABELS.get(policy, policy)
            bars = ax.bar(x + i * width - (len(policies) - 1) * width / 2,
                           improvements, width, label=label, color=color, edgecolor="white")
            for bar, val in zip(bars, improvements):
                if abs(val) > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.1 if val >= 0 else -0.3),
                            f"{val:+.2f}%", ha="center", va="bottom", fontsize=8,
                            fontweight="bold")

        ax.set_xlabel("Cache Size (entries)")
        ax.set_ylabel("Improvement vs LRU (%)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(cache_sizes)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)

    fig.suptitle("Performance Improvement Over LRU Baseline", fontsize=15, fontweight="bold", y=1.02)
    _save(fig, output_dir, "improvement_vs_lru")


def _load_logs(results_dir: Path):
    """Load per-request logs from *_log.json files."""
    logs = {}
    for p in results_dir.glob("*_log.json"):
        # Parse key from filename: dataset_policy_csN_tT_log.json
        key = p.stem.replace("_log", "")
        with open(p) as f:
            logs[key] = json.load(f)
    return logs


def plot_latency_cdf(results_dir: Path, output_dir: Path, threshold: float = 0.85):
    """CDF of per-request latency for each policy (largest cache size)."""
    logs = _load_logs(results_dir)
    if not logs:
        print("  No per-request logs found — skipping latency CDF")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, log in sorted(logs.items()):
        # Extract policy from key
        parts = key.split("_")
        # Find the policy name (between dataset and csN)
        policy = None
        for i, p in enumerate(parts):
            if p.startswith("cs"):
                policy = "_".join(parts[1:i]) if i > 1 else parts[0]
                break
        if policy is None:
            continue

        latencies = sorted([r["total_time_ms"] for r in log])
        n = len(latencies)
        if n == 0:
            continue
        cdf = np.arange(1, n + 1) / n
        color = POLICY_COLORS.get(policy, "#999999")
        label = POLICY_LABELS.get(policy, policy)
        ax.plot(latencies, cdf, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Per-Request Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Latency CDF by Eviction Policy")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.05)

    _save(fig, output_dir, "latency_cdf")


def plot_hit_rate_over_time(results_dir: Path, output_dir: Path,
                            threshold: float = 0.85, window: int = 200):
    """Sliding-window hit rate over time for each policy (largest cache size)."""
    logs = _load_logs(results_dir)
    if not logs:
        print("  No per-request logs found — skipping hit rate curve")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, log in sorted(logs.items()):
        parts = key.split("_")
        policy = None
        for i, p in enumerate(parts):
            if p.startswith("cs"):
                policy = "_".join(parts[1:i]) if i > 1 else parts[0]
                break
        if policy is None:
            continue

        hits = [r["hit"] for r in log]
        n = len(hits)
        if n < window:
            continue

        # Sliding window hit rate
        rates = []
        window_hits = sum(hits[:window])
        rates.append(window_hits / window)
        for i in range(window, n):
            window_hits += hits[i] - hits[i - window]
            rates.append(window_hits / window)

        x = list(range(window, n + 1))
        color = POLICY_COLORS.get(policy, "#999999")
        label = POLICY_LABELS.get(policy, policy)
        ax.plot(x, rates, label=label, color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Query Number")
    ax.set_ylabel(f"Hit Rate (sliding window = {window})")
    ax.set_title("Cache Hit Rate Over Time")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, None)

    _save(fig, output_dir, "hit_rate_over_time")


def plot_summary_table(results, output_dir: Path, threshold: float = 0.85):
    filtered = [r for r in results if abs(r["similarity_threshold"] - threshold) < 0.01]
    if not filtered:
        return

    lines = []
    lines.append(f"{'Policy':<22} {'Size':>6} {'HitRate':>9} {'TokenSave':>10} "
                 f"{'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'Evictions':>10} "
                 f"{'Mem MB':>8} {'QPS':>8}")
    lines.append("-" * 113)

    for r in sorted(filtered, key=lambda x: (x["cache_size"], x["policy"])):
        label = POLICY_LABELS.get(r["policy"], r["policy"])
        mem_mb = r.get("peak_memory_mb", 0.0)
        qps = r.get("throughput_qps", 0.0)
        lines.append(
            f"{label:<22} {r['cache_size']:>6} {r['hit_rate']:>9.4f} "
            f"{r['token_saving_ratio']:>10.4f} {r['latency_p50']:>8.1f} "
            f"{r['latency_p95']:>8.1f} {r['latency_p99']:>8.1f} "
            f"{r['total_evictions']:>10} "
            f"{mem_mb:>8.1f} {qps:>8.1f}"
        )

    text = "\n".join(lines)
    path = output_dir / "summary_table.txt"
    with open(path, "w") as f:
        f.write(text)
    print(f"  Saved {path}")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", type=str, default="results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="figures",
                        help="Figures output directory")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Default threshold for single-threshold plots")

    args = parser.parse_args()
    results_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} benchmark results\n")

    plot_hit_rate_by_cache_size(results, output_dir, args.threshold)
    plot_token_saving_by_cache_size(results, output_dir, args.threshold)
    plot_latency_comparison(results, output_dir, args.threshold)
    plot_improvement_summary(results, output_dir, args.threshold)
    plot_latency_cdf(results_dir, output_dir, args.threshold)
    plot_hit_rate_over_time(results_dir, output_dir, args.threshold)
    plot_summary_table(results, output_dir, args.threshold)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
