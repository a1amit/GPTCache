# GPTCache Benchmark Guide

## Project Overview

This is a fork of [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) that adds a **Token-Cost-Aware W-TinyLFU** eviction policy. Standard eviction strategies (LRU, LFU, FIFO) treat all cached entries as equal, but LLM cache entries vary enormously in regeneration cost -- a 2000-token response is far more expensive to recompute than a 20-token one. Our W-TinyLFU implementation combines a frequency-based admission filter (TinyLFU with Count-Min Sketch and Bloom filter doorkeeper) with a segmented LRU main cache, and optionally weighs eviction decisions by the response token count so that high-cost entries are preferentially retained.

## Installation

```bash
git clone https://github.com/a1amit/GPTCache.git
cd GPTCache
git checkout feature/wtinylfu-cost-aware
```

> **⚠️ RECOMMENDED: Run benchmarks locally with `--workers -1` (all CPU cores). This takes ~5 minutes. Docker runs sequentially inside the container and can take 40+ minutes. Use Docker only for reproducibility verification, not for day-to-day benchmarking.**

### Option A: Docker (no setup required)

Build once, then run the full evaluation (tests + benchmarks + ablation + figures) in a single command:

```bash
docker build -t gptcache-bench .

# Quick run (output stays inside container)
docker run --rm gptcache-bench

# Save JSON results, ablation data, and figures to your host machine
# Linux / macOS / Git Bash
docker run --rm \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/results_ablation:/app/results_ablation" \
    -v "$(pwd)/figures:/app/figures" \
    gptcache-bench

# Windows CMD
docker run --rm -v "%cd%\results:/app/results" -v "%cd%\results_ablation:/app/results_ablation" -v "%cd%\figures:/app/figures" gptcache-bench

# Windows PowerShell
docker run --rm -v "${PWD}\results:/app/results" -v "${PWD}\results_ablation:/app/results_ablation" -v "${PWD}\figures:/app/figures" gptcache-bench
```

**Important:** Without the `-v` volume mounts, all output stays inside the container and is lost when it exits.

**Performance note:** The Docker container runs benchmarks sequentially (`--workers 0`) to avoid process-pool deadlocks in constrained container environments. If you need faster runs, either allocate more CPU/memory to Docker Desktop (Settings → Resources) or use the local installation (Option B) with `--workers -1`.

The container executes four phases:
1. **Unit tests** (38 tests) -- proves correctness
2. **Benchmarks** (LRU, FIFO, LFU, W-TinyLFU, W-TinyLFU+Cost at cache sizes 50/100/200) -- proves performance gain
3. **Ablation study** -- workload profiles, component ablation, window size + cache size sweep
4. **Visualization** -- generates comparison plots and improvement-vs-LRU analysis

After the run completes, the following files will be available on your host:

| Directory | Contents |
|-----------|----------|
| `results/` | Per-policy JSON result files (`*_cs{size}_t{threshold}.json`) with hit rate, latency percentiles, token savings, throughput, and memory usage. Per-request logs (`*_log.json`) for CDF analysis. Combined `summary.json`. |
| `results_ablation/` | `ablation_summary.json` with workload profile comparison, component ablation, and window size parameter sweep results. |
| `figures/` | Publication-quality plots (PDF + PNG): hit rate, token savings, latency percentiles, improvement vs LRU, latency CDF, hit rate over time. Plain-text `summary_table.txt`. |

### Option B: Local virtual environment

```bash
# Create and activate a virtual environment (Python 3.13 recommended)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install the project plus pinned benchmark dependencies
pip install -e .
pip install -r requirements-bench.txt
```

### Option C: Conda

An `environment.yml` is provided for conda users with all dependencies pinned:

```bash
conda env create -f environment.yml
conda activate gptcache-wtinylfu
pip install -e .
```

## Quick Start (Synthetic Data)

> **Docker users:** The container already runs tests, benchmarks, ablation, and visualization automatically. The sections below are for local (Option B/C) usage only.

Run a fast benchmark using a generated Zipfian workload -- no dataset downloads required:

```
python benchmarks/run_benchmarks.py --dataset synthetic --n_samples 500 --cache_sizes 50 --policies "lru,wtinylfu" --output results/ --workers -1 --repeats 3
```

This will replay 500 synthetic queries against each policy and print a comparison table to the console. Results are saved as JSON files in the `results/` directory.

## Full Benchmark (Real Datasets)

For realistic evaluation, use conversation datasets from HuggingFace. These require `pip install datasets` and may need you to accept the dataset license on HuggingFace.

**LMSYS-Chat-1M** (requires HuggingFace token + license acceptance):

```
# Bash
export HF_TOKEN="hf_your_token"
# PowerShell
$env:HF_TOKEN = "hf_your_token"

python benchmarks/run_benchmarks.py --dataset lmsys --n_samples 3000 --cache_sizes "50,100,200" --policies "lru,fifo,lfu,wtinylfu,wtinylfu_nocost" --thresholds 0.85 --output results_lmsys/ --workers -1 --repeats 3
```

**WildChat-1M:**

```
python benchmarks/run_benchmarks.py --dataset wildchat --n_samples 3000 --cache_sizes "50,100,200" --policies "lru,fifo,lfu,wtinylfu,wtinylfu_nocost" --thresholds 0.85 --output results_wildchat/ --workers -1 --repeats 3
```

**Synthetic** (no downloads required):

```
python benchmarks/run_benchmarks.py --dataset synthetic --n_samples 3000 --cache_sizes "50,100,200" --policies "lru,fifo,lfu,wtinylfu,wtinylfu_nocost" --thresholds 0.85 --output results/ --workers -1 --repeats 3
```

## Parallel Execution and Repeated Trials

The benchmark runner supports parallel execution and repeated trials for statistical significance:

| Flag | Behavior |
|------|----------|
| `--workers 0` | Sequential (default) -- shows per-query progress |
| `--workers 4` | Use 4 parallel processes |
| `--workers -1` | Use all available CPU cores |
| `--repeats 1` | Single trial (default) |
| `--repeats 3` | Run each config 3 times with shuffled query order, report mean ± std |

All prompt embeddings are pre-computed in a single batch before any benchmark configs run. Each config then runs in its own process with an independent cache instance. When `--repeats N` is used (N > 1), the query order is shuffled with a different random seed per trial to create meaningful variation in eviction decisions. The summary JSON reports mean, std, min, and max for each metric.

A typical 15-config benchmark with 3 repeats (45 total runs) takes ~5-10 minutes with `--workers -1`.

## Visualization

After running benchmarks, generate publication-quality plots:

```bash
python benchmarks/visualize.py --input results/ --output figures/
```

This produces the following in the `figures/` directory:

- `hit_rate_by_cache_size.{pdf,png}` -- hit rate grouped by cache size and policy
- `token_saving_by_cache_size.{pdf,png}` -- token saving ratio by cache size and policy
- `latency_comparison.{pdf,png}` -- p50/p95/p99 lookup latency by policy
- `improvement_vs_lru.{pdf,png}` -- percentage improvement over LRU baseline (hit rate + token saving)
- `latency_cdf.{pdf,png}` -- cumulative distribution of per-request latency
- `hit_rate_over_time.{pdf,png}` -- sliding-window hit rate over the query stream
- `summary_table.txt` -- plain-text comparison table

## Available Eviction Policies

| Policy key        | Description                                                |
|-------------------|------------------------------------------------------------|
| `lru`             | Least Recently Used                                        |
| `fifo`            | First In, First Out                                        |
| `lfu`             | Least Frequently Used                                      |
| `rr`              | Random Replacement                                         |
| `wtinylfu`        | W-TinyLFU with cost-aware admission (token cost weighting) |
| `wtinylfu_nocost` | W-TinyLFU without cost-awareness (frequency only)          |

## W-TinyLFU Tunable Parameters

These can be passed via `extra_params` in the simulator or directly to the `WTinyLFUEviction` constructor:

| Parameter              | Default | Description                                                               |
|------------------------|---------|---------------------------------------------------------------------------|
| `window_pct`           | `1.0`   | Window cache size as a percentage of total capacity                       |
| `probation_pct`        | `20.0`  | Probation segment as a percentage of the main cache                       |
| `cost_aware`           | `True`  | Enable cost-weighted eviction decisions (uses response token count)       |
| `cms_width_multiplier` | `1`     | Count-Min Sketch width = next_power_of_2(maxsize * this multiplier)       |
| `reset_multiplier`     | `10`    | CMS frequency counters reset every maxsize * this multiplier increments   |

## Running Tests Manually

If you installed locally (Option B/C) and want to run just the unit tests without benchmarks:

```bash
# Run all eviction policy unit tests (38 tests)
python -m pytest tests/unit_tests/eviction/ -v -o "addopts=" \
    --ignore=tests/unit_tests/eviction/test_distributed_cache.py

# Run only the W-TinyLFU tests (10 tests)
python -m pytest tests/unit_tests/eviction/test_wtinylfu.py -v -o "addopts="

# Run only the data structure tests (16 tests)
python -m pytest tests/unit_tests/eviction/test_count_min_sketch.py \
    tests/unit_tests/eviction/test_doorkeeper.py \
    tests/unit_tests/eviction/test_segmented_lru.py -v -o "addopts="
```

Note: The `-o "addopts="` flag overrides GPTCache's `pytest.ini` which adds
`--html` flags requiring an optional dependency we don't need.
