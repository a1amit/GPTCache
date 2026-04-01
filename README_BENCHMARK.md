# GPTCache Benchmark Guide

## Project Overview

This is a fork of [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) that adds a **Token-Cost-Aware W-TinyLFU** eviction policy. Standard eviction strategies (LRU, LFU, FIFO) treat all cached entries as equal, but LLM cache entries vary enormously in regeneration cost -- a 2000-token response is far more expensive to recompute than a 20-token one. Our W-TinyLFU implementation combines a frequency-based admission filter (TinyLFU with Count-Min Sketch and Bloom filter doorkeeper) with a segmented LRU main cache, and optionally weighs eviction decisions by the response token count so that high-cost entries are preferentially retained.

## Installation

```bash
# Clone the repository
git clone https://github.com/a1amit/GPTCache.git
cd GPTCache
git checkout feature/wtinylfu-cost-aware

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

## Quick Start (Synthetic Data)

Run a fast benchmark using a generated Zipfian workload -- no dataset downloads required:

```bash
python benchmarks/run_benchmarks.py \
    --dataset synthetic \
    --n_samples 500 \
    --cache_sizes 50 \
    --policies lru,wtinylfu \
    --output results/ \
    --workers -1
```

This will replay 500 synthetic queries against each policy and print a comparison table to the console. Results are saved as JSON files in the `results/` directory.

## Full Benchmark (Real Datasets)

For realistic evaluation, use conversation datasets from HuggingFace. These require `pip install datasets` and may need you to accept the dataset license on HuggingFace.

**LMSYS-Chat-1M** (requires HuggingFace token + license acceptance):

```bash
export HF_TOKEN="hf_your_token"  # or $env:HF_TOKEN on PowerShell
python benchmarks/run_benchmarks.py \
    --dataset lmsys \
    --n_samples 3000 \
    --cache_sizes 20,50,100 \
    --policies lru,fifo,lfu,wtinylfu,wtinylfu_nocost \
    --thresholds 0.85 \
    --output results_lmsys/ \
    --workers -1
```

**WildChat-1M:**

```bash
python benchmarks/run_benchmarks.py \
    --dataset wildchat \
    --n_samples 3000 \
    --cache_sizes 20,50,100 \
    --policies lru,fifo,lfu,wtinylfu,wtinylfu_nocost \
    --thresholds 0.85 \
    --output results_wildchat/ \
    --workers -1
```

**Synthetic** (no downloads required):

```bash
python benchmarks/run_benchmarks.py \
    --dataset synthetic \
    --n_samples 3000 \
    --cache_sizes 10,20,50 \
    --policies lru,fifo,lfu,wtinylfu,wtinylfu_nocost \
    --thresholds 0.85 \
    --output results/ \
    --workers -1
```

## Parallel Execution

The benchmark runner supports parallel execution via the `--workers` flag:

| Flag | Behavior |
|------|----------|
| `--workers 0` | Sequential (default) -- shows per-query progress |
| `--workers 4` | Use 4 parallel processes |
| `--workers -1` | Use all available CPU cores |

All prompt embeddings are pre-computed in a single batch before any benchmark configs run. Each config then runs in its own process with an independent cache instance. This brings a typical 15-config benchmark from ~45 minutes down to ~4 minutes.

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

## Running Tests

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

## Docker Usage (Recommended for Reproducibility)

One command runs everything — tests, benchmarks, and plot generation:

```bash
# Build the image
docker build -t gptcache-bench .

# Run full evaluation (tests + benchmarks + figures)
docker run --rm gptcache-bench
```

This executes:
1. **Unit tests** (38 tests) — proves correctness
2. **Benchmarks** (LRU, FIFO, LFU, W-TinyLFU, W-TinyLFU+Cost at cache sizes 10/20/50/100/200) — proves performance gain, runs in parallel using all available cores
3. **Ablation study** — workload profiles, component ablation, window size sweep
4. **Visualization** — generates comparison plots and improvement-vs-LRU analysis

To save results and figures to your host machine:

```bash
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

## Conda

```bash
conda env create -f environment.yml
conda activate gptcache-wtinylfu
pip install -e .
```
