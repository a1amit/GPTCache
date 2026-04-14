# Deliverables Checklist

Mapping of each guideline requirement to its location in this repository.

## 1. Baseline Framework Selection (one-page justification)

- **Location**: `report.tex` Section 1.3 ("Baseline Framework Selection")
- Includes: feature table, numbered justification, alternatives-considered table

## 2. Performance Test Suite

| Deliverable | Location |
|---|---|
| Benchmark README | [`README_BENCHMARK.md`](../../README_BENCHMARK.md) |
| Benchmark scripts | [`benchmarks/`](../../benchmarks/) (`run_benchmarks.py`, `run_ablation.py`, `simulator.py`, `visualize.py`, `metrics.py`) |
| Sample JSON output | [`results_synthetic/`](./results_synthetic/), [`results_lmsys_080/`](./results_lmsys_080/), [`results_lmsys_085/`](./results_lmsys_085/) |
| CI pipeline | [`.github/workflows/benchmark.yml`](../../.github/workflows/benchmark.yml) |

## 3. Extension Design (code + tests)

| Deliverable | Location |
|---|---|
| W-TinyLFU policy | [`gptcache/manager/eviction/wtinylfu_eviction.py`](../../gptcache/manager/eviction/wtinylfu_eviction.py) (223 lines) |
| Count-Min Sketch | [`gptcache/manager/eviction/count_min_sketch.py`](../../gptcache/manager/eviction/count_min_sketch.py) (112 lines) |
| Bloom Doorkeeper | [`gptcache/manager/eviction/doorkeeper.py`](../../gptcache/manager/eviction/doorkeeper.py) (67 lines) |
| Segmented LRU | [`gptcache/manager/eviction/segmented_lru.py`](../../gptcache/manager/eviction/segmented_lru.py) (100 lines) |
| Factory registration | [`gptcache/manager/eviction/manager.py`](../../gptcache/manager/eviction/manager.py) |
| Unit tests (38 total) | [`tests/unit_tests/eviction/`](../../tests/unit_tests/eviction/) |
| Feature branch | `feature/wtinylfu-cost-aware` |

## 4. Evaluation & Analysis (PDF report)

| Deliverable | Location |
|---|---|
| Report source | [`report.tex`](./report.tex) |
| Figures (synthetic) | [`figures_synthetic/`](./figures_synthetic/) |
| Figures (LMSYS t=0.80) | [`figures_lmsys_080/`](./figures_lmsys_080/) |
| Figures (LMSYS t=0.85) | [`figures_lmsys_085/`](./figures_lmsys_085/) |
| Raw results (synthetic) | [`results_synthetic/`](./results_synthetic/) |
| Raw results (LMSYS t=0.80) | [`results_lmsys_080/`](./results_lmsys_080/) |
| Raw results (LMSYS t=0.85) | [`results_lmsys_085/`](./results_lmsys_085/) |

## 5. Reproducibility

| Deliverable | Location |
|---|---|
| Dockerfile | [`Dockerfile`](../../Dockerfile) |
| Conda environment | [`environment.yml`](../../environment.yml) |
| Pip requirements | [`requirements-bench.txt`](../../requirements-bench.txt) |
| Installation README | [`README.md`](../../README.md) |

## Quick reproduce

```bash
git clone https://github.com/a1amit/GPTCache.git
cd GPTCache && git checkout feature/wtinylfu-cost-aware

# Docker (one command)
docker build -t gptcache-bench . && docker run --rm gptcache-bench

# Or local (~5 min)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e . && pip install -r requirements-bench.txt
python -m pytest tests/unit_tests/eviction/ -v -o "addopts=" \
    --ignore=tests/unit_tests/eviction/test_distributed_cache.py
python benchmarks/run_benchmarks.py --dataset synthetic \
    --n_samples 3000 --cache_sizes "50,100,200" \
    --policies "lru,fifo,lfu,wtinylfu,wtinylfu_nocost" \
    --thresholds 0.85 --output results/ --workers -1 --repeats 3
python benchmarks/visualize.py --input results/ --output figures/
```
