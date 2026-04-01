#!/bin/bash
set -e

echo "============================================"
echo "  GPTCache W-TinyLFU — Full Evaluation"
echo "============================================"
echo ""

# --- Phase 1: Unit Tests (Correctness) ---
echo ">>> Phase 1: Running unit tests..."
echo ""
python -m pytest tests/unit_tests/eviction/ -v \
    --ignore=tests/unit_tests/eviction/test_distributed_cache.py \
    -o "addopts="
echo ""
echo ">>> All tests passed."
echo ""

# --- Phase 2: Benchmarks (Performance) ---
echo ">>> Phase 2: Running benchmarks..."
echo "    Policies: LRU, FIFO, LFU, W-TinyLFU, W-TinyLFU+Cost"
echo "    Cache sizes: 10, 20, 50, 100, 200"
echo "    Dataset: synthetic (Zipfian workload, vocab=500, variable response costs)"
echo ""
python -u benchmarks/run_benchmarks.py \
    --dataset synthetic \
    --n_samples 3000 \
    --cache_sizes 10,20,50,100,200 \
    --policies lru,fifo,lfu,wtinylfu,wtinylfu_nocost \
    --thresholds 0.85 \
    --output results/ \
    --workers -1
echo ""

# --- Phase 3: Ablation Study ---
echo ">>> Phase 3: Running ablation study..."
echo "    - Workload profiles (repetitive short vs novel long)"
echo "    - Component ablation (LRU vs W-TinyLFU vs W-TinyLFU+Cost)"
echo "    - Window size parameter sweep"
echo ""
python -u benchmarks/run_ablation.py \
    --output results_ablation/ \
    --cache_size 20
echo ""

# --- Phase 4: Generate Plots ---
echo ">>> Phase 4: Generating figures..."
python -u benchmarks/visualize.py --input results/ --output figures/
echo ""

echo "============================================"
echo "  Done. Results in results/, figures in figures/"
echo "  Ablation in results_ablation/"
echo "============================================"
