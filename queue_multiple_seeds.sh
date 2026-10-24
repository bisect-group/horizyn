#!/usr/bin/env bash
# Run all default retrieval splits for seeds 666 777 888. Usage: [RUNS_PER_GPU=2] ./queue_multiple_seeds.sh [GPU ids...] (default: 0)
set -euo pipefail
cd "$(dirname "$0")"
gpus=("${@:-0}")
python -m emulator_bench.run_benchmarks --gpus "${gpus[@]}" --seeds 666 777 888 --runs_per_gpu "${RUNS_PER_GPU:-2}" --amp
python -m emulator_bench.aggregate_results
