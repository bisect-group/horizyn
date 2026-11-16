#!/usr/bin/env bash
# Horizyn paper runs: 5 retrieval splits, 100 epochs, bf16 (--amp). train also evaluates (test).
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|rescore|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1 AMP=0 (fp32)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets/enzyme_retrieval_dataset}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/Horizon}
PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-2}
SEEDS=${SEEDS:-"666 777 888"}
read -ra gpus <<<"$GPUS"
read -ra seeds <<<"$SEEDS"
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache|train|rescore|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|rescore|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

AMP_FLAG=--amp
[[ ${AMP:-1} == 1 ]] || AMP_FLAG=

want cache && run $PY -m emulator_bench.run_benchmarks --dataset_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --gpus "${gpus[@]}" --features_only
want train && run $PY -m emulator_bench.run_benchmarks --dataset_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --gpus "${gpus[@]}" --runs_per_gpu "$RUNS_PER_GPU" --seeds "${seeds[@]}" $AMP_FLAG
# Scores without preliminary ECs (the paper rows): ec_level_accuracy_no_b.json per run.
want rescore && run $PY -m emulator_bench.rescore --out_root "$OUT_ROOT"
want aggregate && {
  run $PY -m emulator_bench.aggregate_results --out_root "$OUT_ROOT"
  run $PY -m emulator_bench.aggregate_results --out_root "$OUT_ROOT" --no_b
}
exit 0
