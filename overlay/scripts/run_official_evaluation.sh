#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the directory containing the datasets}
CONTROL=${CONTROL:-$ROOT/run_outputs/official_evaluation}

mkdir -p "$CONTROL"
rm -f "$CONTROL/JOB_SUCCESS" "$CONTROL/JOB_FAILED"
cd "$ROOT" || exit 1

if [[ -n "${VENV:-}" ]]; then
  source "$VENV/bin/activate"
fi
python tools/install_imagenet_r_metadata.py \
  --mapping descriptions/imagenet_r_classes_v1.json \
  --dataset-root "$DATA_ROOT/imagenet_r" || exit 1

export DATA="$DATA_ROOT"
export CIFAR100_HF_ARCHIVE=${CIFAR100_HF_ARCHIVE:-$DATA_ROOT/cifar100_hf/cifar100.npz}
export CLIP_DOWNLOAD_ROOT=${CLIP_DOWNLOAD_ROOT:-$ROOT/.cache/clip}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python main.py --config \
  options/multi_steps/formal_topecl_attribute_ablation/cifar100_seed42_official.yaml \
  >"$CONTROL/cifar100.console.log" 2>&1 &
cifar_pid=$!

python main.py --config \
  options/multi_steps/formal_topecl_attribute_ablation/imagenet_r_seed42_official.yaml \
  >"$CONTROL/imagenet_r.console.log" 2>&1 &
imagenet_pid=$!

printf '%s\n' "$cifar_pid" > "$CONTROL/cifar100.pid"
printf '%s\n' "$imagenet_pid" > "$CONTROL/imagenet_r.pid"

wait "$cifar_pid"
cifar_code=$?
wait "$imagenet_pid"
imagenet_code=$?
printf '%s\n' "$cifar_code" > "$CONTROL/cifar100.exitcode"
printf '%s\n' "$imagenet_code" > "$CONTROL/imagenet_r.exitcode"

if [[ "$cifar_code" -eq 0 && "$imagenet_code" -eq 0 ]]; then
  printf 'complete\n' > "$CONTROL/JOB_SUCCESS"
  exit 0
fi

printf 'failed\n' > "$CONTROL/JOB_FAILED"
exit 1
