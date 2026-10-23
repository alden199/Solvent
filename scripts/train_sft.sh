#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash scripts/train_sft.sh [--dry-run] knowledge|cold_start|bootstrap|agent|CONFIG.yaml [key=value ...]' \
    'Required environment: MODEL, DATASET_DIR, DATASET, OUTPUT_DIR' \
    'Optional: PYTHON (python), NPROC_PER_NODE (8), NNODES (1), NODE_RANK (0)' \
    'Multi-node: set MASTER_ADDR and MASTER_PORT identically on every node.' \
    'Dry-run validates paths and dataset registration; it never imports torch or starts training.'
}

DRY_RUN=0
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK="$ROOT/third_party/LLaMA-Factory"
case "$1" in
  knowledge|cold_start|bootstrap|agent) CONFIG="$ROOT/configs/sft/$1.yaml" ;;
  *) CONFIG="$(realpath -e -- "$1")" ;;
esac
shift
: "${MODEL:?Set MODEL to a model directory or Hugging Face model ID}"
: "${DATASET_DIR:?Set DATASET_DIR to your dataset registration directory}"
: "${DATASET:?Set DATASET to one or more comma-separated registered dataset names}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a training output directory}"
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
for VALUE in "$NPROC_PER_NODE" "$NNODES"; do
  if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'NPROC_PER_NODE and NNODES must be positive integers.' >&2
    exit 2
  fi
done
if [[ ! "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
  printf '%s\n' 'NODE_RANK must be between zero and NNODES minus one.' >&2
  exit 2
fi
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
  : "${MASTER_PORT:?Set MASTER_PORT for multi-node training}"
fi
if [[ -n "${MASTER_PORT:-}" ]] && { [[ ! "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || (( MASTER_PORT > 65535 )); }; then
  printf '%s\n' 'MASTER_PORT must be an integer between 1 and 65535.' >&2
  exit 2
fi
if [[ ! -f "$CONFIG" || ! -f "$FRAMEWORK/src/train.py" ]]; then
  printf '%s\n' 'Missing SFT configuration or vendored framework.' >&2
  exit 2
fi
DATASET_DIR="$(realpath -e -- "$DATASET_DIR")"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
if [[ -d "$MODEL" ]]; then
  MODEL="$(realpath -e -- "$MODEL")"
elif [[ "$MODEL" == /* || "$MODEL" == ./* || "$MODEL" == ../* ]]; then
  printf '%s\n' 'The requested local model directory does not exist.' >&2
  exit 2
fi
for ARG in "$@"; do
  if [[ "$ARG" != *=* || "$ARG" == --* ]]; then
    printf '%s\n' 'Overrides must use key=value syntax.' >&2
    exit 2
  fi
  case "${ARG%%=*}" in
    model_name_or_path|dataset|dataset_dir|output_dir)
      printf '%s\n' 'Use MODEL, DATASET, DATASET_DIR and OUTPUT_DIR for required inputs.' >&2
      exit 2 ;;
  esac
done
"$PYTHON" - "$DATASET_DIR" "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "dataset_info.json"
if not manifest.is_file() or manifest.stat().st_size > 16 * 1024 * 1024:
    raise SystemExit("dataset_info.json is missing or exceeds the 16 MiB registration limit")
info = json.loads(manifest.read_text(encoding="utf-8"))
for name in sys.argv[2].split(","):
    if name not in info or not isinstance(info[name], dict):
        raise SystemExit(f"Dataset is not registered: {name}")
    entry = info[name]
    if "file_name" in entry and not (root / entry["file_name"]).exists():
        raise SystemExit(f"Dataset file or directory does not exist for: {name}")
    if not any(k in entry for k in ("file_name", "hf_hub_url", "ms_hub_url", "om_hub_url", "script_url", "cloud_file_name")):
        raise SystemExit(f"Dataset registration has no source: {name}")
PY
export PYTHONPATH="$FRAMEWORK/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
LAUNCH=("$PYTHON" -m torch.distributed.run --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE")
if (( NNODES == 1 )) && [[ -z "${MASTER_ADDR:-}" && -z "${MASTER_PORT:-}" ]]; then
  LAUNCH+=(--standalone)
else
  LAUNCH+=(--master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}")
fi
LAUNCH+=("$FRAMEWORK/src/train.py" "$CONFIG" "model_name_or_path=$MODEL" "dataset_dir=$DATASET_DIR" "dataset=$DATASET" "output_dir=$OUTPUT_DIR" "deepspeed=$ROOT/configs/sft/ds_z3_config.json" "$@")
if (( DRY_RUN )); then
  printf '%q ' "${LAUNCH[@]}"
  printf '\n'
  exit 0
fi
exec "${LAUNCH[@]}"
