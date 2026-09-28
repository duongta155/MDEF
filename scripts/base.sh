#!/usr/bin/env bash
# Common environment for all scripts. Source this at the top of every script:
#   source "$(dirname "$0")/../base.sh"
# It loads paths from configs/storage.yaml and exports the canonical
# DATA_ROOT, CKPT_ROOT, PRED_ROOT, RADBERT_CKPT used by every command below.

set -euo pipefail

ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
CONFIG="$ROOT/configs/storage.yaml"

# Minimal YAML parsing for the keys we need (no external deps).
read_yaml_key () {
    awk -v key="$1" '
        $1 == key":" { gsub("\"", "", $2); print $2; exit }
    ' "$CONFIG"
}

export REPO_ROOT="$ROOT"
export DATA_ROOT="$(read_yaml_key data_root)"
export RADGENOME_PATH="$(read_yaml_key radgenome_path)"
export CTRATE_PATH="$(read_yaml_key ctrate_path)"
export INSPECT_PATH="$(read_yaml_key inspect_path)"
export RADBERT_CKPT="$(read_yaml_key radbert_ckpt)"
export CKPT_ROOT="$ROOT/$(read_yaml_key checkpoints)"
export PRED_ROOT="$ROOT/$(read_yaml_key predictions)"
export RETRIEVED_ROOT="$ROOT/$(read_yaml_key retrieved)"
export PROCESSED_ROOT="$(read_yaml_key processed_data)"

mkdir -p "$CKPT_ROOT" "$PRED_ROOT" "$RETRIEVED_ROOT"

# Python runner: prefer uv if available, fall back to current python.
if command -v uv >/dev/null 2>&1; then
    export PY="uv run python"
else
    export PY="python"
fi
