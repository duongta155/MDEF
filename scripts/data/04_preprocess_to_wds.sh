#!/usr/bin/env bash
# Convert each held-out tier's volumes to WebDataset (WDS) shards under
# $PROCESSED_ROOT. WDS shards are what src/data/wds_dataset.py reads
# during training and inference (per region tensors stored under
# sample["vision_x"]).
source "$(dirname "$0")/../base.sh"

for TIER in radgenome ctrate inspect; do
    case "$TIER" in
        radgenome) RAW=$RADGENOME_PATH ;;
        ctrate)    RAW=$CTRATE_PATH ;;
        inspect)   RAW=$INSPECT_PATH ;;
    esac
    OUT="$PROCESSED_ROOT/${TIER}_wds"
    mkdir -p "$OUT"
    $PY -m src.data.preprocess_to_wds \
        --raw_dir "$RAW" --out_dir "$OUT" --tier "$TIER"
done
