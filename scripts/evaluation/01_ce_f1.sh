#!/usr/bin/env bash
# Clinical Efficacy (CE) F1 evaluation. Computes Precision, Recall, F1
# micro-averaged over the 18 RadBERT conditions for each tier, and
# per-condition P/R/F1 used in Table III of the paper.
source "$(dirname "$0")/../base.sh"

for TIER in rg ctrate inspect; do
    PREDS="$PRED_ROOT/ensemble_3way_or/${TIER}/predictions.csv"
    OUT="$PRED_ROOT/ensemble_3way_or/${TIER}"
    [ -f "$PREDS" ] || { echo "missing: $PREDS"; continue; }
    $PY -m src.evaluation.ce_evaluation \
        --predictions_csv "$PREDS" \
        --radbert_path "$RADBERT_CKPT" \
        --output_dir "$OUT" \
        --batch_size 32
done
