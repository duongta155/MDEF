#!/usr/bin/env bash
# Generate the per-condition Precision / Recall / F1 LaTeX table body
# (Table III of the paper) from the cached ce_per_condition.csv files.
source "$(dirname "$0")/../base.sh"

$PY -m src.evaluation.per_cond_pr_table \
    --predictions_root "$PRED_ROOT" \
    --output_path "$PRED_ROOT/per_cond_pr_table.tex"
