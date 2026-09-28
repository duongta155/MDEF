#!/usr/bin/env bash
# Stage 4: per-condition logical-OR late fusion across the three members'
# 18-vectors b_p(V). Produces the final fused label vector \hat{y}(V)
# that drives the CE F1 headline of the paper.
source "$(dirname "$0")/../base.sh"

$PY -m src.fusion.logical_or \
    --infer_config configs/care_rg_infer.yaml \
    --predictions_root "$PRED_ROOT" \
    --output_root "$PRED_ROOT/ensemble_3way_or"
