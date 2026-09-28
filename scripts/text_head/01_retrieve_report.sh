#!/usr/bin/env bash
# Stage 5: deterministic text head Psi. Maps the fused 18-vector
# \hat{y}(V) and the per-member SD-union texts {R_p(V)}_p to a single
# textual report T(V) consistent with the OR fusion. Used only for NLG
# scoring; the CE F1 metric is unchanged under any choice of head.
source "$(dirname "$0")/../base.sh"

$PY -m src.text_head.retrieve_report \
    --infer_config configs/care_rg_infer.yaml \
    --predictions_root "$PRED_ROOT" \
    --output_root "$RETRIEVED_ROOT"
