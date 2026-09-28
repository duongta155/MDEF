#!/usr/bin/env bash
# Stage 2: stochastic decoding per ensemble member, K=10 candidates at
# temperatures linearly staggered over [0.7, 1.3]. Produces the SD-union
# text R_p(V) for each member on each held-out tier. The 18-condition
# RadBERT label vector b_p(V) is also computed in this pass (Stage 3).
source "$(dirname "$0")/../base.sh"

for TIER in radgenome ctrate inspect; do
    case "$TIER" in
        radgenome) SHARD="$PROCESSED_ROOT/radgenome_wds";    OUTKEY="rg" ;;
        ctrate)    SHARD="$PROCESSED_ROOT/ctrate_wds";       OUTKEY="ctrate" ;;
        inspect)   SHARD="$PROCESSED_ROOT/inspect_wds";      OUTKEY="inspect" ;;
    esac
    for MEMBER in generator_R_1 generator_R_2 generator_P; do
        OUT="$PRED_ROOT/$MEMBER/${OUTKEY}_K10_UNION"
        mkdir -p "$OUT"
        $PY -m src.evaluation.evaluator \
            --design sd_union \
            --model_path "$CKPT_ROOT/$MEMBER/final_model.pt" \
            --infer_config configs/care_rg_infer.yaml \
            --output_dir "$OUT" \
            --shard_dir "$SHARD" \
            --eval_batch_size 8
    done
done
