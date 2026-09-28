#!/usr/bin/env bash
# Train the pathology-conditioned ensemble member (M_P in the paper).
# Warm-starts from a Reg2RG checkpoint further pre-trained with an
# auxiliary multi-label classification head over the 18 RadBERT conditions.
source "$(dirname "$0")/../base.sh"

$PY -m src.training.trainer \
    --config configs/mdef_train.yaml \
    --seed 1 \
    --init_from pathology_warmstart \
    --output_dir "$CKPT_ROOT/generator_3"
