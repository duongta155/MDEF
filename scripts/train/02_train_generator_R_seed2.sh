#!/usr/bin/env bash
# Train the second region-grounded ensemble member (M_R^(2) in the paper).
# Same recipe as member 1 but seeded differently to decorrelate
# per-condition mention errors.
source "$(dirname "$0")/../base.sh"

$PY -m src.training.trainer \
    --config configs/care_rg_train.yaml \
    --seed 2 \
    --init_from reg2rg \
    --output_dir "$CKPT_ROOT/generator_R_2"
