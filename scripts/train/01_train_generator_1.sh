#!/usr/bin/env bash
# Train the first region-grounded ensemble member (M_R^(1) in the paper).
# Initialised from the public Reg2RG checkpoint; trained on the RadGenome
# 24k split with seed=1.
source "$(dirname "$0")/../base.sh"

$PY -m src.training.trainer \
    --config configs/mdef_train.yaml \
    --seed 1 \
    --init_from reg2rg \
    --output_dir "$CKPT_ROOT/generator_1"
