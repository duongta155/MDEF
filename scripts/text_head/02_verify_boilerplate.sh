#!/usr/bin/env bash
# Verify the Stage 5 boilerplate sentences are derived from the RadGenome
# 24k training split only (i.e., satisfy the "no target-tier data" claim
# of Stage 5 / Sec. IV-E). Reports per-region frequencies.
source "$(dirname "$0")/../base.sh"

$PY -m src.text_head.verify_boilerplate \
    --train_csv "$RADGENOME_PATH/radgenome_train_reports.csv"
