#!/usr/bin/env bash
# Paired-volume bootstrap 95% CIs for the NLG metrics of CARE-RG vs each
# single-trajectory baseline (Reg2RG, CT-GRAPH, MARCH) on each tier.
# Reproduces the bootstrap summary used in Sec. V-B of the paper.
source "$(dirname "$0")/../base.sh"

$PY -m src.evaluation.bootstrap_cis \
    --tiers rg ctrate inspect \
    --baselines Reg2RG CT-GRAPH MARCH \
    --metrics BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L \
    --bootstraps 500 \
    --workers 16 \
    --output_dir "$PRED_ROOT/bootstrap_cis"
