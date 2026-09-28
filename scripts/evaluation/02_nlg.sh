#!/usr/bin/env bash
# NLG fluency evaluation (BLEU-1/2/3/4, METEOR, ROUGE-L) of the
# retrieved-sentence report against the ground-truth report on each tier.
source "$(dirname "$0")/../base.sh"

$PY -m src.evaluation.evaluator \
    --mode nlg \
    --retrieved_root "$RETRIEVED_ROOT" \
    --output_root "$RETRIEVED_ROOT"
