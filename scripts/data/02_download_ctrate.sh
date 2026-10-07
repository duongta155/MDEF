#!/usr/bin/env bash
# Download CT-RATE (used as the same-modality cross-dataset held-out tier).
source "$(dirname "$0")/../base.sh"
mkdir -p "$CTRATE_PATH"
hf download ibrahimhamamci/CT-RATE \
    --repo-type dataset --local-dir "$CTRATE_PATH"
