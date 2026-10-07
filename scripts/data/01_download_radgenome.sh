#!/usr/bin/env bash
# Download RadGenome-ChestCT. Requires an authenticated HuggingFace session (`hf auth login`).
# See README.md Section 2.
source "$(dirname "$0")/../base.sh"
mkdir -p "$RADGENOME_PATH"
hf download RadGenome/RadGenome-ChestCT \
    --repo-type dataset --local-dir "$RADGENOME_PATH"
