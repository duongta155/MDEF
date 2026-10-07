#!/usr/bin/env bash
# Download INSPECT (cross-institution Stanford CTPA cohort, held-out tier).
#
# INSPECT is released by Stanford AIMI for non-commercial use under a data use agreement:
#   https://aimi.stanford.edu/datasets/inspect-Multimodal-Dataset-for-Pulmonary-Embolism-Diagnosis-and-Prognosis
# Request access on that page, accept the agreement, then copy the SAS URL AIMI issues and export
# it as INSPECT_SAS_URL. The URL expires, so run this soon after it is issued.
#
#   export INSPECT_SAS_URL='https://...?sv=...'
#   bash scripts/data/03_download_inspect.sh
#
# Two terms of the agreement that this script keeps:
#   - the volumes stay owner-only, so the directory is created with mode 700
#   - azcopy writes the SAS token into its own logs, so the log directory is kept local and
#     owner-only as well. Never paste those logs anywhere and never print the SAS URL.
source "$(dirname "$0")/../base.sh"

: "${INSPECT_SAS_URL:?set INSPECT_SAS_URL to the SAS URL issued by Stanford AIMI}"
command -v azcopy >/dev/null || { echo "azcopy not found, see https://aka.ms/downloadazcopy"; exit 1; }

mkdir -p "$INSPECT_PATH"
chmod 700 "$INSPECT_PATH"

export AZCOPY_LOG_LOCATION="$INSPECT_PATH/.azcopy_logs"
export AZCOPY_JOB_PLAN_LOCATION="$INSPECT_PATH/.azcopy_plans"
mkdir -p "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"
chmod 700 "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"

azcopy copy "$INSPECT_SAS_URL" "$INSPECT_PATH" --recursive --check-md5 FailIfDifferent

echo "INSPECT downloaded to $INSPECT_PATH"
