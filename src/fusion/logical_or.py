#!/usr/bin/env python
"""Label-level GAV+PathB ensembling.

Each prediction gets its own full 512-token RadBERT pass, then per-condition
binary labels are combined across models via OR (any model detected),
AND (both models detected), or majority. This sidesteps the text-level
truncation that killed the previous text-concatenation ensemble.

Workflow per fusion:
  1. Dedupe each source predictions.csv by volume_name.
  2. Run RadBERT on Pred_combined_report of each source (cached on disk).
  3. Run RadBERT on GT_combined_report from each source (cached; should
     be identical across sources for the same dataset).
  4. Inner-join volumes present in ALL sources.
  5. Combine per-volume per-condition binary labels (OR / AND / MAJORITY).
  6. Compute micro/macro F1 against the GT label matrix.

Total wall time approximately 15-25 minutes for 5 fusions x 1 mode (OR).
Output: ensemble_gav_pathb_label/<fusion>__<mode>/ce_summary.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = "<REPO>"
ANALYSIS_ROOT = "<DATA_ROOT>/predictions"
OUT_ROOT = os.path.join(ANALYSIS_ROOT, "ensemble_gav_pathb_label")
CACHE_DIR = os.path.join(OUT_ROOT, "_radbert_cache")

# Need sys.path tweaks to import evaluate_ce's helpers and RadBERT classifier.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis/08_full_architecture_experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis/08_full_architecture_experiments/evaluation"))
sys.path.insert(0, os.path.join(PROJECT_ROOT,
                                "Baseline_Model/Reg2RG/evaluation/ce_evaluator_ct2rep"))

from configs.base_config import LABEL_COLS, NUM_CONDITIONS
from classifier import RadBertClassifier
from evaluate_ce import run_radbert_inference, compute_ce_metrics

RADBERT_PATH = os.path.join(
    PROJECT_ROOT, "Baseline_Model/Reg2RG/checkpoints/RadBertClassifier.pth")

GAV = os.path.join(ANALYSIS_ROOT, "n06_gav_24kwds/n06_generate_verify")
PATHB = os.path.join(ANALYSIS_ROOT, "n06_pathB_s01_24kwds/n06_generate_verify")
GAV2 = os.path.join(ANALYSIS_ROOT, "n06_gav2_seed1337_24kwds/n06_generate_verify")

# Fusion table: (name, [source predictions.csv paths]).
FUSIONS = [
    ("rg_K10U_K10U",
     [os.path.join(GAV, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_K10_UNION/predictions.csv")]),
    ("ctrate_K10U_K10U",
     [os.path.join(GAV, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "ctrate_K10_UNION/predictions.csv")]),
    # CT-Rate 4-way (added 2026-05-21 for consistent-configuration paper claim:
    # the committed CARE-RG architecture is the same 4-way OR on every tier).
    ("ctrate_4way",
     [os.path.join(GAV, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(GAV, "ctrate_TTA3/predictions.csv"),
      os.path.join(PATHB, "ctrate_TTA3/predictions.csv")]),
    ("rg_TTA3_TTA3",
     [os.path.join(GAV, "eval_radgenome_TTA3/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_TTA3/predictions.csv")]),
    ("rg_K10U_TTA3",
     [os.path.join(GAV, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_TTA3/predictions.csv")]),
    ("rg_4way",
     [os.path.join(GAV, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(GAV, "eval_radgenome_TTA3/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_TTA3/predictions.csv")]),
    # INSPECT third-tier validation (added 2026-05-19 evening once all 4 INSPECT predictions landed)
    ("inspect_K10U_K10U",
     [os.path.join(GAV, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_K10_UNION_wds/predictions.csv")]),
    ("inspect_TTA3_TTA3",
     [os.path.join(GAV, "inspect_TTA3_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_TTA3_wds/predictions.csv")]),
    ("inspect_4way",
     [os.path.join(GAV, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(GAV, "inspect_TTA3_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_TTA3_wds/predictions.csv")]),
    # 3-seed × 2-mode = 6-way fusion (added 2026-05-22 once GAV2 seed=1337 evals
    # finished). 3-way K10U-only and 3-way TTA3-only fusions test whether
    # multi-seed alone (GAV+GAV2) closes the gap vs multi-init (GAV+PathB).
    ("rg_3way_K10U",
     [os.path.join(GAV, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(GAV2, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_K10_UNION/predictions.csv")]),
    ("rg_3way_TTA3",
     [os.path.join(GAV, "eval_radgenome_TTA3/predictions.csv"),
      os.path.join(GAV2, "eval_radgenome_TTA3/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_TTA3/predictions.csv")]),
    ("rg_6way",
     [os.path.join(GAV, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(GAV2, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_K10_UNION/predictions.csv"),
      os.path.join(GAV, "eval_radgenome_TTA3/predictions.csv"),
      os.path.join(GAV2, "eval_radgenome_TTA3/predictions.csv"),
      os.path.join(PATHB, "eval_radgenome_TTA3/predictions.csv")]),
    ("ctrate_3way_K10U",
     [os.path.join(GAV, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(GAV2, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "ctrate_K10_UNION/predictions.csv")]),
    ("ctrate_3way_TTA3",
     [os.path.join(GAV, "ctrate_TTA3/predictions.csv"),
      os.path.join(GAV2, "ctrate_TTA3/predictions.csv"),
      os.path.join(PATHB, "ctrate_TTA3/predictions.csv")]),
    ("ctrate_6way",
     [os.path.join(GAV, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(GAV2, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(PATHB, "ctrate_K10_UNION/predictions.csv"),
      os.path.join(GAV, "ctrate_TTA3/predictions.csv"),
      os.path.join(GAV2, "ctrate_TTA3/predictions.csv"),
      os.path.join(PATHB, "ctrate_TTA3/predictions.csv")]),
    ("inspect_3way_K10U",
     [os.path.join(GAV, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(GAV2, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_K10_UNION_wds/predictions.csv")]),
    ("inspect_3way_TTA3",
     [os.path.join(GAV, "inspect_TTA3_wds/predictions.csv"),
      os.path.join(GAV2, "inspect_TTA3_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_TTA3_wds/predictions.csv")]),
    ("inspect_6way",
     [os.path.join(GAV, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(GAV2, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_K10_UNION_wds/predictions.csv"),
      os.path.join(GAV, "inspect_TTA3_wds/predictions.csv"),
      os.path.join(GAV2, "inspect_TTA3_wds/predictions.csv"),
      os.path.join(PATHB, "inspect_TTA3_wds/predictions.csv")]),

    # 3-way HIGHTEMP K10U variants (added 2026-05-27 to test HIGHTEMP > default).
    ('rg_3way_HIGHTEMP_K10U',
     [os.path.join(GAV,   'eval_radgenome_K10_UNION_HIGHTEMP/predictions.csv'),
      os.path.join(GAV2,  'eval_radgenome_K10_UNION_HIGHTEMP/predictions.csv'),
      os.path.join(PATHB, 'eval_radgenome_K10_UNION_HIGHTEMP/predictions.csv')]),
    ('ctrate_3way_HIGHTEMP_K10U',
     [os.path.join(GAV,   'ctrate_K10_UNION_HIGHTEMP/predictions.csv'),
      os.path.join(GAV2,  'ctrate_K10_UNION_HIGHTEMP/predictions.csv'),
      os.path.join(PATHB, 'ctrate_K10_UNION_HIGHTEMP/predictions.csv')]),
    ('inspect_3way_HIGHTEMP_K10U',
     [os.path.join(GAV,   'inspect_K10_UNION_HIGHTEMP_wds/predictions.csv'),
      os.path.join(GAV2,  'inspect_K10_UNION_HIGHTEMP_wds/predictions.csv'),
      os.path.join(PATHB, 'inspect_K10_UNION_HIGHTEMP_wds/predictions.csv')]),
]


def _cache_key(predictions_csv, column):
    src = predictions_csv.replace(ANALYSIS_ROOT + "/", "").replace("/", "__")
    return os.path.join(CACHE_DIR, f"{src}__{column}.npz")


def get_or_compute_labels(predictions_csv, column, model, device):
    """Run RadBERT on a column of a CSV, cached by (csv_path, column)."""
    cache_file = _cache_key(predictions_csv, column)
    if os.path.exists(cache_file):
        d = np.load(cache_file, allow_pickle=True)
        return d["labels"], d["volumes"]

    df = pd.read_csv(predictions_csv).drop_duplicates(subset=["volume_name"],
                                                       keep="first")
    df = df[df[column].notna() & (df[column].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    print(f"  RadBERT pass on {column} of {os.path.basename(os.path.dirname(predictions_csv))} "
          f"({len(df)} samples)")
    labels = run_radbert_inference(df, column, model, device, batch_size=16)
    volumes = df["volume_name"].astype(str).tolist()
    np.savez(cache_file, labels=labels, volumes=np.array(volumes))
    return labels, np.array(volumes)


def align(labels, volumes, common):
    """Reorder labels to match the common volume ordering."""
    v_to_i = {v: i for i, v in enumerate(volumes.tolist()
                                          if hasattr(volumes, "tolist") else volumes)}
    return np.stack([labels[v_to_i[v]] for v in common], axis=0)


def fuse(name, source_paths, model, device, mode):
    out_dir = os.path.join(OUT_ROOT, f"{name}__{mode}")
    os.makedirs(out_dir, exist_ok=True)
    ce_file = os.path.join(out_dir, "ce_summary.json")
    if os.path.exists(ce_file):
        with open(ce_file) as f:
            r = json.load(f)
        print(f"  [{name} {mode}] skip (done) f1={r.get('micro_f1', 0):.4f}")
        return r.get("micro_f1", 0)

    print(f"\n{'=' * 70}")
    print(f"  Fusion: {name}  mode={mode}  sources={len(source_paths)}")
    print(f"{'=' * 70}")

    pred_labels = []
    pred_volumes = []
    for src in source_paths:
        labels, vols = get_or_compute_labels(src, "Pred_combined_report",
                                              model, device)
        pred_labels.append(labels)
        pred_volumes.append(vols)
    gt_labels, gt_vols = get_or_compute_labels(source_paths[0],
                                                "GT_combined_report",
                                                model, device)

    common = set(gt_vols.tolist() if hasattr(gt_vols, "tolist") else gt_vols)
    for v in pred_volumes:
        common &= set(v.tolist() if hasattr(v, "tolist") else v)
    common = sorted(common)
    print(f"  Common volumes across {len(source_paths) + 1} sources (incl GT): {len(common)}")
    if len(common) == 0:
        print(f"  ABORT: no common volumes")
        return None

    aligned_preds = [align(l, v, common) for l, v in zip(pred_labels, pred_volumes)]
    aligned_gt = align(gt_labels, gt_vols, common)

    stacked = np.stack(aligned_preds, axis=0)  # (M, N, C)
    if mode == "or":
        combined = stacked.any(axis=0).astype(int)
    elif mode == "and":
        combined = stacked.all(axis=0).astype(int)
    elif mode == "majority":
        thresh = len(aligned_preds) / 2.0
        combined = (stacked.sum(axis=0) > thresh).astype(int)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    per_cond, summary = compute_ce_metrics(aligned_gt, combined, LABEL_COLS)
    pd.DataFrame(per_cond).to_csv(
        os.path.join(out_dir, "ce_per_condition.csv"), index=False)
    with open(ce_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  CE: P={summary['micro_precision']:.4f}  "
          f"R={summary['micro_recall']:.4f}  "
          f"F1={summary['micro_f1']:.4f}  "
          f"macro_F1={summary['macro_f1']:.4f}")
    return summary["micro_f1"]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("=" * 70)
    print("  Label-level GAV+PathB ensemble")
    print("=" * 70)
    print(f"  Output root: {OUT_ROOT}")
    print(f"  RadBERT cache: {CACHE_DIR}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading RadBERT classifier on {device}...")
    model = RadBertClassifier(n_classes=NUM_CONDITIONS)
    model.load_state_dict(torch.load(RADBERT_PATH, map_location=device),
                           strict=False)
    model = model.to(device)
    print("Loaded.\n")

    # Mode plan: OR for everything (the primary ensemble hypothesis),
    # plus AND and MAJORITY for the 4-way fusion as diagnostics.
    results = {}
    for name, paths in FUSIONS:
        modes = ["or"]
        if len(paths) >= 3:
            modes.append("majority")
        if len(paths) == 2:
            modes.append("and")
        for mode in modes:
            try:
                results[f"{name}__{mode}"] = fuse(name, paths, model, device, mode)
            except Exception as e:
                print(f"  [{name} {mode}] FAILED: {e}")
                results[f"{name}__{mode}"] = None

    print("\n" + "=" * 70)
    print("  SUMMARY vs current bests and text-level ensemble")
    print("=" * 70)
    print(f"\n  CURRENT SINGLE-MODEL BESTS:")
    print(f"    PathB+TTA3            RG=0.3539  (in-dist 🏆)")
    print(f"    GAV+K10+UNION         CT-Rate=0.2983  (cross-tier 🏆)")
    print(f"\n  TEXT-LEVEL ENSEMBLE (truncation-limited):")
    print(f"    CT-Rate K10U+K10U     0.303  (+0.005 small win)")
    print(f"    RG K10U+K10U          0.3231 (truncated, = GAV alone)")
    print()
    print(f"  {'fusion__mode':40s}  {'F1':>8s}")
    print(f"  {'-' * 40}  {'-' * 8}")
    for key, f1 in results.items():
        s = f"{f1:.4f}" if f1 is not None else "FAILED"
        print(f"  {key:40s}  {s:>8s}")


if __name__ == "__main__":
    main()
