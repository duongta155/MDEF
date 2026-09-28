#!/usr/bin/env python3
"""
bootstrap_nlg_cis.py
====================

Paired-volume bootstrap 95% CIs for the v3 retrieved-NLG row vs each
baseline (Reg2RG, CT-GRAPH, MARCH) on every tier and every NLG metric.
Mirrors paper Table V's CE-F1 bootstrap CIs at the NLG level.

For each (tier, baseline, metric):
  1. Align baseline + CARE-RG predictions by volume_name (intersection).
  2. Bootstrap B=2000 resamples WITH REPLACEMENT over volume indices.
  3. For each resample, compute (NLG_CARE-RG - NLG_baseline).
  4. Report mean, lower 2.5%, upper 97.5% as the 95% CI.

GPU is used for batch RadBERT inference on the resampled subsets is NOT
required -- BLEU/METEOR/ROUGE are CPU-only -- BUT we fill VRAM by running
B in parallel on GPU via vectorised torch ops where possible. The Python
NLG libs (nltk corpus_bleu, rouge_score) are CPU, so we run B bootstraps
in a process pool with 16 workers, each computing metrics for one
resample. This fills CPU not VRAM; we still pin to a GPU for the
companion RadBERT label-consistency check at the end (see below).

We additionally compute, for each tier:
  * RadBERT(v3_retrieved_text) -> verify the text head's labels match
    the fused 18-vector that produced CE F1 0.3999 / 0.3381 / 0.1813
    (i.e., the text head is a presentation layer, not a label rewriter).
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

A = "<DATA_ROOT>/predictions"
OUT_ROOT = os.path.join(A, "care_rg_bootstrap_nlg")
CLASSIFIER_DIR = ("<USER_ROOT>/source_code/"
                   "Medical_Report_Generation/Baseline_Model/"
                   "Reg2RG/evaluation/ce_evaluator_ct2rep")
RADBERT_PATH = ("<REPO>/"
                "Baseline_Model/Reg2RG/checkpoints/RadBertClassifier.pth")

TIER_TO_BASE = {
    "rg": "RG",
    "ctrate": "CT-Rate",
    "inspect": "INSPECT",
}

# Predictions paths per tier per system.
PRED_PATHS = {
    "rg": {
        "Reg2RG":   f"{A}/n_march_resident_24kwds/n_march_resident/eval_radgenome/predictions.csv",
        "CT-GRAPH": f"{A}/n_ct_graph_24kwds/n_ct_graph/eval_radgenome/predictions.csv",
        "MARCH":    f"{A}/n_march_resident_24kwds/n_march_resident/march_full_pipeline/predictions.csv",
        "CARE-RG":  f"{A}/care_rg_retrieved/rg/predictions.csv",
    },
    "ctrate": {
        "Reg2RG":   f"{A}/n_march_resident_24kwds/n_march_resident/ctrate/predictions.csv",
        "CT-GRAPH": f"{A}/n_ct_graph_24kwds/n_ct_graph/ctrate/predictions.csv",
        "MARCH":    f"{A}/n_march_resident_24kwds/n_march_resident/march_full_pipeline_ctrate/predictions.csv",
        "CARE-RG":  f"{A}/care_rg_retrieved/ctrate/predictions.csv",
    },
    "inspect": {
        "Reg2RG":   f"{A}/n_march_resident_24kwds/n_march_resident/inspect_wds/predictions.csv",
        "CT-GRAPH": f"{A}/n_ct_graph_24kwds/n_ct_graph/inspect_wds/predictions.csv",
        "MARCH":    f"{A}/n_march_resident_24kwds/n_march_resident/march_full_pipeline_inspect/predictions.csv",
        "CARE-RG":  f"{A}/care_rg_retrieved/inspect/predictions.csv",
    },
}


import re

_NII_RE = re.compile(r"\.nii\.gz$", re.IGNORECASE)
_LETTER_SUFFIX_RE = re.compile(r"_([a-z])(?:_\d+)?$", re.IGNORECASE)


def _canon_id(s):
    """Normalize volume IDs across baseline schemas:
        valid_1199_a_1.nii.gz   -> valid_1199a    (Reg2RG/CT-GRAPH/MARCH)
        valid_1199_a            -> valid_1199a    (other tier files)
        valid_1199a             -> valid_1199a    (GAV/CARE-RG, already canonical)
        PE4529af3               -> PE4529af3      (INSPECT, unchanged)
    """
    s = str(s).strip()
    s = _NII_RE.sub("", s)
    s = _LETTER_SUFFIX_RE.sub(r"\1", s)
    return s


def _detect_cols(df):
    if "volume_name" in df.columns:
        return "volume_name", "GT_combined_report", "Pred_combined_report"
    if "AccessionNo" in df.columns:
        return "AccessionNo", "GroundTruth", "Findings"
    raise ValueError(f"unknown columns: {list(df.columns)}")


def _load_preds(path):
    df = pd.read_csv(path)
    id_col, gt_col, pred_col = _detect_cols(df)
    df = df[[id_col, gt_col, pred_col]].copy()
    df.columns = ["id", "gt", "pred"]
    df = df[df["gt"].notna() & df["pred"].notna()]
    df["id"] = df["id"].astype(str).map(_canon_id)
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df


def _safe_meteor(ref_toks, hyp_toks):
    """Wrap nltk meteor_score to swallow degenerate Fraction(0,0) cases
    (both ref and hyp tokenize to empty / a single punctuation token after
    lowercasing). These cases return 0 from a 'real' metric perspective."""
    try:
        if not ref_toks or not hyp_toks:
            return 0.0
        from nltk.translate.meteor_score import meteor_score
        return float(meteor_score([ref_toks], hyp_toks))
    except (ZeroDivisionError, ValueError):
        return 0.0


def _safe_rouge_l(rs, ref, hyp):
    try:
        if not ref.strip() or not hyp.strip():
            return 0.0
        return float(rs.score(ref, hyp)["rougeL"].fmeasure)
    except (ZeroDivisionError, ValueError):
        return 0.0


def _compute_one_metric(gts, preds, metric):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    # Filter pairs where either side is empty after tokenization to avoid
    # Fraction(0, 0) in METEOR. Bootstrap with replacement can yield batches
    # full of edge cases, so be defensive.
    pairs = [(str(r) if r else "", str(h) if h else "")
             for r, h in zip(gts, preds)]
    pairs = [(r, h) for r, h in pairs if r.strip() and h.strip()]
    if not pairs:
        return 0.0
    gts, preds = zip(*pairs)
    if metric.startswith("BLEU"):
        n = int(metric.split("-")[1])
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        ref_toks = [[r.lower().split()] for r in gts]
        hyp_toks = [h.lower().split() for h in preds]
        sm = SmoothingFunction().method1
        try:
            return corpus_bleu(ref_toks, hyp_toks, weights=weights,
                                smoothing_function=sm) * 100
        except (ZeroDivisionError, ValueError):
            return 0.0
    if metric == "METEOR":
        return float(np.mean([_safe_meteor(r.lower().split(), h.lower().split())
                              for r, h in zip(gts, preds)])) * 100
    if metric == "ROUGE-L":
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(np.mean([_safe_rouge_l(rs, r, h)
                              for r, h in zip(gts, preds)])) * 100
    raise ValueError(metric)


def _resample_metric(args):
    # For BLEU we still compute corpus_bleu on the resampled subset because
    # corpus BLEU is NOT a simple average of per-sample BLEUs. For METEOR
    # and ROUGE-L the corpus score IS the mean of per-sample scores, so we
    # accept pre-computed per-sample score arrays via a fast path.
    idxs, gts_a, preds_a, gts_b, preds_b, metric = args
    try:
        if metric in ("METEOR_PRE", "ROUGE-L_PRE"):
            # gts_a/preds_a are unused; preds_b/gts_b carry the precomputed
            # per-sample float arrays.
            scores_a = preds_a  # numpy array, len = n_volumes
            scores_b = preds_b
            sub_a = float(np.mean([scores_a[i] for i in idxs]))
            sub_b = float(np.mean([scores_b[i] for i in idxs]))
            return sub_a - sub_b
        gts_sub_a = [gts_a[i] for i in idxs]
        preds_sub_a = [preds_a[i] for i in idxs]
        gts_sub_b = [gts_b[i] for i in idxs]
        preds_sub_b = [preds_b[i] for i in idxs]
        score_a = _compute_one_metric(gts_sub_a, preds_sub_a, metric)
        score_b = _compute_one_metric(gts_sub_b, preds_sub_b, metric)
        return score_a - score_b
    except Exception:
        return None


def _precompute_per_sample(gts, preds):
    """Returns numpy arrays of per-sample METEOR and ROUGE-L scores (in
    [0,100]). Each i-th score is the metric computed on (gts[i], preds[i])
    treating each sample independently. Robust to empties via _safe_*.
    Mean across all i = corpus METEOR / ROUGE-L (these metrics are linear
    per-sample averages, unlike corpus BLEU)."""
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    n = len(gts)
    meteor = np.zeros(n, dtype=np.float32)
    rouge_l = np.zeros(n, dtype=np.float32)
    for i in range(n):
        r = str(gts[i]) if gts[i] is not None else ""
        h = str(preds[i]) if preds[i] is not None else ""
        if not r.strip() or not h.strip():
            continue
        meteor[i] = _safe_meteor(r.lower().split(), h.lower().split()) * 100
        rouge_l[i] = _safe_rouge_l(rs, r, h) * 100
    return meteor, rouge_l


def bootstrap_pair(tier, baseline_name, metrics, B, n_workers, seed=0):
    paths = PRED_PATHS[tier]
    base = _load_preds(paths[baseline_name])
    care = _load_preds(paths["CARE-RG"])
    merged = base.merge(care, on="id", suffixes=("_b", "_a"))
    n = len(merged)
    print(f"  {tier:7s}  CARE-RG vs {baseline_name:9s}  n={n}")
    gts_a = merged["gt_a"].astype(str).tolist()
    preds_a = merged["pred_a"].astype(str).tolist()
    gts_b = merged["gt_b"].astype(str).tolist()
    preds_b = merged["pred_b"].astype(str).tolist()

    # Pre-compute METEOR and ROUGE-L per sample (CARE-RG and baseline).
    # These metrics are linear per-sample averages, so bootstrap on indices
    # is exactly equivalent to recomputing per resample but ~500x faster.
    pre_meteor_a, pre_rouge_a = None, None
    pre_meteor_b, pre_rouge_b = None, None
    if "METEOR" in metrics or "ROUGE-L" in metrics:
        t0 = time.time()
        print(f"    precomputing per-sample METEOR/ROUGE-L (n={n})...",
              end=" ", flush=True)
        pre_meteor_a, pre_rouge_a = _precompute_per_sample(gts_a, preds_a)
        pre_meteor_b, pre_rouge_b = _precompute_per_sample(gts_b, preds_b)
        print(f"({time.time()-t0:.1f}s)")

    rng = np.random.default_rng(seed)
    all_idxs = [rng.integers(0, n, size=n).tolist() for _ in range(B)]

    out = {}
    for metric in metrics:
        t0 = time.time()
        if metric == "METEOR":
            # Vectorized bootstrap: just resample the precomputed scores.
            idxs_arr = np.array(all_idxs)
            sub_a = pre_meteor_a[idxs_arr].mean(axis=1)
            sub_b = pre_meteor_b[idxs_arr].mean(axis=1)
            deltas = (sub_a - sub_b).tolist()
            mean = float(np.mean(deltas)); lo = float(np.percentile(deltas, 2.5))
            hi = float(np.percentile(deltas, 97.5)); pg = float((np.array(deltas) > 0).mean())
            out[metric] = {"mean_delta": mean, "ci_lower": lo, "ci_upper": hi,
                            "p_greater": pg, "n_resamples_ok": len(deltas)}
            print(f"    {metric:8s}  Δ={mean:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  "
                  f"P(Δ>0)={pg:.3f}  ({time.time()-t0:.1f}s)")
            continue
        if metric == "ROUGE-L":
            idxs_arr = np.array(all_idxs)
            sub_a = pre_rouge_a[idxs_arr].mean(axis=1)
            sub_b = pre_rouge_b[idxs_arr].mean(axis=1)
            deltas = (sub_a - sub_b).tolist()
            mean = float(np.mean(deltas)); lo = float(np.percentile(deltas, 2.5))
            hi = float(np.percentile(deltas, 97.5)); pg = float((np.array(deltas) > 0).mean())
            out[metric] = {"mean_delta": mean, "ci_lower": lo, "ci_upper": hi,
                            "p_greater": pg, "n_resamples_ok": len(deltas)}
            print(f"    {metric:8s}  Δ={mean:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  "
                  f"P(Δ>0)={pg:.3f}  ({time.time()-t0:.1f}s)")
            continue
        # BLEU: keep the multiprocessing path (corpus BLEU is non-additive).
        deltas = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_resample_metric,
                                 (idxs, gts_a, preds_a, gts_b, preds_b, metric))
                       for idxs in all_idxs]
            for f in as_completed(futures):
                v = f.result()
                if v is not None:
                    deltas.append(v)
        deltas = np.array(deltas)
        if len(deltas) == 0:
            print(f"    {metric:8s}  ALL RESAMPLES FAILED")
            out[metric] = {"mean_delta": 0.0, "ci_lower": 0.0,
                            "ci_upper": 0.0, "p_greater": 0.0,
                            "n_resamples_ok": 0}
            continue
        mean = float(deltas.mean())
        lo = float(np.percentile(deltas, 2.5))
        hi = float(np.percentile(deltas, 97.5))
        out[metric] = {"mean_delta": mean, "ci_lower": lo, "ci_upper": hi,
                        "p_greater": float((deltas > 0).mean())}
        dt = time.time() - t0
        print(f"    {metric:8s}  Δ={mean:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  "
              f"P(Δ>0)={out[metric]['p_greater']:.3f}  ({dt:.1f}s)")
    return out


def label_consistency_check(tier, device):
    """Run RadBERT on v3 retrieved text. Compare to the fused 18-vector
    that produced CE F1. A perfect text head would yield identical
    labels; a lossy text head loses some conditions to negation/
    truncation. Report micro-F1(RadBERT(text_head) vs fused OR labels)
    -- this is an upper bound for paper sanity-check."""
    import torch
    sys.path.insert(0, CLASSIFIER_DIR)
    from classifier import RadBertClassifier
    from transformers import AutoTokenizer

    print(f"  loading RadBERT for {tier} label-consistency check...")
    tok = AutoTokenizer.from_pretrained("zzxslp/RadBERT-RoBERTa-4m",
                                         do_lower_case=True)
    model = RadBertClassifier(n_classes=18)
    model.load_state_dict(torch.load(RADBERT_PATH, map_location=device),
                          strict=False)
    model.eval().to(device)

    # Load retrieved text + the fused labels (already cached in
    # ensemble_gav_pathb_label/_radbert_cache/ from the 3 proposers,
    # OR-fused on the fly).
    cache = os.path.join(A, "ensemble_gav_pathb_label", "_radbert_cache")
    nm = {"rg": "eval_radgenome_K10_UNION", "ctrate": "ctrate_K10_UNION",
          "inspect": "inspect_K10_UNION_wds"}[tier]
    fns = [
        f"n06_gav_24kwds__n06_generate_verify__{nm}__predictions.csv__Pred_combined_report.npz",
        f"n06_gav2_seed1337_24kwds__n06_generate_verify__{nm}__predictions.csv__Pred_combined_report.npz",
        f"n06_pathB_s01_24kwds__n06_generate_verify__{nm}__predictions.csv__Pred_combined_report.npz",
    ]
    labels = []
    vols_ref = None
    for fn in fns:
        d = np.load(os.path.join(cache, fn), allow_pickle=True)
        labels.append(d["labels"])
        if vols_ref is None:
            vols_ref = [str(v) for v in d["volumes"]]
    fused = np.stack(labels).any(axis=0).astype(int)

    # Load v3 retrieved
    care = _load_preds(PRED_PATHS[tier]["CARE-RG"])
    care = care.set_index("id").reindex(vols_ref)
    care = care[care["pred"].notna()]
    keep_idx = [i for i, v in enumerate(vols_ref) if v in care.index]
    fused = fused[keep_idx]
    texts = care["pred"].tolist()

    # Batched RadBERT inference on the text head outputs.
    preds = []
    bs = 32
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, return_tensors="pt", max_length=512,
                       padding="max_length", truncation=True)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            logits = model(ids, mask)
            preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # Micro-F1 of text-head labels vs fused labels.
    tp = int(((preds == 1) & (fused == 1)).sum())
    fp = int(((preds == 1) & (fused == 0)).sum())
    fn = int(((preds == 0) & (fused == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    res = {"n": int(len(texts)), "tp": tp, "fp": fp, "fn": fn,
           "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
    print(f"  text-head vs fused labels: P={p:.4f} R={r:.4f} F1={f1:.4f} "
          f"(n={len(texts)})")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", nargs="+", default=["rg", "ctrate", "inspect"])
    p.add_argument("--baselines", nargs="+", default=["Reg2RG", "CT-GRAPH", "MARCH"])
    p.add_argument("--metrics", nargs="+",
                   default=["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                            "METEOR", "ROUGE-L"])
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--skip_consistency", action="store_true")
    args = p.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available()
                          else "cpu")

    results = {}
    for tier in args.tiers:
        print(f"\n=== {tier.upper()} bootstrap ({args.bootstraps} resamples, "
              f"{args.workers} workers) ===")
        results[tier] = {}
        for base in args.baselines:
            results[tier][base] = bootstrap_pair(tier, base, args.metrics,
                                                  args.bootstraps,
                                                  args.workers)
        out_path = os.path.join(OUT_ROOT, f"{tier}_bootstrap_cis.json")
        with open(out_path, "w") as f:
            json.dump(results[tier], f, indent=2)
        print(f"  wrote {out_path}")

        if not args.skip_consistency:
            print(f"\n=== {tier.upper()} text-head label consistency check ===")
            cons = label_consistency_check(tier, device)
            with open(os.path.join(OUT_ROOT,
                                    f"{tier}_label_consistency.json"), "w") as f:
                json.dump(cons, f, indent=2)

    # Markdown table for paste into paper.
    md_path = os.path.join(OUT_ROOT, "bootstrap_summary.md")
    with open(md_path, "w") as f:
        f.write("# NLG bootstrap 95% CIs (CARE-RG v3 retrieved vs baselines)\n\n")
        for tier in args.tiers:
            f.write(f"## {tier.upper()}\n\n")
            f.write("| Baseline | Metric | ΔNLG | 95% CI | P(Δ>0) |\n")
            f.write("|---|---|---|---|---|\n")
            for base in args.baselines:
                for m in args.metrics:
                    d = results[tier][base][m]
                    f.write(f"| {base} | {m} | {d['mean_delta']:+.2f} | "
                            f"[{d['ci_lower']:+.2f}, {d['ci_upper']:+.2f}] | "
                            f"{d['p_greater']:.3f} |\n")
            f.write("\n")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
