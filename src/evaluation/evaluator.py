"""
Generate predictions + evaluate CE for a trained novel architecture model.
Handles RadGenome, CT-RATE, and INSPECT validation sets.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_ROOT))
BASELINE_SRC = os.path.join(PROJECT_ROOT, "Baseline_Model", "Reg2RG", "src")

sys.path.insert(0, EXPERIMENT_ROOT)
sys.path.insert(0, BASELINE_SRC)

from configs.base_config import (
    REGIONS, RESULTS_ROOT, CACHE_DIR, LABEL_COLS,
    LLAMA_PATH, VIT3D_PATH, PERCEIVER_PATH, BASELINE_CKPT,
    VALID_DATA_FOLDER, VALID_MASK_FOLDER, VALID_REPORT_FILE,
    REFERENCE_SCORES,
)
from configs.experiment_configs import get_novel_experiment_config


def fmt_time(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"


def _to_tensor(t):
    return t.as_tensor() if hasattr(t, 'as_tensor') else t


def inference_collator(instances):
    lang_xs = [inst['lang_x'] for inst in instances]
    vision_xs = [inst['vision_x'] for inst in instances]
    mask_xs = [inst['mask_x'] for inst in instances]
    region2areas = [inst['region2area'] for inst in instances]
    acc_nums = [inst.get('acc_num', '') for inst in instances]
    gt_reports = [inst.get('gt_combined_report', '') for inst in instances]

    max_len = max(x.shape[0] for x in lang_xs)
    padded = []
    for x in lang_xs:
        if x.shape[0] < max_len:
            padded.append(torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=x.dtype)]))
        else:
            padded.append(x)
    lang_xs = _to_tensor(torch.stack(padded, dim=0))

    vision_temp = {a: [] for a in REGIONS}
    mask_temp = {a: [] for a in REGIONS}
    vs = next(iter(vision_xs[0].values())).shape
    ms = next(iter(mask_xs[0].values())).shape
    useless = []
    for area in REGIONS:
        flag = False
        for i in range(len(vision_xs)):
            if area in vision_xs[i]:
                vision_temp[area].append(vision_xs[i][area])
                mask_temp[area].append(mask_xs[i][area])
                flag = True
            else:
                vision_temp[area].append(torch.zeros(vs))
                mask_temp[area].append(torch.zeros(ms))
        if not flag:
            useless.append(area)
    images = torch.cat([_to_tensor(v['image']).unsqueeze(0) for v in vision_xs], dim=0)
    for a in useless:
        vision_temp.pop(a)
        mask_temp.pop(a)
    useful = list(vision_temp.keys())
    vision_xs = {a: _to_tensor(torch.cat([_.unsqueeze(0) for _ in vision_temp[a]], dim=0)) for a in useful}
    vision_xs['image'] = _to_tensor(images)
    mask_xs = {a: _to_tensor(torch.cat([_.unsqueeze(0) for _ in mask_temp[a]], dim=0)) for a in useful}

    return dict(lang_x=lang_xs, vision_x=vision_xs, mask_x=mask_xs,
                region2area=region2areas, acc_nums=acc_nums, gt_reports=gt_reports)


def run_ce_evaluation(predictions_csv, output_dir, gpu_id=0,
                       gt_column=None, pred_column=None):
    """Run the proven 08-pipeline evaluate_ce.py as a subprocess.

    gt_column/pred_column override evaluate_ce.py's defaults
    (GT_combined_report / Pred_combined_report). MARCH pipeline writes
    GroundTruth / Findings, so its caller must pass those.
    """
    import subprocess
    eval_script = os.path.join(PROJECT_ROOT, "analysis/08_full_architecture_experiments/evaluation/evaluate_ce.py")
    radbert_path = os.path.join(PROJECT_ROOT, "Baseline_Model/Reg2RG/checkpoints/RadBertClassifier.pth")

    if not os.path.exists(eval_script):
        print(f"  ERROR: {eval_script} not found")
        return None

    print(f"  Running 08-pipeline CE evaluation...")
    env = os.environ.copy()
    cmd = [sys.executable, eval_script,
           "--predictions_csv", predictions_csv,
           "--radbert_path", radbert_path,
           "--output_dir", output_dir,
           "--gpu_id", "0"]
    if gt_column is not None:
        cmd += ["--gt_column", gt_column]
    if pred_column is not None:
        cmd += ["--pred_column", pred_column]
    # Use THIS python interpreter (venv-aware), not system python3
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  CE evaluation failed (exit {result.returncode})")
        print(f"  stdout: {result.stdout[-1000:]}")
        print(f"  stderr: {result.stderr[-1000:]}")
        return None

    # Read back the ce_summary.json produced by evaluate_ce.py
    ce_file = os.path.join(output_dir, "ce_summary.json")
    if os.path.exists(ce_file):
        with open(ce_file) as f:
            results = json.load(f)
        print(f"  CE: P={results.get('micro_precision', 0):.4f} "
              f"R={results.get('micro_recall', 0):.4f} "
              f"F1={results.get('micro_f1', 0):.4f}")
        return results
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, default=VALID_DATA_FOLDER)
    parser.add_argument("--mask_folder", type=str, default=VALID_MASK_FOLDER)
    parser.add_argument("--report_file", type=str, default=VALID_REPORT_FILE)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for generation. H100 95GB can fit 8+; L40S 46GB fits 2-4.")
    parser.add_argument("--shard_dir", type=str, default=None,
                        help="If set, load eval set from WDS tar shards "
                             "(faster: no NFS random-access). When provided, "
                             "--data_folder / --mask_folder / --report_file "
                             "are ignored.")
    parser.add_argument("--proposer_ckpt", type=str, default=None,
                        help="Path A zero-shot composition: after loading --model_path "
                             "(verifier weights), overwrite proposer/base-model weights "
                             "from this ckpt. Use to test e.g. GAV-best-verifier + S01 "
                             "without any joint retraining (T1.5.1 in experimental_plan.md).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Trust CUDA_VISIBLE_DEVICES set by the shell launcher — don't override.
    device = torch.device("cuda:0")

    design_config = get_novel_experiment_config(args.design)

    print(f"\n{'=' * 60}")
    print(f"  EVAL: {args.design}")
    print(f"{'=' * 60}")

    # Skip prediction generation if predictions.csv already exists AND ce_summary too
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    ce_path = os.path.join(args.output_dir, "ce_summary.json")
    if os.path.exists(csv_path) and not os.path.exists(ce_path):
        # File exists but no CE yet. Could be either:
        #  - whole eval done, CE crashed/skipped -> run CE only
        #  - partial save from prior killed run -> resume below
        # Heuristic: if predictions.csv has >= 0.95 * dataset_size rows, treat
        # as complete and run CE. Otherwise resume.
        # We don't know dataset size yet, so always read first and decide.
        pass

    # Resume support: if predictions.csv exists from a killed run, load already-
    # generated rows and skip those acc_nums in the dataset.
    existing_rows = []
    done_acc_nums = []
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            existing_rows = existing_df.to_dict("records")
            done_acc_nums = [str(x) for x in existing_df["volume_name"].tolist()
                             if isinstance(x, str) and x]
            print(f"  Resume: found {len(existing_rows)} prior predictions, "
                  f"skipping those acc_nums")
        except Exception as e:
            print(f"  Warning: predictions.csv exists but unreadable ({e}); ignoring.")
            existing_rows = []
            done_acc_nums = []

    # Load dataset — WDS shards if --shard_dir, else NFS random-access.
    if args.shard_dir:
        from Dataset.radgenome_wds_dataset import RadGenomeWDSDataset_Test
        print(f"  Using WDS-shard reader: {args.shard_dir}")
        dataset = RadGenomeWDSDataset_Test(
            text_tokenizer=LLAMA_PATH,
            shard_dir=args.shard_dir,
            inferenced_id=done_acc_nums,
        )
    else:
        from Dataset.radgenome_dataset_test import RadGenomeDataset_Test
        dataset = RadGenomeDataset_Test(
            text_tokenizer=LLAMA_PATH,
            data_folder=args.data_folder,
            mask_folder=args.mask_folder,
            csv_file=args.report_file,
            cache_dir=CACHE_DIR,
            inferenced_id=done_acc_nums,
        )

    # If existing predictions already cover the whole eval set, skip generation
    # and go straight to CE.
    if existing_rows and len(dataset) == 0:
        print(f"  All {len(existing_rows)} samples already predicted, CE only.")
        ce_results = run_ce_evaluation(csv_path, args.output_dir, args.gpu_id)
        if ce_results:
            ref = REFERENCE_SCORES["baseline"]
            delta = ce_results["micro_f1"] - ref["radgenome_f1"]
            print(f"\n  vs baseline (0.2775): {'+' if delta > 0 else ''}{delta:.4f}")
        return
    if args.max_samples:
        import torch.utils.data as du
        dataset = du.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    print(f"  Dataset: {len(dataset)} samples")

    # Load model
    from Model.Reg2RG import Reg2RG
    base_model = Reg2RG(
        text_tokenizer_path=LLAMA_PATH, lang_model_path=LLAMA_PATH,
        pretrained_visual_encoder=VIT3D_PATH, pretrained_adapter=PERCEIVER_PATH,
    )
    from training.train_novel import build_novel_model
    model = build_novel_model(base_model, args.design, design_config)
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    # Free the GAV/main ckpt dict before loading the proposer ckpt to keep
    # peak CPU RSS down — important for 3-way concurrent Path A on the
    # SLURM cgroup. Without these dels, all three pairings hold both ckpt
    # dicts during merge and OOM on a 160 GB cap.
    import gc
    del ckpt
    gc.collect()

    # Path A composition: overwrite the proposer/base half of the model with
    # weights from a separate ckpt (e.g. an AbnormCond-tuned model), keeping
    # the verifier/scorer weights from --model_path. The two ckpts may differ
    # in key prefix (n06 wraps base in `self.base_model`, while S01/D01/etc.
    # are direct Reg2RG subclasses), so we try direct match, prefix-add, and
    # prefix-strip strategies and pick the one that overlaps most.
    if args.proposer_ckpt:
        # Move model to GPU FIRST so the CPU copy of the LM weights is freed
        # before we load the proposer ckpt. This brings peak CPU during the
        # merge from ~42 GB down to ~25 GB.
        model = model.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Path A: loading proposer ckpt from {args.proposer_ckpt}")
        proposer_state = torch.load(args.proposer_ckpt, map_location="cpu", weights_only=False)
        model_keys = set(model.state_dict().keys())

        candidates = {"direct": {}, "add_base_model_prefix": {}, "strip_base_model_prefix": {}}
        for k, v in proposer_state.items():
            if k in model_keys:
                candidates["direct"][k] = v
            pref_k = f"base_model.{k}"
            if pref_k in model_keys:
                candidates["add_base_model_prefix"][pref_k] = v
            if k.startswith("base_model."):
                stripped = k[len("base_model."):]
                if stripped in model_keys:
                    candidates["strip_base_model_prefix"][stripped] = v
        strategy, subset = max(candidates.items(), key=lambda kv: len(kv[1]))
        if not subset:
            raise ValueError(
                f"Path A: no key overlap between proposer ckpt and model state_dict. "
                f"proposer has {len(proposer_state)} keys, model has {len(model_keys)}; "
                f"none matched under any prefix strategy.")
        print(f"  Path A: strategy='{strategy}', overwriting {len(subset)} keys "
              f"(proposer total: {len(proposer_state)}, model total: {len(model_keys)})")
        # Move the subset to the model's current device before load_state_dict
        # to avoid PyTorch silently materialising another CPU copy.
        subset = {k: v.to(device) for k, v in subset.items()}
        missing, unexpected = model.load_state_dict(subset, strict=False)
        if unexpected:
            print(f"  Path A: warning, {len(unexpected)} unexpected keys ignored: {unexpected[:3]}...")
        print(f"  Path A: proposer half overwritten; verifier weights from --model_path retained")
        del proposer_state, subset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = model.to(device).eval()
    print(f"  Model loaded")

    # Generate — batch_size=4 to use H100 VRAM (14GB model + ~12GB activations per sample)
    EVAL_BATCH = args.eval_batch_size
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False, num_workers=2,
                        collate_fn=inference_collator, pin_memory=True)

    # Build per-design generate-time hooks. n17 SCVS needs a radbert_extract
    # callable [str -> (C,) float tensor] to score K sampled reports during
    # consensus voting; without it the model raises ValueError.
    extra_gen_kwargs = {}
    if args.design == "n17_self_consistency":
        radbert_path = os.path.join(PROJECT_ROOT, "Baseline_Model/Reg2RG/checkpoints/RadBertClassifier.pth")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Baseline_Model/Reg2RG/evaluation/ce_evaluator_ct2rep"))
        from classifier import RadBertClassifier
        from transformers import AutoTokenizer
        rb_model = RadBertClassifier(n_classes=len(LABEL_COLS))
        rb_state = torch.load(radbert_path, map_location="cpu", weights_only=False)
        rb_model.load_state_dict(rb_state, strict=False)
        rb_model = rb_model.to(device).eval()
        rb_tok = AutoTokenizer.from_pretrained("zzxslp/RadBERT-RoBERTa-4m",
                                               cache_dir=CACHE_DIR, do_lower_case=True)
        @torch.no_grad()
        def radbert_extract(report_text):
            enc = rb_tok(report_text, return_tensors="pt", truncation=True,
                         max_length=512, padding=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            logits = rb_model(input_ids, attn)
            probs = torch.sigmoid(logits).squeeze(0)  # (C,)
            return probs.float().cpu()
        extra_gen_kwargs["radbert_extract"] = radbert_extract
        print(f"  n17 hook: radbert_extract ready (RadBertClassifier + zzxslp/RadBERT-RoBERTa-4m)")

    # Resume-safe predictions: results starts populated with prior runs' rows.
    # Save incrementally every SAVE_EVERY samples via atomic temp+rename so a
    # mid-eval kill loses at most SAVE_EVERY samples of work.
    results = list(existing_rows)
    sample_counter = len(existing_rows)
    SAVE_EVERY = 50
    last_saved = sample_counter

    def _save_atomic():
        tmp = csv_path + ".tmp"
        pd.DataFrame(results).to_csv(tmp, index=False)
        os.replace(tmp, csv_path)

    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            lang_x = batch["lang_x"].to(device)
            vision_x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch["vision_x"].items()}
            mask_x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch["mask_x"].items()}
            region2area = batch["region2area"]

            # Test-Time Augmentation (TTA) — env-gated.
            # When USE_TTA=N (N>=2), apply N input perturbations per sample
            # and concatenate generated reports. CE evaluator extracts the
            # union of conditions from text → improved recall on OOD.
            # Free at training cost; ~N× inference cost. Added 2026-05-15.
            tta_n = int(os.environ.get("USE_TTA", "1"))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if tta_n <= 1:
                    reports = model.generate(lang_x, vision_x, mask_x, region2area, **extra_gen_kwargs)
                else:
                    tta_reports = [None]  # placeholder for ensemble
                    bsz_in = lang_x.shape[0]
                    accum = [[] for _ in range(bsz_in)]
                    for tta_i in range(tta_n):
                        # Apply different intensity scaling per pass (in fp16-safe range).
                        vision_x_tta = {}
                        for k, v in vision_x.items():
                            if isinstance(v, torch.Tensor):
                                scale = 0.9 + 0.05 * tta_i
                                shift = -0.05 + 0.025 * tta_i
                                vision_x_tta[k] = v * scale + shift
                            else:
                                vision_x_tta[k] = v
                        out = model.generate(lang_x, vision_x_tta, mask_x, region2area, **extra_gen_kwargs)
                        if not isinstance(out, list): out = [out]
                        for j in range(min(len(out), bsz_in)):
                            accum[j].append(out[j])
                        # Drop intermediate tensors between TTA passes — vision_x_tta
                        # and `out` are no longer needed once accum is updated. Without
                        # this, the next iteration's model.generate() hits OOM on L40S
                        # because PyTorch keeps the previous pass's activations + kv-cache
                        # cached even though they're unreferenced.
                        del vision_x_tta, out
                        torch.cuda.empty_cache()
                    # Concatenate all TTA outputs per sample (union of conditions)
                    reports = [" ".join(cands) for cands in accum]

            if not isinstance(reports, list):
                reports = [reports]
            bsz = len(reports)
            for j in range(bsz):
                results.append({
                    "sample_id": sample_counter,
                    "volume_name": batch["acc_nums"][j] if j < len(batch["acc_nums"]) else "",
                    "GT_combined_report": batch["gt_reports"][j] if j < len(batch["gt_reports"]) else "",
                    "Pred_combined_report": reports[j],
                })
                sample_counter += 1

            # Incremental atomic save every SAVE_EVERY new samples
            if sample_counter - last_saved >= SAVE_EVERY:
                _save_atomic()
                last_saved = sample_counter

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (len(loader) - i - 1)
                print(f"\r  [{(i+1)/len(loader)*100:5.1f}%] batch {i+1}/{len(loader)} "
                      f"({sample_counter} samples) ETA: {fmt_time(eta)}", end="", flush=True)

    print()
    _save_atomic()
    print(f"  Predictions saved: {csv_path} ({len(results)} samples)")

    # Run CE evaluation
    ce_results = run_ce_evaluation(csv_path, args.output_dir, args.gpu_id)

    # Compare to reference
    if ce_results:
        ref = REFERENCE_SCORES["baseline"]
        f1 = ce_results["micro_f1"]
        delta = f1 - ref["radgenome_f1"]
        print(f"\n  vs baseline (0.2775): {'+'if delta>0 else ''}{delta:.4f}")
        ref_s01 = REFERENCE_SCORES["s01_best_tune"]
        delta_s01 = f1 - ref_s01["radgenome_f1"]
        print(f"  vs best tune s01 (0.3043): {'+'if delta_s01>0 else ''}{delta_s01:.4f}")


if __name__ == "__main__":
    main()
