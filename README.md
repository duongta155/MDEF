# CARE-RG: Candidate Aggregation with Recall-Enhanced Ensembling for 3D CT Report Generation

> Paper under double-blind review.

## Overview

Automated radiology report generation from 3D chest CT is dominated by a single architectural pattern. A 3D vision encoder feeds an autoregressive language model that produces one report via greedy or beam decoding. This single-trajectory paradigm commits to one realisation of which findings to mention per scan. Under a clinical label extractor, it manifests as high precision but bounded report-level recall on low-prevalence conditions.

CARE-RG is an inference-time deep ensemble. It converts a single trajectory into multiple uncorrelated trajectories. The trajectories are then fused at the clinical-label level rather than the text level. The full system has five stages, described in Section IV of the paper.

1. Stage 1 (Ensemble members): three fine-tuned generators that share a training recipe. They differ only in pre-trained initialisation and random seed.
2. Stage 2 (Stochastic decoding): each member emits the union of `K=10` candidate reports. Candidates are sampled at temperatures linearly staggered over `[0.7, 1.3]`.
3. Stage 3 (RadBERT label extraction): a pre-trained 18-condition extractor maps each member's text to a binary label vector.
4. Stage 4 (Per-condition logical-OR late fusion): the three label vectors are fused at the clinical-label level. The fusion geometrically attacks the recall ceiling.
5. Stage 5 (Tier-invariant inference policy): every inference-time hyperparameter is fixed once on RadGenome. The same values are reused unchanged on CT-RATE and INSPECT.

Under one fixed configuration, CARE-RG achieves CE F1 **0.3999** on RadGenome (in-distribution), **0.3381** on CT-RATE (cross-dataset), and **0.1813** on INSPECT (cross-institution). The improvements over the strongest single-trajectory baseline are `+0.079`, `+0.045`, and `+0.010` respectively.

## Table of Contents

1. [Requirements](#1-requirements)
2. [Dataset Setup](#2-dataset-setup)
3. [Reproduction Guide](#3-reproduction-guide)
4. [Training](#4-training)
5. [Inference and Fusion](#5-inference-and-fusion)
6. [Text Head and NLG Scoring](#6-text-head-and-nlg-scoring)
7. [Evaluation](#7-evaluation)
8. [Repository Structure](#8-repository-structure)

## 1. Requirements

### System Requirements

* Python 3.12 (required)
* CUDA 11.8 or newer
* GPU with at least 24 GB VRAM. We recommend 45 GB or more for the 7B language-model backbone with LoRA.
* 64 GB of RAM for CT-volume preprocessing
* 400 GB of disk space for the three datasets plus WDS shards and per-member predictions

### Installation

```bash
uv sync
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Dependencies

Key dependencies are declared in `pyproject.toml`.

* PyTorch 2.0+, transformers, accelerate, peft (for LoRA fine-tuning)
* webdataset (for volume shards)
* nltk, rouge_score (for NLG metrics)
* monai, nibabel (for CT volume IO and preprocessing)

## 2. Dataset Setup

### 2.1 Required Datasets

| Dataset | Description | Access |
|---------|-------------|--------|
| RadGenome-ChestCT | In-distribution training and validation | [HuggingFace dataset](https://huggingface.co/datasets/mychen76/RadGenome-ChestCT) |
| CT-RATE | Same-modality cross-dataset held-out tier | [HuggingFace dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
| INSPECT | Cross-institution Stanford CTPA cohort | [PhysioNet](https://physionet.org/content/inspect/1.0.0/) |
| RadBERT-RoBERTa-4m | 18-condition clinical extractor weights | [HuggingFace model](https://huggingface.co/zzxslp/RadBERT-RoBERTa-4m). The classifier head is supplied with the public Reg2RG release. |

INSPECT requires PhysioNet credentialed access. First complete the required training course at [CITI Program](https://about.citiprogram.org/).

### 2.2 Configuration

Edit `configs/storage.yaml` to point at your local data layout. This file is the single source of truth for paths. Every script and module reads from it.

```yaml
global:
  data_root: "/data"
  radgenome_path: "/data/radgenome_chestct"
  ctrate_path:    "/data/ctrate"
  inspect_path:   "/data/inspect"
  radbert_ckpt:   "/data/radbert/RadBertClassifier.pth"
```

## 3. Reproduction Guide

To reproduce the headline results of the paper, run the following stages in order. Each script reads paths from `configs/storage.yaml`.

### 3.1 Data preparation

```bash
bash scripts/data/01_download_radgenome.sh
bash scripts/data/02_download_ctrate.sh
bash scripts/data/03_download_inspect.sh
bash scripts/data/04_preprocess_to_wds.sh
```

The last step converts every volume to WebDataset shards.

### 3.2 Train the three ensemble members

Every member follows the same training recipe in `configs/care_rg_train.yaml`. Only the initialisation and seed change between members.

```bash
bash scripts/train/01_train_generator_R_seed1.sh
bash scripts/train/02_train_generator_R_seed2.sh
bash scripts/train/03_train_generator_P.sh
```

Checkpoints land in `checkpoints/generator_R_1/`, `checkpoints/generator_R_2/`, and `checkpoints/generator_P/`.

### 3.3 Inference and label-level fusion

```bash
bash scripts/inference/01_sd_union.sh
bash scripts/inference/02_or_fusion.sh
```

The first script runs Stage 2 and Stage 3. The second script runs Stage 4.

### 3.4 Text head (Stage 5) for NLG scoring

```bash
bash scripts/text_head/01_retrieve_report.sh
bash scripts/text_head/02_verify_boilerplate.sh
```

### 3.5 Evaluation

```bash
bash scripts/evaluation/01_ce_f1.sh
bash scripts/evaluation/02_nlg.sh
bash scripts/evaluation/03_bootstrap_cis.sh
bash scripts/evaluation/04_per_cond_pr_table.sh
```

The first script computes micro CE F1 and the per-condition table. The second script computes BLEU 1 through 4, METEOR, and ROUGE-L. The third script computes paired bootstrap 95% confidence intervals. The fourth script regenerates the LaTeX table body used in the paper.

## 4. Training

A single command-line invocation also works directly.

```bash
uv run python -m src.training.trainer \
    --config configs/care_rg_train.yaml \
    --seed 1 --init_from reg2rg \
    --output_dir checkpoints/generator_R_1
```

The trainer logs to `logs/{generator_name}/`. It saves a checkpoint per epoch (`save_every: 1`). Total wall time is roughly 24 hours per member on a single H100 GPU at `batch_size=8`.

## 5. Inference and Fusion

Run Stochastic Decoding (Stage 2) and RadBERT extraction (Stage 3) for one member on one tier as follows.

```bash
uv run python -m src.evaluation.evaluator \
    --design sd_union \
    --model_path checkpoints/generator_R_1/final_model.pt \
    --infer_config configs/care_rg_infer.yaml \
    --output_dir predictions/generator_R_1/rg_K10_UNION \
    --shard_dir /data/care_rg/processed/radgenome_wds \
    --eval_batch_size 8
```

After all three members have emitted their per-tier predictions, run the OR fusion (Stage 4) as a single Python command.

```bash
uv run python -m src.fusion.logical_or \
    --infer_config configs/care_rg_infer.yaml \
    --predictions_root predictions \
    --output_root predictions/ensemble_3way_or
```

## 6. Text Head and NLG Scoring

The Stage 5 text head Psi is deterministic. It takes the fused label and the three SD-union texts as input. The text head is invoked separately from the headline CE F1 pipeline. The CE F1 metric is invariant to any choice of head.

```bash
uv run python -m src.text_head.retrieve_report \
    --infer_config configs/care_rg_infer.yaml \
    --predictions_root predictions \
    --output_root predictions/retrieved
```

The boilerplate sentences used by the text head appear at least 10,000 times each in the RadGenome 24k training split. You can reproduce that audit yourself.

```bash
uv run python -m src.text_head.verify_boilerplate \
    --train_csv /data/radgenome_chestct/radgenome_train_reports.csv
```

## 7. Evaluation

The paper reports the following metrics.

* CE family (Table II): Precision, Recall, F1 micro-averaged over 18 RadBERT conditions. Per-condition P, R, F1 also appear in Table III.
* NLG family (Table II): BLEU 1 through 4, METEOR, ROUGE-L.
* Bootstrap CIs (Table IV): paired-volume bootstrap 95% confidence intervals for the CE F1 and NLG gaps.
* Per-condition table generator (Table III LaTeX body): outputs a `tabular` snippet ready to drop into the paper.

## 8. Repository Structure

```plaintext
care-rg/
├── configs/
│   ├── care_rg_train.yaml      Stage 1 fine-tuning recipe (shared across members)
│   ├── care_rg_infer.yaml      Stage 2 and Stage 5 tier-invariant constants
│   └── storage.yaml            paths
├── scripts/
│   ├── base.sh                 sourced by every script. Reads storage.yaml.
│   ├── data/                   dataset download and WDS preprocessing
│   ├── train/                  per-member training launchers
│   ├── inference/              Stage 2 SD-union and Stage 4 OR fusion
│   ├── text_head/              Stage 5 retrieval and boilerplate audit
│   └── evaluation/             CE, NLG, bootstrap CIs, per-condition table
├── checkpoints/
│   ├── generator_R_1/          M_R^(1) (Reg2RG-init, seed=1)
│   ├── generator_R_2/          M_R^(2) (Reg2RG-init, seed=2)
│   └── generator_P/            M_P (pathology-warmstart, seed=1)
└── src/
    ├── data/                   WebDataset readers, train and eval splits
    ├── models/                 3D-ViT vision encoder and region-grounded LM head
    ├── training/               training loop (Stage 1)
    ├── inference/              Stage 2 SD-union per member
    ├── fusion/                 Stage 3 plus Stage 4. RadBERT extractor and OR fusion.
    ├── text_head/              Stage 5 deterministic retrieved-sentence head
    ├── evaluation/             CE F1, NLG, bootstrap, per-condition table
    └── utils/                  shared utilities (logger, etc.)
```

## Citation

```bibtex
@article{care_rg_2026,
  title  = {CARE-RG: Candidate Aggregation with Recall-Enhanced Ensembling for 3D CT Report Generation},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License

This code is provided for research and review purposes only.
