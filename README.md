# MDEF: Multi-Report Deep Ensemble Fusion for 3D CT Report Generation

Tung Duong Ta, Tim Oates, Manas Gaur, Trong-Nghia Nguyen, Tien-Cuong Nguyen, Tuan-Cuong Vuong, Trang Xuan Mai, and Thien Van Luong

> Accepted as a regular paper at the Workshop on Large Language Models for Multimodal Data Fusion (LLM4MDF), IEEE International Conference on Data Mining (ICDM 2026).

## Overview

Automated 3D chest CT report generation is dominated by a single-report paradigm. A 3D vision encoder feeds an autoregressive language model. The language model decodes one textual report per input volume via greedy or beam search. Under the clinical-efficacy (CE) F1 metric over 18 chest-CT pathologies extracted by a pre-trained RadBERT clinical extractor, this paradigm shows high precision but low recall. Many pathologies are simply not mentioned in the single report.

MDEF turns the single-report paradigm into a multi-report paradigm. The pipeline has five stages, described in Section IV of the paper.

1. Stage 1 (Ensemble of report generators). Three fine-tuned report generators that share a training recipe. They differ only in pre-trained initialisation and random seed.
2. Stage 2 (Stochastic Decoding per generator). Each generator samples `K=10` candidate reports at temperatures linearly staggered over `[0.7, 1.3]`. The K candidates are concatenated into one extended report `R_c(V)` per generator.
3. Stage 3 (RadBERT label extraction). The pre-trained RadBERT clinical extractor maps each generator's extended report to a binary 18-pathology vector.
4. Stage 4 (Per-pathology logical-OR late fusion). The three binary vectors are fused at the clinical-pathology level. Any pathology mentioned by any generator survives the fusion.
5. Stage 5 (Text head). A deterministic text head generates the final report from the fused 18-pathology vector and the per-generator extended reports.

Every inference-time hyperparameter is fixed once on RadGenome. The same values are reused unchanged on CT-RATE and INSPECT.

Trained only on RadGenome and evaluated on RadGenome, CT-RATE, and INSPECT, MDEF reaches CE F1 **0.3999** on RadGenome (in-distribution), **0.3381** on CT-RATE (cross-dataset), and **0.1813** on INSPECT (cross-institution). MDEF outperforms MARCH, the best single-report baseline, by `+0.079` CE F1 on RadGenome and `+0.045` on CT-RATE, with a small positive margin (`+0.008`) on INSPECT. The CE F1 gain is driven by recall. Averaged over the 18 pathologies, recall rises by `+70%` to `+97%` relative across the three datasets.

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
* 400 GB of disk space for the three datasets plus WDS shards and per-generator predictions

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
| RadBERT-RoBERTa-4m | 18-pathology clinical extractor weights | [HuggingFace model](https://huggingface.co/zzxslp/RadBERT-RoBERTa-4m). The classifier head is supplied with the public Reg2RG release. |

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

### 3.2 Train the three report generators

Every generator follows the same training recipe in `configs/mdef_train.yaml`. Only the initialisation and seed change between generators. The paper denotes them as `G_1`, `G_2`, and `G_3`.

```bash
bash scripts/train/01_train_generator_1.sh    # G_1: Reg2RG-init, seed=1
bash scripts/train/02_train_generator_2.sh    # G_2: Reg2RG-init, seed=2
bash scripts/train/03_train_generator_3.sh    # G_3: pathology-warmstart, seed=1
```

Checkpoints land in `checkpoints/generator_1/`, `checkpoints/generator_2/`, and `checkpoints/generator_3/`.

### 3.3 Inference and pathology-level fusion

```bash
bash scripts/inference/01_sd_union.sh
bash scripts/inference/02_or_fusion.sh
```

The first script runs Stage 2 and Stage 3. The second script runs Stage 4 (logical-OR over the three 18-pathology binary vectors).

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

The first script computes micro CE F1 and the per-pathology table. The second script computes BLEU 1 through 4, METEOR, and ROUGE-L. The third script computes paired bootstrap 95% confidence intervals. The fourth script regenerates the LaTeX table body used in the paper.

## 4. Training

A single command-line invocation also works directly.

```bash
uv run python -m src.training.trainer \
    --config configs/mdef_train.yaml \
    --seed 1 --init_from reg2rg \
    --output_dir checkpoints/generator_1
```

The trainer logs to `logs/{generator_name}/`. It saves a checkpoint per epoch (`save_every: 1`). Total wall time is roughly 24 hours per generator on a single H100 GPU at `batch_size=8`.

## 5. Inference and Fusion

Run Stochastic Decoding (Stage 2) and RadBERT extraction (Stage 3) for one generator on one tier as follows.

```bash
uv run python -m src.evaluation.evaluator \
    --design sd_union \
    --model_path checkpoints/generator_1/final_model.pt \
    --infer_config configs/mdef_infer.yaml \
    --output_dir predictions/generator_1/rg_K10_UNION \
    --shard_dir /data/mdef/processed/radgenome_wds \
    --eval_batch_size 8
```

After all three generators have emitted their per-tier predictions, run the OR fusion (Stage 4) as a single Python command.

```bash
uv run python -m src.fusion.logical_or \
    --infer_config configs/mdef_infer.yaml \
    --predictions_root predictions \
    --output_root predictions/ensemble_3way_or
```

## 6. Text Head and NLG Scoring

The Stage 5 text head is deterministic. It takes the fused 18-pathology vector and the three extended reports `{R_c(V)}` as input. The text head is invoked separately from the headline CE F1 pipeline. The CE F1 metric is invariant to any choice of head.

```bash
uv run python -m src.text_head.retrieve_report \
    --infer_config configs/mdef_infer.yaml \
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

* CE family (Table II). Precision, Recall, F1 micro-averaged over 18 RadBERT pathologies. Per-pathology P, R, F1 also appear in Table III.
* NLG family (Table II). BLEU 1 through 4, METEOR, ROUGE-L.
* Bootstrap CIs (Table IV). Paired-volume bootstrap 95% confidence intervals for the CE F1 and NLG gaps.
* Per-pathology table generator (Table III LaTeX body). Outputs a `tabular` snippet ready to drop into the paper.

## 8. Repository Structure

```plaintext
mdef/
├── configs/
│   ├── mdef_train.yaml         Stage 1 fine-tuning recipe (shared across generators)
│   ├── mdef_infer.yaml         Stage 2 and Stage 5 tier-invariant constants
│   └── storage.yaml            paths
├── scripts/
│   ├── base.sh                 sourced by every script. Reads storage.yaml.
│   ├── data/                   dataset download and WDS preprocessing
│   ├── train/                  per-generator training launchers
│   ├── inference/              Stage 2 SD-union and Stage 4 OR fusion
│   ├── text_head/              Stage 5 retrieval and boilerplate audit
│   └── evaluation/             CE, NLG, bootstrap CIs, per-pathology table
├── checkpoints/
│   ├── generator_1/            G_1 (Reg2RG-init, seed=1)
│   ├── generator_2/            G_2 (Reg2RG-init, seed=2)
│   └── generator_3/            G_3 (pathology-warmstart, seed=1)
└── src/
    ├── data/                   WebDataset readers, train and eval splits
    ├── models/                 3D-ViT vision encoder and report-generator backbone
    ├── training/               training loop (Stage 1)
    ├── inference/              Stage 2 SD-union per generator
    ├── fusion/                 Stage 3 plus Stage 4. RadBERT extractor and OR fusion.
    ├── text_head/              Stage 5 deterministic retrieved-sentence head
    ├── evaluation/             CE F1, NLG, bootstrap, per-pathology table
    └── utils/                  shared utilities (logger, etc.)
```

## Citation

```bibtex
@inproceedings{ta2026mdef,
  title     = {MDEF: Multi-Report Deep Ensemble Fusion for 3D CT Report Generation},
  author    = {Ta, Tung Duong and Oates, Tim and Gaur, Manas and Nguyen, Trong-Nghia and Nguyen, Tien-Cuong and Vuong, Tuan-Cuong and Mai, Trang Xuan and Luong, Thien Van},
  booktitle = {IEEE International Conference on Data Mining Workshops (ICDMW), Workshop on Large Language Models for Multimodal Data Fusion (LLM4MDF)},
  year      = {2026}
}
```

## License

This code is released under the [Apache License 2.0](LICENSE). Some model files are adapted from third-party projects under their own licenses. See [NOTICE](NOTICE) for details.

The datasets are not redistributed here. RadGenome-ChestCT, CT-RATE, and INSPECT remain under their own licenses and data use agreements.
