"""WebDataset-style reader for pre-tensorized RadGenome 24k tar shards.

Drop-in replacement for `RadGenomeDataset_Train` whose shards are produced by
`analysis/09_novel_architectures/scripts/convert_radgenome_to_wds.py`.

Shard layout:
    <shard_dir>/shard_0000.tar
    <shard_dir>/shard_0001.tar
    ...
    <shard_dir>/manifest.json

Each tar entry is `<global_idx_zero_padded_6>.pt`, a torch.save dict with:
    mask_img_tensors  : dict[str -> fp16 (256, 256, 64) tensor]   (single channel)
    mask_tensors      : dict[str -> fp16 (256, 256, 64) tensor]
    region_reports    : dict[str -> str]
    image_path        : str (original .nii.gz path, for debugging)
    sample_idx        : int (global)

This class:
  1. Builds a tarfile member-offset index once on first init (cached to
     `<shard_dir>/_index_cache.pkl`).
  2. On `__getitem__(idx)`: opens the appropriate shard, reads the tar
     member's bytes, deserializes via torch.load, then does the cheap
     post-processing (fp16 -> fp32 cast, .repeat(3,...) channel expand,
     batch dim, tokenization) to produce the same dict as
     RadGenomeDataset_Train.__getitem__.

Speedup vs original on contended NFS: 5-10x. The expensive work
(.nii.gz decode, MONAI CropForeground/Resize, HU clamp/normalize) is
pre-done once at conversion time.
"""

import glob
import io
import os
import pickle
import random
import tarfile

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

CONDITIONS = [
    'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
    'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
    'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
    'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
    'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
    'Interlobular septal thickening',
]

REGIONS = [
    'abdomen', 'bone', 'breast', 'esophagus', 'heart', 'lung',
    'mediastinum', 'pleura', 'thyroid', 'trachea and bronchie',
]


class RadGenomeWDSDataset(Dataset):
    """Reads pre-tensorized RadGenome samples from tar shards.

    Output dict matches RadGenomeDataset_Train.__getitem__ exactly:
        lang_x, vision_x, mask_x, region2area, attention_mask, label
    """

    def __init__(self, text_tokenizer, shard_dir,
                 max_region_size=10, max_img_size=1, image_num=32,
                 region_num=33, max_seq=2048, voc_size=32000,
                 force_num_frames=True):
        # Tokenizer setup — must match RadGenomeDataset_Train so the LM's
        # special-token vocabulary lines up.
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        special_token = {"additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}

        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(image_token)
            self.image_padding_tokens.append(image_padding_token)

        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region" + str(i * region_num + j) + ">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(region_token)
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(special_token)
        self.text_tokenizer.pad_token_id = 0
        self.text_tokenizer.bos_token_id = 1
        self.text_tokenizer.eos_token_id = 2

        self.voc_size = voc_size
        self.max_seq = max_seq
        self.shard_dir = shard_dir

        self._build_index()
        print(f"RadGenomeWDSDataset: {len(self.index)} samples across "
              f"{len(set(p for p, _ in self.index))} shards in {shard_dir}")

    def _build_index(self):
        """Scan shards once, cache (shard_path, member_name) per global idx.

        Cache is invalidated automatically when the on-disk shard count
        differs from the cached count (e.g., new shards landed since the
        cache was written). Without this guard a partially-converted
        shard dir's index gets baked in and subsequent runs see a stale
        sample count even though more shards are now available.
        """
        cache_path = os.path.join(self.shard_dir, "_index_cache.pkl")
        n_shards_on_disk = len(glob.glob(os.path.join(self.shard_dir, "shard_*.tar")))
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            n_shards_cached = len(set(p for p, _ in cached))
            if n_shards_cached == n_shards_on_disk:
                self.index = cached
                return
            print(f"RadGenomeWDSDataset: cache stale "
                  f"({n_shards_cached} cached vs {n_shards_on_disk} on disk); rebuilding")

        shard_paths = sorted(glob.glob(os.path.join(self.shard_dir, "shard_*.tar")))
        if not shard_paths:
            raise RuntimeError(f"No shard_*.tar found under {self.shard_dir}")

        # Members are emitted in shard order; member name encodes global idx.
        index = []
        for shard_path in shard_paths:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(".pt"):
                        index.append((shard_path, member.name))
        # Sort by global idx (encoded in member filename) so __getitem__(idx)
        # is deterministic regardless of tar member ordering.
        index.sort(key=lambda x: int(os.path.splitext(x[1])[0]))
        self.index = index

        with open(cache_path, "wb") as f:
            pickle.dump(self.index, f)

    def __len__(self):
        return len(self.index)

    def _read_sample(self, idx):
        """Read raw saved dict from shard."""
        shard_path, member_name = self.index[idx]
        with tarfile.open(shard_path, "r") as tar:
            buf = tar.extractfile(member_name).read()
        return torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)

    def text_add_image_tokens(self, text):
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text
        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + \
                '<region>' + self.region_padding_tokens[i] + '</region>. '
        text = region_text + text
        return text

    def __getitem__(self, index):
        data = self._read_sample(index)
        return self._postprocess(data)

    def _postprocess(self, data):
        """Convert a raw saved sample dict into the trainer-ready dict.

        Public so the streaming sibling RadGenomeWDSStream can reuse this
        logic without duplicating ~80 lines of tensor reconstruction +
        tokenization.

        Reconstruct full tensor shape: (256,256,64) fp16 -> (1,3,256,256,64) fp32.
        Original RadGenomeDataset_Train did: tensor.repeat(3,1,1,1).unsqueeze(0)
        after a (1, D, H, W) intermediate. We saved tensor[0, 0] (a single
        channel slice of the 3-channel tensor — channels 0,1,2 are identical
        by construction, so no information is lost).
        """
        # Domain-generalization augmentations (training-only, env-gated).
        # Each simulates a scanner/protocol variation that the model may
        # face at cross-tier eval time. None need target-domain data —
        # single-source DG via input-space perturbation. Added 2026-05-15.
        import os as _os
        aug_intensity = _os.environ.get("AUG_INTENSITY", "0") == "1"
        aug_window    = _os.environ.get("AUG_WINDOW", "0") == "1"
        aug_noise     = _os.environ.get("AUG_NOISE", "0") == "1"
        aug_flip      = _os.environ.get("AUG_FLIP", "0") == "1"

        mask_img_tensors = {}
        for key, t in data["mask_img_tensors"].items():
            t = t.float()                   # (256, 256, 64) fp32
            if aug_intensity:
                # Random mult [0.85, 1.15] + shift [-0.1, 0.1] — simulates
                # different scanner contrast / dose levels.
                scale = random.uniform(0.85, 1.15)
                shift = random.uniform(-0.1, 0.1)
                t = t * scale + shift
            if aug_window:
                # Soft HU re-windowing: re-center + re-scale to simulate
                # different window/level settings.
                center_shift = random.uniform(-0.15, 0.15)
                width_scale  = random.uniform(0.8, 1.25)
                t = ((t - center_shift) / width_scale).clamp(-1.5, 1.5)
            if aug_noise:
                # Gaussian noise (low-dose / recon noise simulation).
                noise_std = random.uniform(0.005, 0.02)
                t = t + torch.randn_like(t) * noise_std
            if aug_flip and random.random() < 0.5:
                # Left-right flip (axis 0). NOT applied to mask_tensors —
                # we'd corrupt anatomy labels. Only flip the image.
                t = torch.flip(t, dims=[0])
            t = t.unsqueeze(0)              # (1, 256, 256, 64) — channel dim
            t = t.repeat(3, 1, 1, 1)        # (3, 256, 256, 64)
            t = t.unsqueeze(0)              # (1, 3, 256, 256, 64) — batch dim
            mask_img_tensors[key] = t

        mask_tensors = {}
        for key, t in data["mask_tensors"].items():
            t = t.float()
            t = t.unsqueeze(0)              # (1, 256, 256, 64)
            mask_tensors[key] = t

        region_reports = dict(data["region_reports"])
        # Filter to regions that survived the mask check at conversion time.
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                region_reports.pop(key)

        # Random shuffle of region order — same as original dataset.
        region2area = {}
        shuffled_areas = list(region_reports.keys())
        random.shuffle(shuffled_areas)
        for i, area in enumerate(shuffled_areas):
            region2area[i] = area

        instruction = (
            "Given the provided global and regional information from this CT scan, "
            "please generate a comprehensive medical report for each region. First, "
            "identify the anatomical area corresponding to each region, then provide "
            "detailed information about these anatomical structures and any "
            "abnormalities that are essential. You can refer to the global information "
            "as the context and take it as a supplement when generating each region "
            "report."
        )

        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))
        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report += "The region " + str(i) + " is " + area + ": " + region_report + " "

        self.text_tokenizer.padding_side = "right"

        text_tensor = self.text_tokenizer(
            prompt + ' ' + combined_report,
            max_length=self.max_seq, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        text_input = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]
        text_input[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id

        prompt_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        prompt_length = torch.sum(prompt_tensor["attention_mask"][0])

        label = text_input.clone()
        label[label == self.text_tokenizer.pad_token_id] = -100
        label[label >= self.voc_size] = -100
        label[:prompt_length] = -100

        return {
            'lang_x': text_input,
            'vision_x': mask_img_tensors,
            'mask_x': mask_tensors,
            'region2area': region2area,
            'attention_mask': attention_mask,
            'label': label,
        }


class RadGenomeWDSDataset_Test(RadGenomeWDSDataset):
    """Eval-time WDS reader. Returns the dict shape RadGenomeDataset_Test.__getitem__
    produces (with acc_num + gt_combined_report) so generate_and_eval.py can swap
    NFS-path eval for shard-based eval transparently via a `--shard_dir` flag.

    Difference vs RadGenomeWDSDataset (train-time):
      - prompt does NOT include the GT report (test-time tokenization)
      - returned dict has acc_num + gt_combined_report (for predictions.csv)
      - no shuffle of region order (deterministic eval)
      - supports inferenced_id=[done acc_nums] to skip already-processed samples

    Sharing _read_sample(), _build_index(), and __init__ with the parent.
    Added 2026-05-15 to make WDS-shard eval pipeline equivalent to the
    NFS-path RadGenomeDataset_Test.
    """

    def __init__(self, text_tokenizer, shard_dir, inferenced_id=None, **kw):
        super().__init__(text_tokenizer, shard_dir, **kw)
        # Build acc_num index to support inferenced_id-style skip.
        self._acc_for_idx = {}      # global idx -> acc_num
        if inferenced_id:
            done = set(str(x) for x in inferenced_id)
            keep = []
            for i, (shard, member) in enumerate(self.index):
                acc = self._acc_for(i, shard, member)
                if acc not in done:
                    keep.append((shard, member))
            print(f"RadGenomeWDSDataset_Test: skipping {len(self.index) - len(keep)} "
                  f"already-done samples via inferenced_id")
            self.index = keep

    def _acc_for(self, idx, shard_path, member_name):
        """Read just enough of one sample to extract its acc_num.

        Cheap: just reads the pickled dict's image_path and parses the
        accession number from the basename.
        """
        if idx in self._acc_for_idx:
            return self._acc_for_idx[idx]
        data = self._read_sample(idx)
        img = data.get("image_path", "")
        acc = os.path.basename(os.path.dirname(img)) if img else str(idx)
        self._acc_for_idx[idx] = acc
        return acc

    def __getitem__(self, index):
        data = self._read_sample(index)
        # Reconstruct tensors with the same logic as parent's _postprocess,
        # but skip the training-time label/attention_mask tokenization.
        mask_img_tensors = {}
        for key, t in data["mask_img_tensors"].items():
            t = t.float().unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)
            mask_img_tensors[key] = t

        mask_tensors = {}
        for key, t in data["mask_tensors"].items():
            mask_tensors[key] = t.float().unsqueeze(0)

        region_reports = dict(data["region_reports"])
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                region_reports.pop(key)

        # Deterministic region order at eval time (no shuffle).
        region2area = {i: a for i, a in enumerate(region_reports.keys())}

        instruction = (
            "Given the provided global and regional information from this CT scan, "
            "please generate a comprehensive medical report for each region. First, "
            "identify the anatomical area corresponding to each region, then provide "
            "detailed information about these anatomical structures and any "
            "abnormalities that are essential. You can refer to the global information "
            "as the context and take it as a supplement when generating each region "
            "report."
        )
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))
        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            combined_report += "The region " + str(i) + " is " + area + ": " + region_reports[area] + " "

        text_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]

        img = data.get("image_path", "")
        acc_num = os.path.basename(os.path.dirname(img)) if img else str(index)

        return {
            'acc_num': acc_num,
            'lang_x': text_input,
            'vision_x': mask_img_tensors,
            'mask_x': mask_tensors,
            'region2area': region2area,
            'question': prompt,
            'gt_combined_report': combined_report,
        }
