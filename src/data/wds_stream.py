"""Streaming WebDataset-style reader for pre-tensorized RadGenome tar shards.

Companion to `radgenome_wds_dataset.py`'s `RadGenomeWDSDataset` (random-access).
This reader is an `IterableDataset` that:

  1. Iterates shards in a per-epoch shuffled order.
  2. Within each shard, reads all 200 .pt members SEQUENTIALLY (one tar.open
     per shard rather than one per sample).
  3. Maintains a small in-memory buffer of recently-read samples and
     yields random samples from it (cheap local shuffling).

Why this matters: the random-access reader does `tarfile.open()` on a
25 GB shard every `__getitem__` call. Since HF Trainer shuffles indices
across all shards, each batch hits a different shard, and NFS gets
hammered with random seeks across 121 × 25 GB tar files. On a contended
NFS share this drops the effective step rate by 3-5×.

The streaming reader instead opens each shard once, streams its 200
members, then moves to the next shard — sequential reads, NFS-friendly.
The `shuffle_buffer` provides enough mixing that gradient noise is
preserved at the optimization level (well-established WebDataset pattern).

HF Trainer integration:
  - This is an `IterableDataset` with `__len__` defined (length comes
    from the index cache written by `RadGenomeWDSDataset`, or a per-shard
    count). HF Trainer uses `__len__` for total_steps + tqdm.
  - Multi-worker DataLoader: shards are partitioned by worker_id so
    each worker reads a disjoint subset (no double reads).
  - Checkpoint resume: replay across multi-shard shuffles is non-trivial;
    by default we don't try (the trainer's
    `ignore_data_skip=True` should be set when using this reader).
"""

import glob
import io
import os
import pickle
import random
import tarfile
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .radgenome_wds_dataset import RadGenomeWDSDataset


class RadGenomeWDSStream(IterableDataset):
    """Sequential per-shard streaming reader; same output dict as random-access."""

    def __init__(self, text_tokenizer, shard_dir,
                 max_region_size=10, max_img_size=1, image_num=32,
                 region_num=33, max_seq=2048, voc_size=32000,
                 shuffle_buffer=100, seed=42, force_num_frames=True):
        super().__init__()
        # Reuse the random-access reader's tokenizer + post-processing
        # (we only override the data-reading half). __init__ does the
        # tokenizer setup + indexing scan; we benefit from its cached
        # index for a length estimate.
        self._inner = RadGenomeWDSDataset(
            text_tokenizer=text_tokenizer, shard_dir=shard_dir,
            max_region_size=max_region_size, max_img_size=max_img_size,
            image_num=image_num, region_num=region_num,
            max_seq=max_seq, voc_size=voc_size,
            force_num_frames=force_num_frames,
        )
        self.shard_dir = shard_dir
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.tar")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards found in {shard_dir}")

        # Length: total samples across shards (from the cache the
        # random-access reader builds). Used by HF Trainer to compute
        # total_steps and the tqdm bar.
        self._n_samples = len(self._inner.index)
        self._epoch = 0
        print(f"RadGenomeWDSStream: {len(self.shard_paths)} shards, "
              f"{self._n_samples} total samples, "
              f"shuffle_buffer={self.shuffle_buffer}")

    def __len__(self):
        return self._n_samples

    def set_epoch(self, epoch: int):
        """HF Trainer calls this before each epoch — varies the shard
        shuffling order so different epochs see different orderings."""
        self._epoch = epoch

    def _shard_iter_raw(self, shard_path: str):
        """Yield RAW saved data dicts (~100 MB each) from one shard.

        Postprocessing is deferred to yield time so the shuffle buffer
        holds the smaller raw form rather than the 4x bigger postprocessed
        form (which expands fp16 single-channel to fp32 3-channel).
        """
        try:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if not (member.isfile() and member.name.endswith(".pt")):
                        continue
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    buf = fobj.read()
                    try:
                        data = torch.load(io.BytesIO(buf), map_location="cpu",
                                          weights_only=False)
                    except Exception as e:
                        print(f"  [stream] failed to load {member.name} "
                              f"from {os.path.basename(shard_path)}: {e}")
                        continue
                    yield data
        except (tarfile.ReadError, OSError) as e:
            print(f"  [stream] failed to open shard {shard_path}: {e}")

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        if worker_info is None:
            shard_paths = list(self.shard_paths)
            seed = self.seed + self._epoch
        else:
            # Each worker gets a disjoint subset of shards (round-robin
            # partition by worker_id). Avoids two workers re-reading the
            # same shard concurrently.
            shard_paths = self.shard_paths[worker_info.id::worker_info.num_workers]
            seed = self.seed + self._epoch + worker_info.id

        # Per-epoch shard order
        rng = random.Random(seed)
        rng.shuffle(shard_paths)

        # Buffered shuffle: yield random samples from a rolling buffer of
        # RAW saved dicts (~100 MB each). Postprocess only at yield time
        # so the buffer doesn't hold 4x bigger fp32 3-channel tensors.
        buffer: list = []
        for shard_path in shard_paths:
            for raw in self._shard_iter_raw(shard_path):
                buffer.append(raw)
                if len(buffer) >= self.shuffle_buffer:
                    idx = rng.randrange(len(buffer))
                    buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                    raw_pick = buffer.pop()
                    yield self._inner._postprocess(raw_pick)
        # Drain remaining samples in randomized order
        rng.shuffle(buffer)
        for raw in buffer:
            yield self._inner._postprocess(raw)
