"""Bucket-aware dataset/sampler for the Stage-2 latent encoding loop.

The design is intentionally minimal compared to helios `BucketedSampler`:
each rank consumes a disjoint subset of episodes, and within a rank we walk
buckets in a deterministic shuffled order so all ranks step through the same
shape sequence (helps debugging / future batching). Encoding still runs at
`batch_size=1` per episode because individual episodes have heterogeneous
chunk counts.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .latent_io import read_jsonl


class LatentManifestDataset(Dataset):
    """Wrap a manifest JSONL produced by `scripts/latent/scan_dataset_meta.py`.

    Each item is the raw episode metadata dict; the bucket key tuple is
    materialised on `samples` for the sampler to bin without reparsing JSON.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        records = read_jsonl(manifest_path)
        if not records:
            raise ValueError(f"Empty manifest: {manifest_path}")

        self.samples: list[dict[str, Any]] = records
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for idx, rec in enumerate(records):
            bk = rec.get("bucket_key")
            if bk is None:
                raise ValueError(f"Missing `bucket_key` in manifest row {idx}: {rec}")
            self.buckets[tuple(bk)].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class LatentEpisodeBucketSampler(Sampler[int]):
    """Distribute episode indices across ranks while preserving bucket grouping.

    Behaviour:
    - All ranks see the same shuffled bucket order (driven by `seed`).
    - Inside each bucket, indices are also shuffled identically across ranks.
    - Each rank takes its slice via `indices[global_rank::world_size]` so the
      same episode is encoded exactly once (no duplicates across ranks).
    - `batch_size` is fixed to 1; we yield single-episode int indices.
    """

    def __init__(
        self,
        dataset: LatentManifestDataset,
        global_rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {world_size}")
        if not (0 <= global_rank < world_size):
            raise ValueError(f"global_rank {global_rank} out of [0, {world_size})")

        self.dataset = dataset
        self.global_rank = int(global_rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _all_indices_in_bucket_order(self) -> list[int]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)

        bucket_keys: Sequence[tuple[int, int, int]] = sorted(self.dataset.buckets.keys())
        if self.shuffle:
            order = torch.randperm(len(bucket_keys), generator=gen).tolist()
            bucket_keys = [bucket_keys[i] for i in order]

        all_indices: list[int] = []
        for bk in bucket_keys:
            indices = list(self.dataset.buckets[bk])
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=gen).tolist()
                indices = [indices[i] for i in perm]
            all_indices.extend(indices)
        return all_indices

    def __iter__(self) -> Iterator[int]:
        all_indices = self._all_indices_in_bucket_order()
        for idx in all_indices[self.global_rank :: self.world_size]:
            yield idx

    def __len__(self) -> int:
        total = len(self.dataset)
        per_rank = total // self.world_size
        if self.global_rank < total % self.world_size:
            per_rank += 1
        return per_rank


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m fastwam.datasets.lerobot.latents.latent_bucket_sampler <manifest.jsonl> [world_size]")
        sys.exit(1)

    manifest = sys.argv[1]
    ws = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    ds = LatentManifestDataset(manifest)
    print(f"manifest entries: {len(ds)} | unique buckets: {len(ds.buckets)}")

    counts: list[int] = []
    for r in range(ws):
        sampler = LatentEpisodeBucketSampler(ds, global_rank=r, world_size=ws)
        idxs = list(sampler)
        counts.append(len(idxs))
        print(f"rank {r}: {len(idxs)} indices, first 3 = {idxs[:3]}")
    assert sum(counts) == len(ds), (counts, len(ds))
    print("OK: total per-rank indices sum to dataset size.")
