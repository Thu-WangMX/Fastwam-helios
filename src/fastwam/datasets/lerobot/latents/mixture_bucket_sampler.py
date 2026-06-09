"""按 video_size 分 bucket 的分布式 sampler。

同 bucket 内的 sample 拥有相同的 latent 分辨率，可以组 batch。
不同 bucket 不混合 batch，避免 padding 浪费。

用法:
    sampler = MixtureBucketSampler(
        mixture_dataset,
        global_rank=rank, world_size=world_size,
        batch_size=16,
    )
    dataloader = DataLoader(mixture_dataset, batch_sampler=sampler)
"""
from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler

from .latent_mixture_dataset import LatentMixtureDataset


class MixtureBucketSampler(Sampler[list[int]]):
    """按 video_size 分 bucket 的 batch sampler。

    每个 batch 内所有 sample 来自同一个 bucket（同 video_size）。
    bucket 间的顺序每 epoch 随机打乱。
    分布式：每个 rank 取 bucket 内的不重叠子集。
    """

    def __init__(
        self,
        dataset: LatentMixtureDataset,
        batch_size: int = 16,
        global_rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        self._bucket_groups = dataset.get_all_bucket_groups()
        self._bucket_keys = sorted(self._bucket_groups.keys())

        total_batches = 0
        for bk in self._bucket_keys:
            n_indices = len(self._bucket_groups[bk])
            per_rank = n_indices // world_size
            n_batches = per_rank // batch_size
            if not drop_last and (per_rank % batch_size) > 0:
                n_batches += 1
            total_batches += n_batches
        self._total_batches = total_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)

        bucket_order = list(range(len(self._bucket_keys)))
        if self.shuffle:
            perm = torch.randperm(len(bucket_order), generator=gen).tolist()
            bucket_order = [bucket_order[i] for i in perm]

        for bi in bucket_order:
            bk = self._bucket_keys[bi]
            indices = list(self._bucket_groups[bk])

            if self.shuffle:
                perm = torch.randperm(len(indices), generator=gen).tolist()
                indices = [indices[i] for i in perm]

            # 分布式分片
            my_indices = indices[self.global_rank::self.world_size]

            # 组 batch
            for start in range(0, len(my_indices), self.batch_size):
                batch = my_indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        return self._total_batches


class MixtureSampler(Sampler[int]):
    """简单的加权分布式 sampler（不分 bucket，单条 yield）。

    用于不需要 bucket 分组的场景（所有数据集 video_size 相同）。
    """

    def __init__(
        self,
        dataset: LatentMixtureDataset,
        global_rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.global_rank = global_rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)

        total = len(self.dataset)
        if self.shuffle:
            indices = torch.randperm(total, generator=gen).tolist()
        else:
            indices = list(range(total))

        for idx in indices[self.global_rank::self.world_size]:
            yield idx

    def __len__(self) -> int:
        total = len(self.dataset)
        per_rank = total // self.world_size
        if self.global_rank < total % self.world_size:
            per_rank += 1
        return per_rank
