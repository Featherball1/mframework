import itertools
from typing import Iterator

from mframework.data.dataset import Dataset

"""
Sampler.

A sampler provides a rule by which we iterate over a dataset (e.g. in sequence, at random, so on).
We also have a batch sampler, which wraps other samplers to yield batches of indices at a time.
"""

class Sampler:
    def __iter__(self) -> Iterator[int]: raise NotImplementedError
    
    def __len__(self) -> int: raise NotImplementedError

class SequentialSampler(Sampler):
    """
    The purpose of a sequential sampler is to provide a mechanism to iterate over a dataset in sequence. 
    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self) -> Iterator[int]:
        """
        Return the indices in sequence. 
        """
        for i in range(len(self._dataset)):
            yield i

    def __len__(self) -> int:
        return len(self._dataset)

    
class BatchSampler(Sampler):
    """
    The responsibility of a batch sampler is to wrap other samplers to yield a mini-batch of indices.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool
    ):
        assert batch_size > 0, "batch_size must be positive"
        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last
    
    def __iter__(self):
        batch = []
        for idx in self._sampler:
            batch.append(idx)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch and not self._drop_last:
            yield batch

    def __len__(self) -> int:
        total = len(self._sampler)
        if self._drop_last:
            return total // self._batch_size
        return (total + self._batch_size - 1) // self._batch_size
