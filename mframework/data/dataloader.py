from typing import Any, Iterator

from mframework.data.dataset import Dataset
from mframework.data.sampling import Sampler, BatchSampler

"""
DataLoader.

The point of a dataloader is to provide a wrapper to easily iterate over and work with dataset.
They bundle together datasets and samplers.
The sampler defines how dataset indices are grouped into batches.
"""

class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        sampler: Sampler,
        drop_last: bool = False
    ):
        self._dataset = dataset
        self._drop_last = drop_last
        self._sampler = sampler

    def _collate_to_batch(self, batch: list[Any]):
        """
        Convert a list of dataset items into a batch.
        Assumes each item is a tuple (X, y).
        Returns tuple of lists: ([X_1, X_2, ...], [y_1, y_2, ...])
        """
        # Handles the case of zero length datasets gracefully
        if not batch:
            return batch

        if isinstance(batch[0], tuple):
            return tuple([list(x) for x in zip(*batch)])
        
        return batch

    def __iter__(self):
        for indices in self._sampler:
            if isinstance(indices, int):
                indices = [indices]
            batch = [self._dataset[i] for i in indices]
            batch = self._collate_to_batch(batch)
            yield batch

    def __len__(self):
        return len(self._sampler)
