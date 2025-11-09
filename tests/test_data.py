import pytest
from mframework.data.dataset import ArrayDataset, Dataset
from mframework.data.sampling import SequentialSampler, BatchSampler
from mframework.data.dataloader import DataLoader


# ------------------------
# Fixtures
# ------------------------

@pytest.fixture
def simple_dataset():
    X = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    return ArrayDataset(X, y)


# ------------------------
# Dataset tests
# ------------------------

def test_arraydataset_len(simple_dataset):
    assert len(simple_dataset) == 5


def test_arraydataset_getitem(simple_dataset):
    x, y = simple_dataset[2]
    assert x == 3
    assert y == 30


def test_arraydataset_iterates_correctly(simple_dataset):
    data = [simple_dataset[i] for i in range(len(simple_dataset))]
    assert data == [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]


# ------------------------
# Sampler tests
# ------------------------

def test_sequentialsampler(simple_dataset):
    sampler = SequentialSampler(simple_dataset)
    indices = list(iter(sampler))
    assert indices == [0, 1, 2, 3, 4]
    assert len(sampler) == 5


def test_batchsampler_droplast(simple_dataset):
    sampler = SequentialSampler(simple_dataset)
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
    batches = list(iter(batch_sampler))
    # Expect [0,1], [2,3] — last index (4) dropped
    assert batches == [[0, 1], [2, 3]]
    assert len(batch_sampler) == 2


def test_batchsampler_keep_last(simple_dataset):
    sampler = SequentialSampler(simple_dataset)
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=False)
    batches = list(iter(batch_sampler))
    # Expect [0,1], [2,3], [4]
    assert batches == [[0, 1], [2, 3], [4]]
    assert len(batch_sampler) == 3


def test_batchsampler_empty_dataset():
    class EmptyDataset(Dataset):
        def __len__(self): return 0
    dataset = EmptyDataset()
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=False)
    assert list(batch_sampler) == []
    assert len(batch_sampler) == 0


# ------------------------
# DataLoader tests
# ------------------------

def test_dataloader_yields_batches(simple_dataset):
    sampler = BatchSampler(SequentialSampler(simple_dataset), batch_size=2, drop_last=False)
    dataloader = DataLoader(simple_dataset, sampler)

    batches = list(dataloader)

    # Expect 3 batches: [0,1], [2,3], [4]
    assert len(batches) == 3
    assert batches[0] == ([1, 2], [10, 20])
    assert batches[-1] == ([5], [50])
    assert len(dataloader) == 3  # length = sampler length


def test_dataloader_droplast(simple_dataset):
    sampler = BatchSampler(SequentialSampler(simple_dataset), batch_size=2, drop_last=True)
    dataloader = DataLoader(simple_dataset, sampler)

    batches = list(dataloader)
    # Expect two full batches only
    assert len(batches) == 2
    assert batches == [([1, 2], [10, 20]), ([3, 4], [30, 40])]
    assert len(dataloader) == 2


def test_dataloader_len_matches_sampler(simple_dataset):
    sampler = BatchSampler(SequentialSampler(simple_dataset), batch_size=3, drop_last=False)
    dataloader = DataLoader(simple_dataset, sampler)
    assert len(dataloader) == len(sampler)


def test_dataloader_handles_single_samples(simple_dataset):
    sampler = BatchSampler(SequentialSampler(simple_dataset), batch_size=1, drop_last=False)
    dataloader = DataLoader(simple_dataset, sampler)
    batches = list(dataloader)
    assert batches[0] == ([1], [10])
    assert len(batches) == 5


def test_dataloader_with_non_tuple_items():
    # Custom dataset returning scalars
    class ScalarDataset(Dataset):
        def __len__(self): return 3
        def __getitem__(self, idx): return idx

    sampler = BatchSampler(SequentialSampler(ScalarDataset()), batch_size=2, drop_last=False)
    dataloader = DataLoader(ScalarDataset(), sampler)

    batches = list(dataloader)
    # Since collate just returns raw list if not tuples
    assert batches == [[0, 1], [2]]
    assert len(dataloader) == 2
