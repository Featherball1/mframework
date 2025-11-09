"""
Dataset.

A dataset is essentially our framework wrapper around an ordinary dictionary.
The point of representing a dataset by a class is to be able to specialise to different data dtypes more easily. 

Datasets support two basic operations
    1) Length of data
    2) Data access

At the minute we are only providing "map style" datasets. 
"""

class Dataset:
    def __next__(self): raise NotImplementedError
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "X and y must have the same length"

        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: Add TensorDataset
