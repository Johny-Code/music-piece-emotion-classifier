import torch
from torch.utils.data import TensorDataset


class CustomLyricSortedTensorDataset(TensorDataset):
    def __init__(self, *args, labels_names=None, **kwargs):
        super(CustomLyricSortedTensorDataset, self).__init__(*args, **kwargs)
        self.labels_names = labels_names

        # Sort the data by label names
        self.sorted_indices = sorted(range(len(self.labels_names)), key=lambda i: self.labels_names[i])
    
    def __getitem__(self, index):
        original_index = self.sorted_indices[index]  # Get the original index
        data_point = super(CustomLyricSortedTensorDataset, self).__getitem__(original_index)
        label = self.labels_names[original_index]
        return data_point, label
    