import torch
from torch.utils.data import TensorDataset


class CustomLyricTensorDataset(TensorDataset):
    def __init__(self, *args, labels_names=None, **kwargs):
        super(CustomLyricTensorDataset, self).__init__(*args, **kwargs)
        self.labels = labels_names
    
    def __getitem__(self, index):
        data_point = super(CustomLyricTensorDataset, self).__getitem__(index)
        label_id = self.labels[index]
        return label_id, data_point
    