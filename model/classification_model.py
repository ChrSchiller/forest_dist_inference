import torch
import torch.nn as nn
from .bert import SBERT

class SBERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes, seq_len):
        super().__init__()
        self.sbert = sbert
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes, seq_len)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.classification(x, mask)

class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes, seq_len):
        super().__init__()
        ### note that nn.MaxPool1d(64) as value for MaxPool1d only works if max_length == 64 
        ### (meaning that it was hard-coded in the original SITS-BERT code),
        ### otherwise the code throws an error
        ### (also then the code does not meet the description in the paper)
        ### a better way to do it is to use nn.MaxPool1d(max_length)
        ### also because then the 'squeeze' method makes more sense (the '1' dimension will be dropped)
        self.max_len = seq_len
        self.relu = nn.ReLU()
        # self.pooling = nn.MaxPool1d(64)
        self.pooling = nn.MaxPool1d(self.max_len)

        self.linear = nn.Linear(hidden, num_classes)


    def forward(self, x, mask):
        x = self.pooling(x.permute(0, 2, 1)).squeeze()

        x = self.linear(x)
        return x