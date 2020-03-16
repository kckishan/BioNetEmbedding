from torch.utils.data import Dataset
import networkx as nx
from numpy.random import choice


class GNEDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, indx):
        return indx, self.features[indx]
