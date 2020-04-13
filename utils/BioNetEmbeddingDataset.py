from torch.utils.data import Dataset

class BioNetEmbeddingDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, indx):
        edge = self.interactions[indx]
        return edge[0], edge[1]