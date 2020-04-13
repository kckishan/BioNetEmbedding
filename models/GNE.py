import torch
from torch import nn as nn
from torch.nn import functional as F
from .Sampled_softmax import SampledSoftmax
import os 
import pandas as pd
import numpy as np

def safediv(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b)
    return a / b

class GNE(nn.Module):
    def __init__(self, num_nodes, args, device):
        super(GNE, self).__init__()
        self.args = args
        self.device = device
        self.net_emb_dim = self.args.net_emb_dim
        self.num_nodes = num_nodes
        self.abstract_feature_size = self.net_emb_dim
        self.n_sampled = 10

        self.setup_network_structure()
        self.criterion = nn.CrossEntropyLoss()
        

    def setup_network_structure(self):
        self.net_embedding = nn.Embedding(self.num_nodes, self.net_emb_dim)
        self.hidden_layer = nn.Linear(self.abstract_feature_size, self.args.latent_size)
        self.init_weight(self.net_embedding)
        self.dropout = nn.Dropout(self.args.dropout)
        self.bn_net = nn.BatchNorm1d(self.net_emb_dim)
        self.sampled_softmax = SampledSoftmax(
            self.num_nodes, nsampled=self.n_sampled, nhid=self.args.latent_size, tied_weight=None,device=self.device)

    def forward(self, source, targets):
        emb = self.net_embedding(source)
        net_emb = self.dropout(self.bn_net(emb))
        
        z = self.hidden_layer(net_emb)

        z = safediv(z, z.norm(dim=1, keepdim=True))

        logits, new_targets = self.sampled_softmax(z, targets)
        if self.training:    
            logits = logits.view(-1, self.n_sampled+1)  
            loss = self.criterion(logits, new_targets)
        else:
            logits = logits.view(-1, self.num_nodes)  
            loss = self.criterion(logits, targets)

        return z, loss

    def init_weight(self, layer):
        nn.init.xavier_uniform_(layer.weight)

    def get_embeddings(self):
        in_embeddings = self.net_embedding.weight.data
        out_embeddings = self.sampled_softmax.params.weight.data
        embeddings = (in_embeddings + out_embeddings).cpu().detach().numpy()
        
        return embeddings

    def embedding_checkpoints(self, Embeddings=None, mode="save"):
        file = "results/embeddings.txt"
        if mode == "save":
            if os.path.isfile(file):
                os.remove(file)
            pd.DataFrame(Embeddings).to_csv(file, index=False, header=False)
        if mode == 'load':
            Embeddings = pd.read_csv(file, header=None)
            return np.array(Embeddings)
