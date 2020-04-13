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

class BioNetEmbedding(nn.Module):
    def __init__(self, num_nodes, args, device):
        super(BioNetEmbedding, self).__init__()
        self.args = args
        self.device = device
        self.emb_dim = self.args.emb_dim
        self.num_nodes = num_nodes
        self.n_sampled = 10

        self.setup_network_structure()
        self.criterion = nn.CrossEntropyLoss()

    def setup_network_structure(self):
        #Embedding layer
        self.net_embedding = nn.Embedding(self.num_nodes, self.emb_dim)
        self.init_weight(self.net_embedding)
        
        # Hidden layers
        self.hidden_layer = nn.Linear(self.emb_dim, self.args.latent_size)
        self.init_weight(self.hidden_layer)

        self.layers = nn.ModuleList([self.net_embedding, self.hidden_layer])

        self.sampled_softmax = SampledSoftmax(
            self.num_nodes, nsampled=self.n_sampled, nhid=self.args.latent_size, tied_weight=None,device=self.device)

    def forward(self, source, targets):
        latent = source
        for layer in self.layers:
            latent = layer(latent)

        # normalizing the embedding
        latent = safediv(latent, latent.norm(dim=1, keepdim=True))

        logits, new_targets = self.sampled_softmax(latent, targets)
        if self.training:    
            logits = logits.view(-1, self.n_sampled+1)  
            loss = self.criterion(logits, new_targets)
        else:
            logits = logits.view(-1, self.num_nodes)  
            loss = self.criterion(logits, targets)

        return latent, loss

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
