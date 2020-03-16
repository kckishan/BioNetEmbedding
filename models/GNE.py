import torch
from torch import nn as nn
from torch.nn import functional as F
from .Sampled_softmax import SampledSoftmax


class GNE(nn.Module):
    def __init__(self, net_feature_size, attr_feature_size, args):
        super(GNE, self).__init__()
        self.args = args
        self.net_emb_dim = self.args.net_emb_dim
        self.attr_emb_dim = self.args.attr_emb_dim
        self.net_feature_size = net_feature_size
        self.attr_feature_size = attr_feature_size
        self.abstract_feature_size = self.net_emb_dim + self.attr_emb_dim
        self.setup_network_structure()

    def setup_network_structure(self):
        self.net_embedding = nn.Embedding(self.net_feature_size, self.net_emb_dim)
        self.attr_embedding = nn.Linear(self.attr_feature_size, self.attr_emb_dim)
        self.hidden_layer = nn.Linear(self.abstract_feature_size, self.args.latent_size)
        self.init_weight(self.net_embedding)
        self.init_weight(self.attr_embedding)
        self.dropout = nn.Dropout(self.args.dropout)
        self.attr_act = nn.ELU()
        self.latent_act = nn.Tanh()
        self.bn_net = nn.BatchNorm1d(self.net_emb_dim)
        self.bn_attr = nn.BatchNorm1d(self.net_emb_dim)
        self.sampled_softmax = SampledSoftmax(
            self.net_feature_size, nsampled=10, nhid=self.args.latent_size, tied_weight=None)

    def forward(self, source, features):
        self.net_emb = self.bn_net(self.net_embedding(source))
        self.attr_emb = self.bn_attr(self.attr_act(self.attr_embedding(features)))
        embed = self.dropout(self.merge_features())
        z = self.hidden_layer(embed)
        return z

    def init_weight(self, layer):
        nn.init.xavier_uniform_(layer.weight)

    def merge_features(self, mode="concat"):
        return torch.cat((self.net_emb, self.attr_emb), 1)
