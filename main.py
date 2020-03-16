from utils import *
from argument_parser import argument_parser
import pandas as pd
from torch.utils.data import DataLoader
import torch
from models import GNE
import networkx as nx

args = argument_parser()
table_printer(args)

# Define path
path = './data/' + args.dataset + '/'

geneids = pd.read_csv(path + args.gene_ids_file, sep=" ")
num_genes = geneids.shape[0]
print(num_genes)

# Define the input to GNE model
link_file = path + args.edgelist_file
feature_file = path + args.feature_file

A, nodes = load_network(link_file, num_genes)
N = len(list(nodes))
features = load_data(feature_file, nodes, args.normalize)
train_ones, train_zeros, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                                                                                     every_node=True, connected=False, undirected=True,
                                                                                                     use_edge_cover=True, set_ops=True, asserts=False)

train_ones = torch.Tensor(train_ones).type(torch.LongTensor)
features = torch.Tensor(features).type(torch.FloatTensor)

trainData = GNEDataset(features)
train_loader = DataLoader(trainData, batch_size=32, shuffle=True)
model = GNE(A.shape[0], features.shape[1], args)
print(model)
for i, (input, feat) in enumerate(train_loader):
    emb = model(input, feat)
    print(emb.shape)
    break
