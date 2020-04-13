from utils import *
from argument_parser import argument_parser
import pandas as pd
from torch.utils.data import DataLoader
import torch
from models import GNE
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from time import time


args = argument_parser()
table_printer(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define path
path = './data/' + args.dataset + '/'

geneids = pd.read_csv(path + args.gene_ids_file, sep=" ")
num_genes = geneids.shape[0]

# Define the input to GNE model
link_file = path + args.edgelist_file

A = load_network(link_file, num_genes)
N = A.shape[0]
data_splits = train_val_test_split_adjacency(A, p_val=0.1, p_test=0.05, seed=0, neg_mul=1,
                                             every_node=False, connected=False, undirected=True,
                                              use_edge_cover=False, set_ops=True, asserts=False)

train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = data_splits
# Inspect train/test split
print("Total genes:", N)
print("Training interactions (positive):", len(train_edges))
print("Training interactions (negative):", len(train_edges_false))
print("Validation interactions (positive):", len(val_edges))
print("Validation interactions (negative):", len(val_edges_false))
print("Test interactions (positive):", len(test_edges))
print("Test interactions (negative):", len(test_edges_false))

train_edges = torch.Tensor(train_edges).type(torch.LongTensor)
trainData = GNEDataset(train_edges)

validation_edges = np.concatenate([val_edges, val_edges_false])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

batch_size = 256
train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True)

model = GNE(N, args, device)
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay= args.l2)

val_loss_min = np.Inf
epochs = 100
start_time = time()

patience = 0
best_validation_accuracy = 0.0
early_stopping = 2

for epoch in range(epochs):
    train_loss = 0.0

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        emb, loss = model(data, labels)
        loss.backward()
        optimizer.step()

        # adding losses
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    # Get embeddings from trained model
    embeddings  = model.get_embeddings()
    
     # link prediction test
    ####################################################################################
    adj_matrix_rec = np.dot(embeddings, embeddings.T)
    roc, pr = evaluate_ROC_from_matrix(validation_edges, val_edge_labels, adj_matrix_rec)

    if roc > best_validation_accuracy:
        # Update the best-known validation accuracy.
        best_validation_accuracy = roc

        # Set the iteration for the last improvement to current.
        patience = 0

        # A string to be printed below, shows improvement found.
        improved_str = '*'

        # Save all variables of the TensorFlow graph to file.
        model.embedding_checkpoints(embeddings, mode="save")
    else:
        # An empty string to be printed below.
        # Shows that no improvement was found.
        improved_str = ''
        patience += 1

    # Status-message for printing.
    msg = "Epoch: {0:>6}, Train-Batch Loss: {1:.9f}, Validation AUC: {2:.9f} Validation PR: {3:.9f} {4}"
    print(msg.format(epoch + 1, avg_loss, roc, pr, improved_str))

    # Early stopping: If no improvement found in the required number of iterations, stop training the model
    if patience == early_stopping:
        print("Early stopping")
        # Break out from the for-loop.
        break

print("Total Training time: ", time()-start_time)

embeddings = model.embedding_checkpoints(mode="load")
print(embeddings.shape)

# Test-set
testing_edges = np.concatenate([test_edges, test_edges_false])
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

adj_matrix_rec = np.dot(embeddings, embeddings.T)
test_roc, test_ap = evaluate_ROC_from_matrix(testing_edges, test_edge_labels, adj_matrix_rec)

msg = "GNE Test ROC Score: {0:.9f}, GNE Test AP score: {1:.9f}"
print(msg.format(test_roc, test_ap))