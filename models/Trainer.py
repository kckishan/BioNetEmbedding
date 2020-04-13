import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from utils import *
from .BioNetEmbedding import BioNetEmbedding
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

class Trainer:

    def __init__(self, args):
        self.args = args
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_features(self):

        # Define the input to BioNetEmbedding model
        edgelist_file = './data/' + self.args.dataset + '/' + self.args.edgelist_file

        A = load_network(edgelist_file)
        self.N = A.shape[0]
        self.data = train_val_test_split_adjacency(A, p_val=0.1, p_test=0.05, seed=0, neg_mul=1,
                                                     every_node=False, connected=False, undirected=True,
                                                      use_edge_cover=False, set_ops=True, asserts=False)

        print("Total genes:", self.N)
        print("Training interactions (positive):   ", len(self.data['train_edges']))
        print("Training interactions (negative):   ", len(self.data['train_edges_false']))
        print("Validation interactions (positive): ", len(self.data['val_edges']))
        print("Validation interactions (negative): ", len(self.data['val_edges_false']))
        print("Test interactions (positive):       ", len(self.data['test_edges']))
        print("Test interactions (negative):       ", len(self.data['test_edges_false']))

    def setup_model(self):
        self.model = BioNetEmbedding(self.N , self.args, self.device)
        self.model = self.model.to(self.device)
        print(self.model)

    def setup_training_data(self):
        train_edges = torch.Tensor(self.data['train_edges']).type(torch.LongTensor)
        trainData = BioNetEmbeddingDataset(train_edges)

        self.validation_edges = np.concatenate([self.data['val_edges'], self.data['val_edges_false']])
        self.val_edge_labels = np.concatenate([np.ones(len(self.data['val_edges'])), np.zeros(len(self.data['val_edges_false']))])

        self.train_loader = DataLoader(trainData, batch_size=self.args.batch_size, shuffle=True)

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)

        val_loss_min = np.Inf
        start_time = time()

        patience = 0
        best_validation_accuracy = 0.0
        early_stopping = 2

        for epoch in range(self.args.epochs):
            train_loss = 0.0

            self.model.train()
            for i, (data, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                emb, loss = self.model(data, labels)
                loss.backward()
                optimizer.step()

                # adding losses
                train_loss += loss.item()
            
            avg_loss = train_loss / len(self.train_loader)
            # Get embeddings from trained model
            embeddings  = self.model.get_embeddings()
            
             # link prediction test
            ####################################################################################
            adj_matrix_rec = np.dot(embeddings, embeddings.T)
            roc, pr = evaluate_ROC_from_matrix(self.validation_edges, self.val_edge_labels, adj_matrix_rec)

            if roc > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = roc

                # Set the iteration for the last improvement to current.
                patience = 0

                # A string to be printed below, shows improvement found.
                improved_str = '*'

                # Save all variables of the TensorFlow graph to file.
                self.model.embedding_checkpoints(embeddings, mode="save")
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

    def evaluate(self):
        embeddings = self.model.embedding_checkpoints(mode="load")

        # Test-set
        testing_edges = np.concatenate([self.data['test_edges'], self.data['test_edges_false']])
        test_edge_labels = np.concatenate([np.ones(len(self.data['test_edges'])), np.zeros(len(self.data['test_edges_false']))])

        adj_matrix_rec = np.dot(embeddings, embeddings.T)
        test_roc, test_ap = evaluate_ROC_from_matrix(testing_edges, test_edge_labels, adj_matrix_rec)

        msg = "Test ROC Score: {0:.9f}, Test AP score: {1:.9f}"
        print(msg.format(test_roc, test_ap))



