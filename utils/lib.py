import numpy as np
import scipy.sparse as sp
import warnings
from texttable import Texttable
import networkx as nx
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def table_printer(args):
    """
    Print the parameters of the model in a Tabular format
    Parameters
    ---------
    args: argparser object
        The parameters used for the model
    """
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.add_rows([["Parameter", "Value"]] +
                   [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(table.draw())

# code from graph2gauss https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.
    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)
    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix
    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def load_network(filename, num_genes):
    print("### Loading [%s]..." % (filename))
    edgelist = pd.read_csv(filename, sep=' ')
    G = nx.from_edgelist(edgelist.values[:, :2].tolist())
    A = nx.adjacency_matrix(G)
    return A

def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False
    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges
    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))
    
    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            # make sure original graph is connected
            assert sp.csgraph.connected_components(A)[0] == 1
            A_hold = sp.csgraph.minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(A)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[np.unique(
            random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))
    s_train_zeros = len(train_ones)

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    train_zeros = test_zeros[:s_train_zeros]
    test_zeros = test_zeros[s_train_zeros:]

    val_zeros = test_zeros[:s_val_ones]
    test_zeros = test_zeros[s_val_ones:]
     
    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        def symmetrize(x): return np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        set_of_train_zeros = set(map(tuple, train_zeros))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_zeros.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, train_zeros))) == len(train_ones) * neg_mul
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or sp.csgraph.connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, train_zeros, val_ones, val_zeros, test_ones, test_zeros


def edge_cover(A):
    """
    Approximately compute minimum edge cover.
    Edge cover of a graph is a set of edges such that every vertex of the graph is incident
    to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.
    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix
    Returns
    -------
    edges : array-like, shape [?, 2]
        The edges the form the edge cover
    """

    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges

def get_edge_embeddings(Embeddings, edge_list):
    embs = []
    for i in range(len(edge_list)):
        edge = np.array(edge_list)[i, :]
        node1 = int(edge[0])
        node2 = int(edge[1])
        emb1 = Embeddings[node1]
        emb2 = Embeddings[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Evaluate ROC using predicted matrix
def evaluate_ROC_from_matrix(X_edges, y_true, matrix):
    y_predict = [sigmoid(matrix[int(edge[0]), int(edge[1])]) for edge in X_edges]
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    pr = average_precision_score(y_true, y_predict)
    return roc, pr
