import argparse

def argument_parser():
    """
    Parses the argument to run the model.

    Returns
    -------
    model parameters: ArgumentParser Namespace
        The ArgumentParser Namespace that contains the model parameters.
    """
    parser = argparse.ArgumentParser(description="Run GNE")

    parser.add_argument("--data-folder", nargs="?", default="./data/", help="The data folder.")
    parser.add_argument("--dataset", nargs="?", default="yeast",
                        help="The name of the dataset. Default is yeast.")
    parser.add_argument("--edgelist-file", nargs="?", default="edgelist_biogrid.txt",
                        help="file that contains edgelists.")
    parser.add_argument("--gene-ids-file", nargs="?", default="gene_ids.tsv",
                        help="file that contains gene identifiers.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs. Default is 20.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Number of early-stopping iterations. Default is 10.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate. Default is 0.2.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate. Default is 0.001.")
    parser.add_argument("--l2", type=float, default=1e-5,
                        help="Learning rate. Default is 1e-5.")
    parser.add_argument("--latent-size", type=int, default=128,
                        help="Dimension of latent representation. Default is 128.")
    parser.add_argument("--net-emb-dim", type=int, default=128,
                        help="Dimension of structure representation. Default is 128.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Size of the batch. Default is 256.")
    parser.add_argument("--normalize", action="store_false", default=True)
    return parser.parse_args()
