import numpy as np
import networkx as nx


### Graph operations


def n_homotypic_regions(c, A):
    """
    Returns number of islands of the same cell type (# connected components of
    all homotypic cell grpahs)
    """
    G = nx.from_numpy_array(A)
    ncc = 0
    for ctype in np.unique(c):
        mask = c == ctype
        ncc += nx.number_connected_components(G.subgraph(mask.nonzero()[0]))

    return ncc
