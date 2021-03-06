#!/usr/bin/env python
# coding: utf-8
import os
from itertools import combinations
import psutil
import multiprocessing as mp

import networkx as nx
import cell_lattices as cx

import numpy as np
import scipy.special as sp
from scipy.spatial import distance as dist
from tqdm import tqdm
import pandas as pd

import umap

import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns

# Options for saving output(s)
save     = True
save_dir = os.path.realpath("/home/pbhamidi/git/cell-lattices/data")
# fmt      = "png"
# dpi      = 300

# Get coordinates and adjacency for a periodic square lattice
rows = cols = 5
X = cx.hex_grid(rows, cols)
A = cx.make_adjacency_periodic(rows, cols)

# Make master cell graph
G = nx.from_numpy_matrix(A)

# Get total number of cells and a subset
n = A.shape[0]
n_sub = n // 2

# Define functions
def get_graph_hashes(idx):
    subG        = G.subgraph(idx)
    graph_hash  = nx.algorithms.weisfeiler_lehman_graph_hash(subG)
    binary_hash = cx.indices_to_bin_hash(idx, n)
    return graph_hash, binary_hash

def n_connected_components(idx):
    """Computes number of connected components given cell indices"""
    subG = G.subgraph(idx)
    nc = nx.number_connected_components(subG)
    nc_inv = nx.number_connected_components(
        G.subgraph(
            [i for i in range(n) if i not in set(idx)]
        )
    )
    return nc + nc_inv

# Parallelize calculation of tissue topology (# coneccted components)
if __name__ == '__main__':
    
    # Get total number of permutations this tissue can undergo
    ncomb = int(sp.comb(n, n_sub))

    # Get all cell type combinations as indices (for computing # components)
    combs_idx = [i for i in combinations(np.arange(n), n_sub)]

    print("Making Boolean combinations")

    # Get combinations as Boolean data (for UMAP)
    combs_bool = np.zeros((ncomb, n), dtype=bool)
    for i, idx in enumerate(tqdm(combs_idx)):
        combs_bool[i, idx] = True

    print("Making string combinations")

    # Get combinations as strings
    combs_str = ["".join([str(int(c)) for c in _comb]) for _comb in combs_bool]

    print("Assembling worker pool")

    # Get worker pool
    pool = mp.Pool(psutil.cpu_count(logical=False))
    
    print("Computing n_components")

    # Perform parallel computation
    result_list = pool.map(get_graph_hashes, combs_idx)
    
    # Unravel results
#    n_comp, n_edges = map(list, zip(*result_list))
    graph_hashes, binary_hashes = map(list, zip(*result_list))

    print("Closing worker pool")

    pool.close()

# Combine into dataframe
df = pd.DataFrame(dict(
    binary_hash=binary_hashes
#    combination = combs_str,
#    n_components = n_comp,
))

if save:
    data_fname = "cellgraph_enumeration.csv"
    data_fpath = os.path.join(save_dir, data_fname)
    df.to_csv(data_fpath, index=False)
    print(f"Saved to {data_fpath}")

# # Plot
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=colors)
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of tissue topologies ($5 \times 5$)', fontsize=24)
    
# if save:
#     fname = "topology_UMAP_5x5"
#     fpath = os.path.join(save_dir, fname + "." + fmt)
#     plt.savefig(fpath, dpi=dpi)
# else:
#     plt.show()

