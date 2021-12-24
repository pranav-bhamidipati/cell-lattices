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
from tqdm import tqdm
import pandas as pd

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

# Get total number of permutations this tissue can undergo
ncomb = int(sp.comb(n, n_sub))
n_cpu = psutil.cpu_count(logical=False)

# Create progress bar (approximate, since multiprocess is lazy)
pbar = tqdm(total=ncomb // n_cpu)

# Define functions
def get_graph_hashes(idx):
    subG        = G.subgraph(idx)
    graph_hash  = nx.algorithms.weisfeiler_lehman_graph_hash(subG)
    # binary_hash = cx.indices_to_bin_hash(idx, n)
    # return graph_hash, binary_hash
    
    # Update progress
    pbar.update(1)
    
    return cx.indices_to_binstr(idx, n), graph_hash

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
    
    print("Getting all cell type combinations")

#     print("Making Boolean combinations")
    # Get all cell type combinations as indices (for computing # components)
    combs_idx = [i for i in combinations(np.arange(n), n_sub)]

#     # Get combinations as Boolean data (for UMAP)
#     combs_bool = np.zeros((ncomb, n), dtype=bool)
#     for i, idx in enumerate(tqdm(combs_idx)):
#         combs_bool[i, idx] = True

#     print("Making string combinations")

#     # Get combinations as strings
#     combs_str = ["".join([str(int(c)) for c in _comb]) for _comb in combs_bool]

    print("Assembling worker pool")

    # Get worker pool
    pool = mp.Pool(psutil.cpu_count(logical=False))
    
    print("Computing hashes for graph")

    # Perform parallel computation
    # results = pool.imap_unordered(get_graph_hashes, combs_idx)
    results = pool.imap(get_graph_hashes, combs_idx, chunksize=5)
    
    # Unravel results
#    n_comp, n_edges = map(list, zip(*result_list))
    bin_strings, graph_hashes = map(list, zip(*results))
    
    print("Closing worker pool")

    pool.close()
    pool.join()

pbar.close()

# Combine into dataframe
df = pd.DataFrame(dict(
    binary_str = bin_strings,
    graph_hash = graph_hashes,
#    combination = combs_str,
#    n_components = n_comp,
))
df = df.sort_values("binary_str")

# Get smallest binary hash for each graph hash (~50x compression)
df = df.groupby("graph_hash").agg(min).reset_index()

if save:
    print("Saving!")
    data_fname = "cellgraph_hashes.csv"
    data_fpath = os.path.join(save_dir, data_fname)
    df.to_csv(data_fpath, index=False)
    print(f"Saved to {data_fpath}")
