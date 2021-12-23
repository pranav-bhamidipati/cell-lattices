#!/usr/bin/env python
# coding: utf-8
import os
from itertools import combinations
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

# Get total number of permutations this tissue can undergo
ncomb = int(sp.comb(n, n_sub))

# Get all cell type combinations as indices (for computing # components)
combs_idx = [i for i in combinations(np.arange(n), n_sub)]

print("1")

# Get combinations as Boolean data (for UMAP)
combs_bool = np.zeros((ncomb, n), dtype=bool)
for i, idx in enumerate(tqdm(combs_idx)):
    combs_bool[i, idx] = True

print("2")

# Get combinations as strings
combs_str = ["".join([str(int(c)) for c in _comb]) for _comb in combs_bool]

print("3")

# Define computation
def n_connected_components(idx):
    """Computes number of connected components given cell indices"""
    return nx.number_connected_components(G.subgraph(idx))

# Parallelize calculation of tissue topology (# coneccted components)
if __name__ == '__main__':
    
    # Get worker pool
    n_workers = mp.cpu_count()
    pool = mp.Pool(n_workers)
    
    print("3")

    # Perform parallel computation
    result_list = pool.map(n_connected_components, combs_idx)
    n_comp = np.asarray(result_list)

    print("4")

## Perform UMAP
# Select data
data_slice = slice(None, None, None)
# data_slice = slice(0, 3000, 300)
data       = combs_bool[data_slice]
clusters   = n_comp[data_slice]
colors     = [sns.color_palette()[i] for i in clusters]

# Perform UMAP with progress
reducer   = umap.UMAP(metric="hamming", verbose=True)
embedding = reducer.fit_transform(combs)

#### DUMMY EMBEDDING ############
# embedding = np.random.random((ncomb, 2))
#################################

print("5")

# Combine into dataframe
df = pd.DataFrame(dict(
    combination = combs_str,
    n_components = n_comp,
    umap_x = embedding[:, 0],
    umap_y = embedding[:, 1],
))

print("6")

data_fname = "cellgraph_embedding.csv"
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

