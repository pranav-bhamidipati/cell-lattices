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

import umap

import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns


# Options for saving output(s)
save     = True
save_dir = os.path.realpath("/home/pbhamidi/git/cell-lattices/plots")
fmt      = "png"
dpi      = 300

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

# Make generator of all cell type combinations
gen = combinations(np.arange(n), n_sub)

# Make Boolean array of all combinations
combs = np.zeros((ncomb, n), dtype=bool)
for i in tqdm(range(ncomb)):
    idx = next(gen)
    combs[i, idx] = True

def n_connected_components(idx):
    """Compute number of connected components given cell indices"""
    return nx.number_connected_components(G.subgraph(idx))

# Parallelize calculation of tissue topology (# coneccted components)
if __name__ == '__main__':
    
    # Get cluster
    n_workers = mp.cpu_count()
    pool = mp.Pool(n_workers)
    
    # Compute results as array
    iterator = combinations(np.arange(n), n_sub)
    result_list = pool.map(n_connected_components, iterator)
    ncc = np.asarray(result_list)

## Perform UMAP
# Select data
data_slice = slice(None, None, None)
# data_slice = slice(0, 3000, 300)
data       = combs[data_slice]
clusters   = ncc[data_slice]
colors     = [sns.color_palette()[i] for i in clusters]

# # Perform UMAP with progress
# reducer   = umap.UMAP(metric="manhattan", verbose=True)
# embedding = reducer.fit_transform(combs)
#####
embedding = np.random.random((ncombs, 2))

combs_str = ["".join([str(int(c)) for c in comb]) for comb in combs]
df = pd.DataFrame(dict(
    combination = combs_str,
    n_components = ncc,
    umap_x = embedding[:, 0],
    umap_y = embedding[:, 1],
))

data_fname = os.path.join(save_dir, "cellgraph_embedding.csv")
df.to_csv(data_fname, index=False)


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

