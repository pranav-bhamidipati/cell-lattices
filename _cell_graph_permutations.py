import os
from itertools import permutations

import networkx as nx
import cell_lattices as cx

import numpy as np
import scipy.special as sp
from scipy.spatial import distance as dist

import colorcet as cc
import matplotlib.pyplot as plt
# %matplotlib widget

save     = True
save_dir = os.path.realpath("plots")
dpi      = 300

# Get coordinates and adjacency for a periodic square lattice
rows = cols = 5
X = cx.hex_grid(rows, cols)
A = cx.make_adjacency_periodic(rows, cols)
n = A.shape[0]
n_sub = n // 2

# Get number of tissue permutations
ncomb = int(sp.comb(n, n_sub))

# Make master cell graph
G = nx.from_numpy_matrix(A)

# Initialize outputs
perms = []
ctype = np.zeros(n, dtype=int)
n_components = []

# Iterate over all possible combinations
i = 0
for idx in permutations(np.arange(n), n_sub):
    
    if i in [int(i) for i in ncomb * np.linspace(0, 1, ncomb)]:
        print(f"{i/ncomb:.1%} complete")

    _idx = np.asarray(idx)

    # Add to list of permutations
    _perm = ctype.copy()
    _perm[_idx] = 1
    perms.append(_perm)

    # Get number of connected components
    _ncomp = nx.number_connected_components(G.subgraph(_idx))
    n_components.append(_ncomp)

    i += 1

print(len(n_components))

# Assign cell types
#cell_types = np.zeros(A.shape[0], dtype=int)
#cell_types[:n//2] = 1

# Set plot options
#prows = 4
#pcols = 4
#plot_kw = dict(
#    vmin=0,
#    vmax=1.,
#    cmap="kb",
#    ec="w",
#)
#
# Make plot
#fig   = plt.figure()
#for i in range(prows * pcols):
#    
#    # Assign cell type
#    cell_idx = np.random.choice(np.arange(n), n//2, replace=False)
#    
#    # Get subgraph and its topology
#    subG = G.subgraph(cell_idx)
#    nc   = nx.number_connected_components(subG)
#    
#    # Plot
#    ax  = fig.add_subplot(prows, pcols, i + 1)
#    _val = np.zeros(n)
#    _val[cell_idx] = 1
#    cx.plot_hex_sheet(ax, X, _val, **plot_kw)
#
##    pos  = nx.spring_layout(subG)
##    nx.draw_networkx_nodes(subG, X, node_size=5)
##    nx.draw_networkx_edges(subG, X)
#    
#    ax.set_title(f"{nc} comps", fontsize=8)
#
#if save:
#    fname = "graphtest_periodic.png"
#    fpath = os.path.join(save_dir, fname)
#    print(fpath)
#    plt.savefig(fpath, dpi=dpi)
#else:
#    plt.show()


