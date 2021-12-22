import os

import networkx as nx
import cell_lattices as cx

import numpy as np

import colorcet as cc
import matplotlib.pyplot as plt
# %matplotlib widget

save     = True
save_dir = os.path.realpath("plots")
dpi      = 300

# Get coordinates and adjacency for a periodic square lattice
rows = cols = 8
X = cx.hex_grid(rows, cols)
A = cx.make_adjacency_periodic(rows, cols)
n = A.shape[0]

# Make master cell graph
G = nx.from_numpy_matrix(A)

# Assign cell types
#cell_types = np.zeros(A.shape[0], dtype=int)
#cell_types[:n//2] = 1

# Set plot options
prows = 4
pcols = 4
plot_kw = dict(
    vmin=0,
    vmax=1.,
    cmap="kb",
    ec="w",
)

# Make plot
fig   = plt.figure()
for i in range(prows * pcols):
    
    # Assign cell type
    cell_idx = np.random.choice(np.arange(n), n//2, replace=False)
    
    # Get subgraph and its topology
    subG = G.subgraph(cell_idx)
    nc   = nx.number_connected_components(subG)
    
    # Plot
    ax  = fig.add_subplot(prows, pcols, i + 1)
    _val = np.zeros(n)
    _val[cell_idx] = 1
    cx.plot_hex_sheet(ax, X, _val, **plot_kw)

#    pos  = nx.spring_layout(subG)
#    nx.draw_networkx_nodes(subG, X, node_size=5)
#    nx.draw_networkx_edges(subG, X)
    
    ax.set_title(f"{nc} comps", fontsize=8)

if save:
    fname = "graphtest_periodic.png"
    fpath = os.path.join(save_dir, fname)
    print(fpath)
    plt.savefig(fpath, dpi=dpi)
else:
    plt.show()
