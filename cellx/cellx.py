import numpy as np
import numba

from functools import reduce
from itertools import combinations


def make_adjacency_periodic(rows, cols=0, dtype=np.float32, **kwargs):
    """Construct adjacency matrix for a periodic hexagonal 
    lattice of dimensions rows x cols."""
    
    # Check if square
    if cols == 0:
        cols = rows
    
    # Initialize matrix
    n = rows * cols
    Adj = np.zeros((n,n), dtype=dtype)
    for i in range(cols):
        for j in range(rows):
            
            # Get neighbors of cell at location i, j
            nb = np.array(
                [
                    (i    , j + 1),
                    (i    , j - 1),
                    (i - 1, j    ),
                    (i + 1, j    ),
                    (i - 1 + 2*(j%2), j - 1),
                    (i - 1 + 2*(j%2), j + 1),
                ]
            )
            
            nb[:, 0] = nb[:, 0] % cols
            nb[:, 1] = nb[:, 1] % rows
            
            # Populate Adj
            nbidx = np.array([ni*rows + nj for ni, nj in nb])
            Adj[i*rows + j, nbidx] = 1
    
    return Adj


def get_adjacency_list_periodic(rows, cols=0, dtype=np.float32, **kwargs):
    """Construct adjacency matrix for a periodic hexagonal 
    lattice of dimensions rows x cols."""
    
    # Check if square
    if cols == 0:
        cols = rows
    
    A = make_adjacency_periodic(rows, cols, dtype, **kwargs)

    return A.nonzero()[1].reshape(A.shape[0], 6)


def hex_grid(rows, cols=0, r=1., sigma=0, **kwargs):
    """
    Returns XY coordinates of a regular 2D hexagonal grid 
    (rows x cols) with edge length r. Points are optionally 
    passed through a Gaussian filter with std. dev. = sigma * r.
    """
    
    # Check if square grid
    if cols == 0:
        cols = rows
    
    # Populate grid 
    x_coords = np.linspace(-r * (cols - 1) / 2, r * (cols - 1) / 2, cols)
    y_coords = np.linspace(-np.sqrt(3) * r * (rows - 1) / 4, np.sqrt(3) * r * (rows - 1) / 4, rows)
    X = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            X.append(np.array([x + (j % 2) * r / 2, y]))
    X = np.array(X)
    
    # Apply Gaussian filter if specified
    if sigma != 0:
        X = np.array([np.random.normal(loc=x, scale=sigma*r) for x in X])
    
    return X

def get_outer_idx(rows, cols):
    """Returns the indices of cells on the border of the lattice grid"""
    return np.array([
        rows * c + r
        for c in range(cols)
        for r in range(rows)
        if ((r in (0, rows - 1)) or (c in (0, cols - 1)))
    ])


# Vectorized rounding
vround = np.vectorize(round)


@numba.njit
def normalize(x, xmin, xmax):
    """Normalize `x` given explicit min/max values. """
    return (x - xmin) / (xmax - xmin)

