#import utils as ut
import numpy as np
#import numba


# Vertices of a regular hexagon centered at (0,0) with width 1.
_hex_vertices = (
    np.array([
        np.cos(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6), 
        np.sin(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
    ]).T 
    / np.sqrt(3)
)

# X and Y values of a hexagon's vertices
_hex_x, _hex_y = _hex_vertices.T


def interface_endpoints(Xi, Xj):
    """
    Returns the endpoints of the face shared by neighboring regions 
    (cells) i and j on a regular hexagonal lattice, with centers given 
    by `Xi` and `Xj`. Does not account for periodicity.
    """
    Xmid = (Xi + Xj) / 2
    rij = (Xj - Xi) / 2
    
    # Vector from midpoint to shared vertex in CCW direction
    orth = np.array([-rij[1], rij[0]]) / np.sqrt(3)
    
    return np.array([Xmid - orth, Xmid + orth])


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

