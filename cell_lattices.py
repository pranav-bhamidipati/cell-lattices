import numpy as np
import numba

import matplotlib.pyplot as plt
import colorcet as cc


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


def plot_hex_sheet(
    ax,
    X,
    var,
    r=1.,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    ec=None,
    title=None,
    poly_kwargs=dict(),
    xlim=(),
    ylim=(),
    axis_off=True,
    aspect=None,
    colorbar=False,
    cbar_aspect=20,
    extend=None,
    cbar_kwargs=dict(),
    poly_padding=0.,
    scalebar=False,
    sbar_kwargs=dict(),
    **kwargs
):
    
    # Clear axis (allows you to reuse axis when animating)
    # ax.clear()
    if axis_off:
        ax.axis("off")
    
    # Get min/max values in color space
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()
    
    # Get colors based on supplied values of variable
    if type(cmap) is str:
        _cmap = cc.cm[cmap]
    else:
        _cmap = cmap
    colors = np.asarray(_cmap(normalize(var, vmin, vmax)))
    
    # Get polygon size. Optionally increase size  
    #  so there's no visual gaps between cells
    _r = (r + poly_padding)
    
    # Plot cells as polygons
    for i, (x, y) in enumerate(X):
        
        ax.fill(
            _r * _hex_x + x, 
            _r * _hex_y + y, 
            fc=colors[i], 
            ec=ec,
            **kwargs
        )
    
    # Set figure args, accounting for defaults
    if title is not None:
        ax.set_title(title)
    if not xlim:
        xlim=[X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim=[X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect=1
    ax.set(
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
    )

    if colorbar:
        
        # Calculate colorbar extension if necessary
        if extend is None:
            n = var.shape[0]        
            ns_mask = ~np.isin(np.arange(n), sender_idx)
            is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
            is_over_max  = var.max(initial=0.0, where=ns_mask) > vmax
            _extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
        else:
            _extend = extend
        
        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin, vmax), 
                cmap=_cmap), 
            ax=ax,
            aspect=cbar_aspect,
            extend=_extend,
            **cbar_kwargs
        )
        
    if scalebar:
        _scalebar = ScaleBar(**sbar_kwargs)
        ax.add_artist(_scalebar)

@numba.njit
def normalize(x, xmin, xmax):
    """Normalize `x` given explicit min/max values. """
    return (x - xmin) / (xmax - xmin)

