import numpy as np
import numba

import matplotlib.pyplot as plt
import colorcet as cc

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

def binstr_to_state(binstr):
    """Converts a binary string to binary state vector."""
    return np.array([int(d) for d in binstr])

def binstr_to_bin_hash(binstr):
    """Converts binary string directly to binary hash (type integer)."""
    return state_to_bin_hash(binstr_to_state(binstr))

def binstr_to_indices(binstr):
    return binstr_to_state(binstr).nonzero()[0]

def indices_to_state(idx, n):
    arr = np.zeros(n, dtype=int)
    arr[np.asarray(idx)] = True
    return arr

def state_to_binstr(state):
    return "".join([str(i) for i in state])

def indices_to_binstr(idx, n):
    return state_to_binstr(indices_to_state(idx, n))

def indices_to_bin_hash(idx, n):
    return state_to_bin_hash(indices_to_state(idx, n))

##########################
####### Jake's code ######
##########################

# @numba.njit
def get_hex_neighbours(nx,ny):
    """
    Given the number of rows and columns, ny and nx, get the 6 neighbours of each cell.

    nc = number of cells
    i = 0,1,2,3,...,nc-1
    neighbours = of size nc x 6, the 6 neighbours for each i.

    Accounts for PBCS.
    :param nx:
    :param ny:
    :return:
    """
    nc = nx*ny
    i = np.arange(nc)
    neighbours = np.mod(np.column_stack(((i-nx,i-nx+1,i-1,i+1,i+nx,i+nx+1))),nc)
    return nc,i,neighbours

# @numba.njit
def roll_axis1(M,i,ny):
    """
    Jitted version of roll axis = 1.
    M is the matrix. i is the roll number, and ny is the size of the 2nd dimension.
    :param M:
    :param i:
    :param ny:
    :return:
    """
    Mnew = M.copy()
    if i != 0:
        for j in range(ny):
            Mnew[j] = np.roll(M[j],i)
    return Mnew

# @numba.njit
def get_translation_and_mirror_symmetries(state,nx,ny,nc):
    """
    Enumerates all possible transformations that preserve the pattern.

    I.e. nx * ny translations (shifting 0,1,2..., in the x direction, and in the y direction)
    And for each, considering the lr, ud and lr+ud reflections.
    :param state:
    :param nx:
    :param ny:
    :param nc:
    :return:
    """
    state_mat = state.reshape(nx,ny)
    trans_states_x = np.zeros((nx,nx,ny))
    for i in range(nx):
        trans_states_x[i] = roll_axis1(state_mat,i,ny)
    trans_states = np.zeros((nc,nx,ny),dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            trans_states[nx*j+i] = roll_axis1(trans_states_x[i].T,j,nx).T
    mirror_and_trans_states = np.zeros((nc*4,nc),dtype=np.int64)
    mirror_and_trans_states[::4] = trans_states.reshape(nc,nc)
    for i in range(nc):
        lr = np.fliplr(trans_states[i])
        mirror_and_trans_states[1 + i * 4] = lr.ravel()
        mirror_and_trans_states[2+i*4] = np.flipud(trans_states[i]).ravel()
        mirror_and_trans_states[3+i*4] = np.flipud(lr).ravel()
    return mirror_and_trans_states

def state_to_bin_hash(state):
    """
    Converts a state vector e.g. np.array((0,0,1,1,...)) to a string (reduce), then binarize, using int.
    :param state:
    :return:
    """
    return int(reduce(lambda x, y: x + str(y), state, ''),2)

def bin_hash_to_states(bin_hash,nc):
    """
    Converts a binary hash back to a state vector, the inverse of the above.

    nc = number of cells, important in padding. See below.
    :param bin_hash:
    :param nc:
    :return:
    """
    state_string = format(bin_hash,"b")
    state_string = "0"*(nc - len(state_string)) + state_string
    return np.array(list(state_string)).astype(np.int64)

def get_hash(state,nx,ny,nc):
    """
    Given a state vector, get the unique hash, accounting for all possible transformations.

    Unique hash = min(set of hashes under transformations).
    :param state:
    :param nx:
    :param ny:
    :param nc:
    :return:
    """
    all_states = get_translation_and_mirror_symmetries(state,nx,ny,nc)
    bin_hash = min(map(state_to_bin_hash,all_states))
    return bin_hash

# @numba.njit
def get_states_post_swap(_states,nx):
    """
    Get the state vectors post-swap.

    Two possible swaps are considered (a straight swap i.e. 0 <--> 1) and a diagonal swap (0 <--> nx).

    The idea is that, if we enumerate all possible states (i.e. like 5M if we have 12/25 red), and perform one of two
    swaps, then given all translations are accounted for in the hashing, this should sample ALL possible swaps of any state.

    Think this is correct, but don't have a proof.
    :param _states:
    :param nx:
    :return:
    """
    states = np.row_stack(_states)
    straight_swap = states.copy()
    diag_swap = states.copy()
    straight_swap[:,1],straight_swap[:,0] = states[:,0].copy(),states[:,1].copy()
    diag_swap[:,nx],diag_swap[:,0] = states[:,0].copy(),states[:,nx].copy()
    return straight_swap,diag_swap


# @numba.njit
def get_state_from_combo(combo):
    """
    Combo is a tuple of cell ids, the "red" cells.
    Return a state vector, where 0 = blue, 1 = red.
    :param combo:
    :return:
    """
    state = np.zeros(25, dtype=np.int64)
    for c in combo:
        state[c] = 1
    return state

combos = combinations(np.arange(25),4) #with 4 red cells. Scale up at your own peril. 

states = list(map(get_state_from_combo,combos))

nx,ny,nc = 5,5,25

def _get_hash(state):
    """A wrapper for get_hash, so I can use the map function. """
    return get_hash(state, nx, ny, nc)

