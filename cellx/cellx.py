import numpy as np

import jax.numpy as jnp
import jax.random as random
from jax import grad, jit, vmap, lax, jacrev, jacfwd, jvp, vjp, hessian


#class Lattice(seed, cell_params, sim_params, 

symmetry_vec = jnp.array([[1], [-1]], dtype=int)

@jit
def dE_swap(ij, c, W, AL):
    """
    Energy differential after swapping cells i and j.
    Depends only on i, j, and their neighbors
    """
    i, j = ij
    ci = c[i]
    cj = c[j]

    E_local       = -W[c[(i, j), None], c[AL[(i, j),]]].sum()

    # Get energy if i and j swap cell types. The second line accounts for
    #   the fact that the first line calculates
    #       `E_i(i --> j) + E_j(j --> i)`
    #   But what we want is rather
    #       `E_i(i <-> j) + E_j(j <-> i)`
    #       `= E_i(i --> j) + E_j(j --> i) + (Wii + Wij - 2Wij)`
    E_local_swap  = -W[c[(j, i), None], c[AL[(i, j),]]].sum()  \
                    - 2 * W[ci, cj] + W[ci, ci] + W[cj, cj]
    
    return E_local_swap - E_local 


@jit
def quadratic_form(a, G):
    """Quadratic form of column vector `a` induced by matrix `G`"""
    return a.T @ G @ a


@jit
def P_swap(dE, beta):
    """
    Probability of a swap between cells i and j. Symmetric w.r.t. i and j.
    """
    
    # Glauber dynamics probability
    # return 1 / (1 + jnp.exp(beta * dE))
    
    # Metropolis acceptance probability
    return jnp.minimum(1., jnp.exp(-beta * dE))


@jit
def swap_ij(c, ij):
    """Swaps cells i and j in the cell type state vector `c`. """
    cji = c[ij][::-1]
    return c.at[ij].set(cji)


@jit
def accept_swap(c, P, ij):
    """
    Returns cell state and log-probability after swapping i <--> j
    """
    return swap_ij(c, ij), jnp.log(P)


@jit
def reject_swap(c, P, ij):
    """
    Returns cell state and log-probability after rejecting i <--> j
    """
    return c, jnp.log(1 - P)


@jit
def make_swap(c, P, ij, accept):
    """
    Returns cell state vector and log-probability of event 
    after an accepted/rejected swap of cells `i` and `j`.
    """
    return lax.cond(accept, accept_swap, reject_swap, c, P, ij)


@jit
def get_random_pair(key, AL):
    """Returns indices of a pair of adjacent cells"""
    
    i, Aj = random.randint(
        key=key, shape=(2,), minval=jnp.array([0, 0]), maxval=jnp.array(AL.shape)
    )
    j = AL[i, Aj]
    
    return jnp.array([i, j])


@jit
def take_MC_step(key, c, beta, W, AL):
    """
    Randomly selects a swap between adjacent cells and accepts/rejects.
    Acceptance is based on Metropolis algorithm.
    """
    key, sk1, sk2 = random.split(key, 3)
    
    # Pick random interface and acceptance threshold
    ij     = get_random_pair(sk1, AL)
    thresh = random.uniform(key=sk2)
    
    # Take a Metropolis step
    dE     = dE_swap(ij, c, W, AL)
    P      = P_swap(dE, beta)
    accept = P > thresh
    c, lnP = make_swap(c, P, ij, accept)
    
    return key, c, ij, dE, lnP, accept


@jit
def update(step, args):
    """Updates the state of simulation by one Metropolis time-step."""
    
    # Get state
    (
        key, 
        c_t,
        lnP_t,
        beta_t,
        W_t,
        AL,
    ) = args
    
    # Propose and accept/reject an MC step
    key, c, ij, dE, lnP, accept = take_MC_step(
        key, c_t[step], beta_t[step], AL, W_t[step]
    )
    
    # Return new state
    return (
        key, 
        c_t.at[step + 1].set(c),
        lnP_t.at[step].set(lnP),
        beta_t,
        W_t,
        AL,
    )


@jit 
def lnP_traj(lnP_t):
    """Returns log-probability of a trajectory"""
    pass


@jit
def n_cmatch_t(c_t, AL):
    """Returns number of homotypic interfaces at each time-point."""
    return cmatch_t(c_t, c_t[:, AL]).sum(axis=(1, 2)) // 2


@jit
def get_E_cell(c, W):
    return W[c[:, None], c[AL]].mean(axis=1)


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
