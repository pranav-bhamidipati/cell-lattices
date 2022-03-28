from functools import partial

from . import utils

import numpy as np

import jax.numpy as jnp
import jax.random as random
from jax import grad, jit, vmap, lax, jacrev, jacfwd, jvp, vjp, hessian

#class Lattice(seed, cell_params, sim_params, 


def random_c0(subkeys, odds_c, n):
    """Make random initial conditions given odds ratio of cell types."""

    n_ctypes = len(odds_c)
    n_c = (n * odds_c / odds_c.sum()).astype(int)
    n_c = n_c.at[0].add(n - n_c.sum())

    c0 = jnp.repeat(jnp.arange(n_ctypes), n_c)
    
    nmap = np.ndim(subkeys) - 1
    fun  = lambda sk: random.permutation(sk, c0)
    for _ in range(nmap):
        fun = vmap(fun)
    
    return n_c, fun(subkeys)


@jit
def dE_swap(ij, c, W, AL):
    """
    Energy differential after swapping cells i and j.
    Depends only on i, j, and their neighbors
    """
    new_c = c.at[ij].set(c[ij[::-1]])

    E_local      = -W[    c[ij, None],     c[AL[ij]]].sum()
    E_local_swap = -W[new_c[ij, None], new_c[AL[ij]]].sum()
    
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
#    return 1 / (1 + jnp.exp(beta * dE))
    
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
    return swap_ij(c, ij)


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
def take_MC_step(key, c, beta, W, AL, n):
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
    new_c  = make_swap(c, P, ij, accept)
    
    expected_dE = P * dE

    return key, new_c, expected_dE


@jit
def propose_swap(key, c, beta, W, AL):
    """
    """
    ij     = get_random_pair(key, AL)
    c_swap = swap_ij(c, ij)
    dE     = dE_swap(ij, c, W, AL)
    P      = P_swap(dE, beta)
    
    return ij, c_swap, dE, P


@jit
def local_alignment(c, A, k, I, O):
    
    s      = I[c] @ O
    s_swap = I[c_swap] @ O
    m_diff_nb      = (A_k * diff_nb) @ s / n_diff_nb 



@jit
def local_alignment_change(ij, c, c_swap, AL, k, I, O):
    
    A_k = get_knn_adjacency_matrix(AL, k)
    
    # cells that are neighbors (within k radii) of 
    #   `i` but not `j` and vice-versa - i.e. different neighbors
    diff_nb = jnp.expand_dims(jnp.logical_xor(*A_k[ij]), 1)
    n_diff_nb = 4 * k + 2

    s      = I[c] @ O
    s_swap = I[c_swap] @ O
    m_diff_nb      = (A_k * diff_nb) @ s / n_diff_nb 
    m_diff_nb_swap = (A_k * diff_nb) @ s_swap / n_diff_nb 
    
    return ((m_diff_nb_swap ** 2) - (m_diff_nb ** 2)).sum()

mapped_local_alignment_change = vmap(
    local_alignment_change, in_axes=(None, None, None, None, 0, None, None)
)


#@jit
def take_MC_step2(args, step):
    """
    Randomly selects a swap between adjacent cells and accepts/rejects.
    Acceptance is based on Metropolis algorithm.
    """
    key, c_t, beta_t, W, AL, *align_args = args
    c    = c_t[step]
    beta = beta_t[step]

    new_key, sk1, sk2 = random.split(key, 3)
    
    # Propose a random swap
    ij, c_swap, dE, P = propose_swap(sk1, c, beta, W, AL)
    expected_d_eta = P * mapped_local_alignment_change(
        ij, c, c_swap, AL, *align_args
    ).mean()

    # Accept/reject
    thresh  = random.uniform(key=sk2)
    do_swap = P > thresh
    new_c   = lax.cond(do_swap, lambda: c_swap, lambda: c)

    return (
        new_key, c_t.at[step + 1].set(new_c), beta_t, W, AL, *align_args
    ), expected_d_eta


@partial(jit, static_argnums=(2, 3, 4))
def simulate(theta, args, nsweeps, n, n_ctypes): 
    
    key, c, t, _, *more_args = args
    
    beta_t = jnp.power(10., -utils.map_linear(t, theta[0], theta[1]))
    W = jnp.eye(n_ctypes) * theta[2]
    
    new_args, expected_d_etas = lax.scan(
        take_MC_step2, 
        (key, c, beta_t, W, *more_args),
        jnp.repeat(jnp.arange(nsweeps), n),
    )
    
    return new_args, expected_d_etas


@partial(jit, static_argnums=(2, 3, 4))
def simulate_loss(theta, args, nsweeps, n, n_ctypes): 
    return simulate(theta, args, nsweeps, n, n_ctypes)[1].mean()


@partial(jit, static_argnums=(2, 3))
def update(theta, args, nt, lr):
    """Performs one update step on T."""

    # Compute the gradients on replicates
    eta, grads = jax.value_and_grad(
        simulate,
    )(T, key, l, nt)
    
    new_T = T - grads * lr_toy

    return new_T, loss, grads


@partial(jit, static_argnums=3)
def update_toy(T, key, l, nt, lr_toy):
    """Performs one update step on T."""

    # Compute the gradients on replicates
    loss, grads = jax.value_and_grad(
        simulate_loss,
    )(T, key, l, nt)
    
    new_T = T - grads * lr_toy

    return new_T, loss, grads


@jit
def MC_iteration(step, args):
    key, c, *extra = args
    key, c, expected_dE = take_MC_step(*args)
    return key, c, *extra


@jit
def MC_sweep(key, c, beta, W, AL, n):
    args = (key, c, beta, W, AL, n)
    return lax.fori_loop(0, n, MC_iteration, args)


@jit
def n_cmatch_t(c_t, AL):
    """Returns number of homotypic interfaces at each time-point."""
    return cmatch_t(c_t, c_t[:, AL]).sum(axis=(1, 2)) // 2


@jit
def get_E_cell(c, W):
    return W[c[:, None], c[AL]].mean(axis=1)


#### sorting metrics

def get_identity(n_ctypes):
    """Returns the (n_ctypes, n_ctypes) identity matrix."""
    return jnp.eye(n_ctypes, dtype=int)


def get_difference_matrix(n_ctypes):
    """
    Returns a (n_ctypes, n_ctypes - 1) matrix `O` with -1 on the principal 
    diagonal and 1 elsewhere. `O @ u` thus computes a difference on the 
    components of `u`.
    """
    return 1 - 2 * jnp.eye(n_ctypes, n_ctypes - 1, dtype=int)


@jit 
def get_num_neighbors(k):
    return 1 + 3 * k * (k + 1)


@jit
def pow_matrix(A, k):
    return lax.fori_loop(1, k, lambda i, M: jnp.matmul(M, A), A)


@jit 
def get_knn_adjacency_matrix(AL, k):
    
    n, nnb = AL.shape
    
    diag_true = jnp.diag(jnp.ones(n, dtype=bool))
    
    A = adjacency_matrix_from_adjacency_list(AL, dtype=bool)
    A = A | diag_true
    A = pow_matrix(A, k)
    return A


equal_vec_scalar  = vmap(lambda a, b: a == b, (0, None))
equal_outer_1d_1d = vmap(equal_vec_scalar, (None, 0))
equal_outer_1d_2d = vmap(equal_outer_1d_1d, (None, 0))
equal_outer_2d_1d = vmap(equal_outer_1d_1d, (0, None))

mult_vec_scalar  = vmap(lambda a, b: a * b, (0, None))
mult_outer_1d_1d = vmap(mult_vec_scalar, (None, 0))
mult_outer_1d_2d = vmap(mult_outer_1d_1d, (None, 0))
mult_outer_2d_1d = vmap(mult_outer_1d_1d, (0, None))


@jit
def local_spin(c, AL, k):
    """
    """
    A_k = get_knn_adjacency_matrix(AL, k)
    nnb = get_num_neighbors(k)
    
    s_i = jnp.array([-1, 1])[c]
    
    return A_k @ s_i / nnb


@jit
def knn_alignment_per_cell(c, AL, k, I, O):
    """
    Return alignment of cell types `c` in local neighborhoods.
    `c` is the cell type vector of shape `(n,)` with dtype `int`
    `A` is the `(n, n)`cell-cell adjacency matrix (can be Boolean)
    `I` is the `(n_ctypes, n_ctypes)` identity matrix, where `n_ctypes` 
    is the number of cell types in the tissue.
    `O` is the `(n_ctypes, n_ctypes - 1)` difference matrix with `-1` on 
    the principal diagonal and `1` elsewhere. `I[c] @ O` converts cell 
    types (non-negative `int`) to spins (difference vectors). The sum 
    of spin vector components lies in [-1, 1].
    `nnb` is the number of neighbors in the (regular) lattice within 
    distance `k`.
    """
    A_k = get_knn_adjacency_matrix(AL, k)
    nnb = get_num_neighbors(k)
    
    s_i = I[c] @ O
    m_i = A_k @ s_i / nnb
    
    return 1 - (m_i ** 2).mean(axis=1)


@jit
def knn_alignment_tissue(c, AL, k, I, O):
    """
    Return mean alignment of cell types in a tissue by averaging
    over neighborhoods. This is equivalent to 
    `knn_alignment_per_cell(*args).mean()`
    
    `c` is the cell type vector of shape `(n,)` with dtype `int`
    `A` is the `(n, n)`cell-cell adjacency matrix (can be Boolean)
    `I` is the `(n_ctypes, n_ctypes)` identity matrix, where `n_ctypes` 
    is the number of cell types in the tissue.
    `O` is the `(n_ctypes, n_ctypes - 1)` difference matrix with `-1` on 
    the principal diagonal and `1` elsewhere. `I[c] @ O` converts cell 
    types (non-negative `int`) to spins (difference vectors). The sum 
    of spin vector components lies in [-1, 1].
    `nnb` is the number of neighbors in the (regular) lattice within 
    distance `k`.
    """
    A_k = get_knn_adjacency_matrix(AL, k)
    nnb = get_num_neighbors(k)
    
    s_i = I[c] @ O
    m_i = A_k @ s_i / nnb
    
    return 1 - (m_i ** 2).mean()
    

#### Graph

def adjacency_matrix_from_adjacency_list(AL, dtype=bool):
    """
    Returns adjacency matrix for a nnb-regular graph given the adjacency list.
    """
    n, nnb = AL.shape
    A = jnp.zeros((n, n), dtype=dtype)
    return A.at[jnp.repeat(jnp.arange(n), nnb), AL.flatten()].set(1)


def get_adjacency_matrix_periodic(rows, cols=0):
    """Construct adjacency matrix for a periodic hexagonal 
    lattice of dimensions rows x cols."""
    
    AL = get_adjacency_list_periodic(rows, cols, **kwargs) 
    return adjacency_matrix_from_adjacency_list(AL)


def get_adjacency_list_periodic(rows, cols=0):
    """Construct adjacency matrix for a periodic hexagonal 
    lattice of dimensions rows x cols."""
    
    # Assume square if not specified
    if cols == 0:
        cols = rows
    
    n = rows * cols
    row, col = np.meshgrid(np.arange(rows), np.arange(cols))
    row = row.flatten()
    col = col.flatten()
    
    # Get row of adjacent cells
    dr     = np.array([0, 1, 1, 0, -1, -1]) 
    AL_row = np.add.outer(row, dr) % rows

    # Get column of adjacent cells, accounting for staggering
    dc1    = np.array([1, 0, -1, -1, -1, 0])
    dc2    = np.array([1, 1,  0, -1,  0, 1])
    AL_col = np.add.outer(col, dc1)
    AL_col[1::2] += dc2 - dc1
    AL_col = AL_col % cols
    
    return rows * AL_col + AL_row


def hex_grid(rows, cols=0, r=1., sigma=0, **kwargs):
    """
    Returns XY coordinates of a regular 2D hexagonal grid 
    (rows x cols) with edge length r. Points are optionally 
    passed through a Gaussian filter with std. dev. = sigma * r.
    """
    print("Deprecated: please use `cx.geom.hex_grid")

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
    print("Deprecated: please use `cx.geom.get_outer_idx")
    return np.array([
        rows * c + r
        for c in range(cols)
        for r in range(rows)
        if ((r in (0, rows - 1)) or (c in (0, cols - 1)))
    ])
