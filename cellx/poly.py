import numpy as np
import numpy.polynomial as poly
import scipy.linalg as spla

import jax.numpy as jnp
from jax import lax, vmap, jit, random


def get_boundary_matrix(k):
    """
    Returns a `(k + 2, k)` matrix `B` that multiplies coefficients of
    a polynomial of specified order `k` by `v0 + 4t * (1 - t)` in standard basis.

    In other words, if `u` is the coefficients of a polynomial `f(t)`
    in standard basis, `v` is the coefficients of `g(t) = 4t * (1 - t) * f(t).

    """
    Br1 = jnp.zeros(k)
    Bc1 = jnp.zeros(k + 2)
    Bc1 = Bc1.at[(1, 2),].set([4, -4])

    return jnp.asarray(spla.toeplitz(Bc1, Br1))


def basis_matrix(k, from_basis=poly.Chebyshev, to_basis=poly.Polynomial):
    """
    Returns `(k, k)` matrix `M` that applies a change of polynomial bases such that

        ``v = np.dot(M, u)``

    converts `u` from `from_basis` to `to_basis`.
    Both `from_basis` and `to_basis` must be Numpy polynomial classes.
    """
    M = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        p = from_basis.basis(i).convert(kind=to_basis).coef
        M[:(i + 1), i] = p

    return jnp.asarray(M, dtype=jnp.float32)


@jit
def polyeval(Psi, theta, t):
    """
    Evaluate polynomial at times `t` after change of basis performed by `Psi`
    """
    return jnp.polyval(Psi @ theta, t)
polyeval_batch = vmap(polyeval, (None, 1, None))