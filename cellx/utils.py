import numpy as np

import jax.numpy as jnp
from jax import jit, vmap


# Vectorized rounding
vround = np.vectorize(round)


@jit
def normalize(x, xmin, xmax):
    """Normalize with explicit min/max values. """
    return (x - xmin) / (xmax - xmin)


@jit
def norm_clip(x, xmin, xmax):
    """Normalize with explicit min/max. Clips values to range. """
    xc = jnp.clip(x, xmin, xmax) 
    return (xc - xmin) / (xmax - xmin)


@jit
def exponential_decay(t, v0, r):
    """Exponential decay from value `v0` at time 0 with rate `r`"""
    return v0 * jnp.exp(-r * t)
map_exponential = vmap(exponential_decay, (0, None, None))


@jit
def linear_decay(t, v0, r):
    """Linear decay from value `v0` at time 0 with rate `r`"""
    return v0 - r * t
map_linear = vmap(linear_decay, (0, None, None))

