import numpy as np
import numba


# Vectorized rounding
vround = np.vectorize(round)


@numba.njit
def normalize(x, xmin, xmax):
    """Normalize with explicit min/max values. """
    return (x - xmin) / (xmax - xmin)


@numba.njit
def norm_clip(x, xmin, xmax):
    """Normalize with explicit min/max. Clips values to range. """
    xc = np.clip(x, xmin, xmax) 
    return (xc - xmin) / (xmax - xmin)

