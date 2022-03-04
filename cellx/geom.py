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



