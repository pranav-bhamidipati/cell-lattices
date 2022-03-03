import numpy as np
import numba

import matplotlib.pyplot as plt
import colorcet as cc


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

### Uncomment these lines and use matplotlib-scalebar package for scalebars
#    if scalebar:
#        _scalebar = ScaleBar(**sbar_kwargs)
#        ax.add_artist(_scalebar)

