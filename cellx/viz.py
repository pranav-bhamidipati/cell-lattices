#import utils as ut
#import geom

import numpy as np
import numba

import matplotlib.pyplot as plt
import colorcet as cc

### Color utils
def rgb_as_int(rgb):
    """Coerce RGB iterable to a tuple of integers"""
    if any([v >= 1. for v in rgb]):
        _rgb = tuple(rgb)
    else:
        _rgb = tuple((round(255 * c) for c in rgb))
    
    return _rgb

    
def rgb_as_float(rgb):
    """Coerce RGB iterable to an ndarray of floats"""
    if any([v >= 1. for v in rgb]) or any([type(v) is int for v in rgb]):
        _rgb = (np.asarray(rgb) / 255).astype(float)
    else:
        _rgb = np.asarray(rgb).astype(float)
    
    return _rgb


def sample_cycle(cycle, size): 
    """Sample a continuous colormap at regular intervals to get a linearly segmented map"""
    return hv.Cycle(
        [cycle[i] for i in ceiling(np.linspace(0, len(cycle) - 1, size))]
    )

def hex2rgb(h):
    """Convert 6-digit hex code to RGB values (0, 255)"""
    h = h.lstrip('#')
    return tuple(int(h[(2*i):(2*(i + 1))], base=16) for i in range(3))

def rgb2hex(rgb):
    """Converts rgb colors to hex"""
    
    RGB = np.zeros((3,), dtype=np.uint8)
    for i, _c in enumerate(rgb):
        
        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.:
            c = int(_c * 255)
        else:
            c = _c
        
        # Calculate new values
        RGB[i] = round(c)
    
    return "#{:02x}{:02x}{:02x}".format(*RGB)

def rgba2hex(rgba, background=(255, 255, 255)):
    """
    Adapted from StackOverflow
    ------------------
    
    Question: Convert RGBA to RGB in Python
    Link: https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python/50332356
    Asked: May 14 '18 at 13:25
    Answered: Nov 7 '19 at 12:40
    User: Feng Wang
    """
    
    rgb = np.zeros((3,), dtype=np.uint8)
    *_rgb, a = rgba

    for i, _c in enumerate(_rgb):
        
        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.:
            c = int(_c * 255)
        else:
            c = _c
        
        # Calculate new values
        rgb[i] = round(a * c + (1 - a) * background[i])
    
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def hexa2hex(h, alpha, background="#ffffff"):
    """
    Returns hex code of the color observed when displaying 
    color `h` (in hex code) with transparency `alpha` on a 
    background color `background` (default white)
    """
    
    # Convert background to RGB
    bg = hex2rgb(background)
    
    # Get color in RGBA
    rgba = *hex2rgb(h), alpha
    
    # Convert to HEX without transparency
    return rgba2hex(rgba, bg)



# Vectorized versions 
vhexa2hex = np.vectorize(hexa2hex)
vrgb2hex  = np.vectorize(rgb2hex)
vhex2rgb  = np.vectorize(hex2rgb)
vrgba2hex = np.vectorize(rgba2hex)

### Plotting
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
    colors = np.asarray(_cmap(ut.normalize(var, vmin, vmax)))
    
    # Get polygon size. Optionally increase size  
    #  so there's no visual gaps between cells
    _r = (r + poly_padding)
    
    # Plot cells as polygons
    for i, (x, y) in enumerate(X):
        
        ax.fill(
            _r * geom._hex_x + x, 
            _r * geom._hex_y + y, 
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


### Animation

_ffmpeg_error_message = """The `ffmpeg` writer must be installed inside the runtime environment. Writer availability can be checked in the current enviornment by executing `matplotlib.animation.writers.list()` in Python. Install location can be checked by running `which ffmpeg` on a command line/terminal."""


### Uncomment these lines and use matplotlib-scalebar package for scalebars
#    if scalebar:
#        _scalebar = ScaleBar(**sbar_kwargs)
#        ax.add_artist(_scalebar)

