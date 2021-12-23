import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf


sns.set(context="paper", style="white")

data_dir = os.path.abspath("./data")
data_fname = os.path.join(data_dir, "cellgraph_embedding.csv")

save_dir = os.path.abspath("./plots")
save     = True
fmt      = "png"
dpi      = 300

pal = [
    "#9e0142",
    "#d8434e",
    "#f67a49",
    "#fdbf6f",
    "#feeda1",
    "#f1f9a9",
    "#bfe5a0",
    "#74c7a5",
    "#378ebb",
    "#5e4fa2",
]
#color_key = {str(d): c for d, c in enumerate(pal)}
color_key = pal
#color_key = plt.get_cmap("Set1").colors

print(f"Reading data from {data_fname}")
df = pd.read_csv(data_fname)
df["n_components"] = np.random.randint(1, 5, size=df.shape[0])
df["n_components"] = df["n_components"].astype("category")


print("Plotting with Datashader")
cvs = ds.Canvas(plot_width=400, plot_height=400)
agg = cvs.points(df, "umap_x", "umap_y", ds.count_cat("n_components"))
img = tf.shade(agg, color_key=color_key, how="linear")

if save:
    
    # Save UMAP as image
    imname = "cellgraph_embedding_image"
    impath = os.path.join(save_dir, imname)
    utils.export_image(img, filename=impath, background="white")
    
    # Make plot from image
    fname = "cellgraph_embedding_plot"
    fpath = os.path.join(save_dir, fname + "." + fmt)
    image = plt.imread(impath + ".png")
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(image)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("UMAP of tissue topologies ($5\mathrm{x}5$)", fontsize=12)

    print(f"Saving figure to {fpath}")
    plt.savefig(fpath, format=fmt, dpi=dpi)

