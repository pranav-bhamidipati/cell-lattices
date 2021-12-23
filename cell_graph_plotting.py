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

print(f"Reading data from {data_fname}")
df = pd.read_csv(data_fname)
df["n_components"] = df["n_components"].astype("category")

print("Plotting with Datashader")
cvs = ds.Canvas(plot_width=400, plot_height=400)
agg = cvs.points(df, "umap_x", "umap_y", ds.count_cat("n_components"))
img = tf.shade(agg, color_key=color_key, how="eq_hist")

if save:
    fname = "cellgraph_embedding"
    fpath = os.path.join(save_dir, fname + "." + fmt)
    utils.export_image(img, filename=fpath, background="black")
    
    print(f"Saving image to {fpath}")


