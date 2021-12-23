#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd

import umap


# Options for reading inputs
data_dir = os.path.realpath("/home/pbhamidi/git/cell-lattices/data")
enum_data_fpath = os.path.join(data_dir, "cellgraph_enumeration.csv")

# Options for saving output(s)
save     = True
save_dir = os.path.realpath("/home/pbhamidi/git/cell-lattices/data")
# fmt      = "png"
# dpi      = 300

# Read in enumerated cell graph data
print("Reading in data")
df = pd.read_csv(enum_data_fpath)
print("Converting combinations to Boolean array")
combs_bool = np.asarray(
    [[bool(int(char)) for char in s] for s in df["combination"]],
    dtype=bool,
)
n_comp     = df["n_components"].values.astype(int)

print("Performing UMAP")

# Perform UMAP with progress
reducer   = umap.UMAP(metric="hamming", verbose=True)
embedding = reducer.fit_transform(combs_bool)

print("Combining data and saving")

# Combine into dataframe
df["umap_x"] = embedding[:, 0]
df["umap_y"] = embedding[:, 1]

if save:
    data_fname = "cellgraph_embedding.csv"
    data_fpath = os.path.join(save_dir, data_fname)
    df.to_csv(data_fpath, index=False)
    print(f"Saved to {data_fpath}")



