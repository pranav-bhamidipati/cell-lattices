# cell-lattices

To get started:
1. clone/fork this repo
2. Construct virtual environment with `conda`
3. Make directories/symlinks for untracked directories (`data`, `plots`)

## Construct environment with `conda`
Move into the repo directory and use the `environment.yml` file to construct a custom virtual environment. You can store this environment along with your other environments (e.g. `base`) by running `conda env create` with the `--name` option

```
conda env create --file ./environment.yml --name cell-lattices
```

or you can store the environment in a user-defined location by providing the `--prefix` option. For example, you can make a folder `env` locally in the repo and store the environment there.

```
conda env create --file ./environment.yml --prefix ./env
```

This can take a while (sometimes up to 10-20 mins) to solve the environment. Follow the prompts to finish creation. If it's taking a super long time, this can indicate server issues with Anaconda - this usually resolves if you try again later.

## Activate environment

Unfortunately, `conda` supports either a name or a prefix but not both. So depending on which you did above, run

```
conda deactivate && conda activate cell-lattices
```

or

```
conda deactivate && conda activate ./env
```

I prefer the latter, so I can debug more easily, the directory path is filesystem-invariant, and I can move the environment around without root permission.

## Make untracked directories/symlinks

This module reads and writes to a data directory `./data` and writes graphical outputs to `./plots`. These can be large, possibly binary files that are expensive/inefficient to track with Git, so we ignore them. You can make these folders locally in the repo.

```
mkdir ./data ./plots
```

Alternatively, you can store these files elsewhere on your filesystem and access them via symbolic links.

```
ln -s /path/to/data ./data
ln -s /path/to/plots ./plots
```

Remember that these folders are ignored by Git, so the user is responsible for keeping track of their contents.

## Optional: Set up a kernel in Jupyter

If you work with Jupyter notebooks or JupyterLab, you can easily construct a custom kernel for this project that will come pre-loaded with the virtual environment. Similar to `conda`, you can either store the kernel in Jupyter's default location (something like `/home/USER/.local/share`) or in a local environment folder if you made one (e.g. `/path/to/cell-lattices/env`). If you chose the first option, run:

```
conda deactivate && conda activate ./env
jupyter kernelspec list    # look at current kernels
python -m ipykernel install --name cell_lattices   # Install kernel
jupyter kernelspec list    # check if it got added
```

And then the kernel should show up in the list of kernels in a Jupyter application. If you get a `Permission denied` error on the `ipykernel install` line, try installing with root permissions.

```
sudo $(which python) -m ipykernel install --name cell_lattices
```

If you store your environments locally, run:

```
conda deactivate && conda activate ./env
jupyter kernelspec list    # look at current kernels
python -m ipykernel install --name cell_lattices --prefix /path/to/cell-lattices/env
jupyter kernelspec list    # check if it got added
```
