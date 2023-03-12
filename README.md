# OSCAR: cOmpressed Sensing based Cost lAndscape Reconstruction

Reconstructing landscapes of variational quantum algorithms (VQAs)
by compressed sensing.

TODO: link our paper.

Use cases and their visualization in our paper
are generated using `cs_*.py` and `vis_*.ipynb`.

Commands that calling `cs_*.py` to generate those use cases
are recorded in [record.md](record.md).


## Installation

Recommend: create an Anaconda environment
and install from source.

```bash
conda create -n oscar python=3.9
conda activate oscar
TODO: requirement
```

```bash
git clone TODO
cd TODO
pip install -e .
```

<!-- If you still fail, here are some information that might help.

`mitiq` does not compatible well with latest Python, NumPy and Qiskit.
They are still upgrading `mitiq` according to this [issue](https://github.com/unitaryfund/mitiq/issues/1385).


For Python==3.9,
```
conda install numpy==1.20.3
pip install qiskit==0.36.2
```

Install NumPy by pip does not work on Mac M1. -->

## Data

TODO: link with `QAOA-Simulator`.

## Examples

TODO: explanation of all the `cs_*.py`.


## Citation


```
```

<!-- #### Contributing

You should set up the linter to run before every commit.
```
pip install pre-commit
pre-commit install
```
Note that linter checks passing is a necessary condition for your contribution to be reviewed.

We are in the process of moving the codebase to numpy-style docstrings. See documentation here: https://numpydoc.readthedocs.io/en/latest/format.html -->
