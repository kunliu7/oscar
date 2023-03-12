# OSCAR: cOmpressed Sensing based Cost lAndscape Reconstruction

Reconstructing landscapes of variational quantum algorithms (VQAs)
by compressed sensing.

TODO: link our paper.

Use cases and their visualization in our paper
are generated using `cs_*.py` and `vis_*.ipynb`.

Commands that calling `cs_*.py` to generate those use cases
are recorded in [record.md](record.md).

## TODO for release

- [ ] whether to pack up Google and IBM data, and tutorial to unzip
- [ ] pack up `figs/grid_search` and `figs/optimization` by linking Tianyi's repo
- [ ] zip `figs/grid_search_recon` (too big) and put somewhere; add tutorial to unzip
- [ ] sparsity data (Table IV)
- [ ] data for Fig. 12, n=20
- [ ] LICENSE


## Installation

Recommend: create an Anaconda environment
and install from source.

```bash
conda create -n oscar python=3.9
conda activate oscar
TODO: requirement
```

Download data:
```bash
sh ./download_data.sh
```

```bash
git clone https://github.com/kunliu7/oscar
cd oscar
pip install -e .
pytest
```

P.S. `pytest` might takes several minutes.

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

- cs_comp_miti.py: compare mitigated landscapes
- cs_distributed.py: recon. distributed landscapes
- cs_evaluate.py: compute recon. error for p=1 and p=2
- cs_high_dim_vary_2d.py: compute recon. error for high-dim landscapes
- cs_opt_on_recon_landscapes.py: optimize on recon. landscapes by interpolation
- cs_second_optimize.py: second optimization proposed in paper

- vis_OSCAR_save_queries.py: visualize #Queries saved by OSCAR

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
