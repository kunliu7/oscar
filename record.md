
# ============== Section 4, Recon error

![vis_figs_section4_recon_error.ipynb](vis_figs_section4_recon_error.ipynb)
![cs_evaluate.py](cs_evaluate.py)

## Maxcut


## Partition
python cs_evaluate.py --aim final -p 2 --ns 12 16 20 24 --problem partition --noise ideal --n_seeds 1


## SK model
python cs_evaluate.py --aim final -p 2 --ns 12 16 20 24 --problem skmodel --noise ideal --n_seeds 1


# ============= Section 5, distributed landscape

![vis_figs_section5_distributed_oscar.ipynb](vis_figs_section5_distributed_oscar.ipynb)
![cs_distributed.py](cs_distributed.py)

## noise-1 and noise-2, n=12,16,20

Done

## noise-2 and noise-3, n=12,16,20

### not normalized
python cs_distributed.py -n 12 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02
python cs_distributed.py -n 16 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02
python cs_distributed.py -n 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02

### normalized by geometric mean
python cs_distributed.py -n 12 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 --normalize geo --norm_frac 0.1
python cs_distributed.py -n 16 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02
python cs_distributed.py -n 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02

### compare baseline, normalized by geometric mean, normalized by linear regression

python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02
python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 --normalize geo --norm_frac 0.1
python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 --normalize linear --norm_frac 0.1

# ============== Use case, debug barren plateaus =========

![vis_figs_bp.ipynb](vis_figs_bp.ipynb)
![cs_eval_barren_plateaus.py](cs_eval_barren_plateaus.py)

<!-- ![vis_figs_init_points.ipynb](vis_figs_init_points.ipynb) -->


# ============== Use case, initialization ================

![vis_figs_init_points.ipynb](vis_figs_init_points.ipynb)
![cs_eval_init_points.py](cs_eval_init_points.py)

## QAOAKit has optimal
### P = 2, n=16, true optima, ideal,


### P = 2, n=16, true optima, noisy-3 (ALREADY HAD DATA)

Violin plot

—----

## Randomized initialization:
### p=2, n = 20, ideal (ALREADY HAD DATA)
python cs_eval_init_points.py -n 20 --noise ideal
figs/find_init_pts_by_recon/2022-11-10_00:51:15_OK/recon-eps=0.600-csSeed=20-sv-ideal-n=20-p=2-seed=0-12-15.npz

### p=2, n = 20, noisy-3 (ALREADY HAD DATA)
python cs_eval_init_points.py -n 20 --noise depolar --p1 0.001 --p2 0.02

# ============== Use case, error mitigation ==============

![cs_comp_miti.py](cs_comp_miti.py)
![vis_figs_comp_miti.ipynb](vis_figs_comp_miti.ipynb)