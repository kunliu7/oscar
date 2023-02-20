
# ============== Section 4, Recon error

![vis_figs_section4_recon_error.ipynb](vis_figs_section4_recon_error.ipynb)
![cs_evaluate.py](cs_evaluate.py)

## Maxcut


### p=2 noisy

#### n12
python cs_evaluate.py --aim final -p 2 --ns 12 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 12 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 8 --error NRMSE

#### n16
python cs_evaluate.py --aim final -p 2 --ns 16 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 16 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 8 --error NRMSE

#### n20

python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 0 1 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 1 2 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 2 3 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 3 4 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 5 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 5 6 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 6 7 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --problem maxcut --noise depolar-0.003-0.007 --n_seeds 7 8 --error NRMSE


## Partition
python cs_evaluate.py --aim final -p 2 --ns 12 16 20 24 --problem partition --noise ideal --n_seeds 1 --error NRMSE


## SK model
python cs_evaluate.py --aim final -p 2 --ns 12 16 20 24 --problem skmodel --noise ideal --n_seeds 8 --error NRMSE


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
python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 --normalize linear --norm_frac 0.1 --error NRMSE

**For visualize**

baseline:
python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 \
    --error NRMSE \
    --recon_dir "figs/recon_distributed_landscape/2022-11-13_16:48:42"

Geometric mean:
python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 \
    --normalize geo --norm_frac 0.1 --error NRMSE \
    --recon_dir "figs/recon_distributed_landscape/2022-11-13_16:49:14"

Linear regression
python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 \
    --normalize linear --norm_frac 0.1 --error NRMSE \
    --recon_dir "figs/recon_distributed_landscape/2022-11-13_16:48:56"


```python
if method == 'baseline':
    path = "figs/recon_distributed_landscape/2022-11-13_16:48:42"
elif method == 'geo':
    path = "figs/recon_distributed_landscape/2022-11-13_16:49:14"
elif method == 'linear':
    path = "figs/recon_distributed_landscape/2022-11-13_16:48:56"
```

## IBM machine

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --noise1 ibm --noise2 ibm \
    --normalize linear --norm_frac 0.1 --error NRMSE

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --noise1 ibm --noise2 ibm \
    --norm_frac 0.1 --error NRMSE

---

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --noise1 ibm --noise2 ibm \
    --normalize linear --norm_frac 0.2 --error NRMSE

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --noise1 ibm --noise2 ibm --error NRMSE

### Enhance simulation of real quantum devices

"""
For Qaoa P=1, we have two landscapes —(1) generated with noisy simulation (2) captured on ibm hardware

Let’s use (2) as a reference, we want to understand if we can get close to (2) by using fraction of samples from (1) by using NCM

We can use NCM for using cheap/noise hardware to imitate more reliable hardware and vice versa. Similarly we can use simulations with simple noise models to imitate a hardware that is not easy to access, and exhibit complex noise
"""

#### With ideal simulation

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 --noise1 ibm-1 \
    --noise2 figs/grid_search/ibm/maxcut/sv-ideal-p=1/maxcut-sv-ideal-n=6-p=1-seed=1-50-100-IBM1-transpiled-H.npz \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE \
    --recon_dir "figs/recon_distributed_landscape/2023-02-19_15:48:23"

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 --noise1 ibm-1 \
    --noise2 figs/grid_search/ibm/maxcut/sv-ideal-p=1/maxcut-sv-ideal-n=6-p=1-seed=1-50-100-IBM1-transpiled-H.npz \
    --norm_frac 0.2 \
    --error NRMSE \
    --recon_dir "figs/recon_distributed_landscape/2023-02-19_16:08:18"

#### With device noise simulation

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 --noise1 ibm-1 \
    --noise2 figs/grid_search/ibm/maxcut/sv-ibm_perth-p=1/maxcut-sv-ibm_perth-n=6-p=1-seed=1-50-100-IBM1-transpiled-H.npz \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE \
    --recon_dir figs/recon_distributed_landscape/2023-02-19_19:10:48

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 --noise1 ibm-1 \
    --noise2 figs/grid_search/ibm/maxcut/sv-ibm_perth-p=1/maxcut-sv-ibm_perth-n=6-p=1-seed=1-50-100-IBM1-transpiled-H.npz \
    --norm_frac 0.2 \
    --error NRMSE \
    --recon_dir figs/recon_distributed_landscape/2023-02-19_19:12:18

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


### BP initialization
python cs_eval_init_points.py -n 16 -p 2 --noise ideal --eps 0.5 --check --inst_seed 0 --stride 11
python cs_eval_init_points.py -n 20 -p 2 --noise ideal --eps 0.6 --check --inst_seed 0 --stride 10

python cs_eval_init_points.py -n 16 -p 2 --noise ideal --eps 0.3 --check --inst_seed 1 --random_seed 42 --stride 10
python cs_eval_init_points.py -n 20 -p 2 --noise ideal --eps 0.6 --check --inst_seed 1 --random_seed 42 --stride 10


# ============== Use case, error mitigation ==============

![cs_comp_miti.py](cs_comp_miti.py)
![vis_figs_comp_miti.ipynb](vis_figs_comp_miti.ipynb)



# ============== optimize on recon. landscape =============

## for single path
python cs_opt_on_recon_ls.py -n 16 -p 1 --seed 0 --noise ideal --opt ADAM --lr 0.001 --maxiter 10000 --init_pt 0.1 -0.1


## for a batch
### n16
python cs_opt_on_recon_ls.py -n 16 --noise ideal --seed_range 8 --opts ADAM COBYLA SPSA
python cs_opt_on_recon_ls.py -n 16 --noise depolar-0.001-0.02 --seed_range 8 --opts ADAM COBYLA SPSA


python cs_opt_on_recon_ls.py -n 20 --noise ideal --seed_range 8 --opts ADAM COBYLA SPSA
python cs_opt_on_recon_ls.py -n 20 --noise depolar-0.001-0.02 --seed_range 8 --opts ADAM COBYLA SPSA


# =================== debug barren plateaus ===================

python cs_debug_bp.py -n 16 -p 1 --seed 0 --noise ideal --opt ADAM --lr 0.001 --maxiter 10000 --init_pt 0.1 -0.1


# ========== second optimization =========



# ============= two local problem ============

python cs_evaluate.py --aim final -p 0 --ns 6 --ansatz twolocal --problem maxcut --noise ideal --n_seeds 1 --error NRMSE

# ============= fix k-2 for twolocal ==============

The table:

|      Problem           | QAOA Ansatz| VQE Ansatz | 
|------------------------|------------|------------|
| 3-reg Maxcut (n=6)     | 0.969      | 0.753      | 
| 3-reg Maxcut (n=4)     | 0.968      | 0.734      | 
| SK Problem (n=4)       | 0.968      | 0.738      | 
| SK Problem (n=6)       | 0.969      | 0.740      | 
| Partition Problem (n=6)| 0.981      | 0.752      | 
| Partition Problem (n=4)| 0.985      | 0.754      |

## maxcut

python cs_high_dim_vary_2d.py --p 0 --n 6 --ansatz twolocal --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 100

python cs_high_dim_vary_2d.py --p 1 --n 4 --ansatz twolocal --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 100

## partition

python cs_high_dim_vary_2d.py --p 0 --n 6 --ansatz twolocal --problem partition --noise ideal --seed 0 --error NRMSE --repeat 100

python cs_high_dim_vary_2d.py --p 1 --n 4 --ansatz twolocal --problem partition --noise ideal --seed 0 --error NRMSE --repeat 100

## skmodel

python cs_high_dim_vary_2d.py --p 0 --n 6 --ansatz twolocal --problem skmodel --noise ideal --seed 0 --error NRMSE --repeat 100

python cs_high_dim_vary_2d.py --p 1 --n 4 --ansatz twolocal --problem skmodel --noise ideal --seed 0 --error NRMSE --repeat 100

# ============= recon ibm landscape ==========

python cs_ibm_landscape.py --mid 1 --seed 0 --shots 2048


# ============= measure sparsity of all the data we have ==========

python cs_measure_sparsity.py --p 2 --ns 16 20 24 --ansatz qaoa --problem maxcut --noise ideal --n_seeds 16

python cs_measure_sparsity.py --p 1 --ns 16 20 24 30 --ansatz qaoa --problem maxcut --noise ideal --n_seeds 16

python cs_measure_sparsity.py --p 2 --ns 12 16 20 24 --ansatz qaoa --problem skmodel --noise ideal --n_seeds 16