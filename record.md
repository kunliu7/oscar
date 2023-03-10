
# Fig. 4, recon error

[vis_figs_section4_recon_error.ipynb](vis_figs_section4_recon_error.ipynb)
[cs_evaluate.py](cs_evaluate.py)

## Maxcut

### p=2 noisy

#### n12
python cs_evaluate.py --aim final -p 2 --ns 12 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 12 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 8 --error NRMSE

#### n16
python cs_evaluate.py --aim final -p 2 --ns 16 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 16 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 8 --error NRMSE

#### n20

python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 0 1 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 1 2 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 2 3 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 3 4 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 4 5 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 5 6 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 6 7 --error NRMSE
python cs_evaluate.py --aim final -p 2 --ns 20 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 7 8 --error NRMSE


## Partition
python cs_evaluate.py --aim final -p 2 --ns 12 16 20 24 --ansatz qaoa --problem partition --noise ideal --n_seeds 1 --error NRMSE


## SK model
python cs_evaluate.py --aim final -p 2 --ns 12 16 20 24 --ansatz qaoa --problem skmodel --noise ideal --n_seeds 8 --error NRMSE


# Distributed landscape reconstruction

Visualization / Table: [vis_figs_section5_distributed_oscar.ipynb](vis_figs_section5_distributed_oscar.ipynb)

Script: [cs_distributed.py](cs_distributed.py)

## Fig. 8

baseline

python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 \
    --error NRMSE

Linear regression

python cs_distributed.py --ns 12 16 20 -p 1 --noise1 depolar-0.003-0.007 --noise2 depolar-0.001-0.02 \
    --normalize linear --norm_frac 0.1 --error NRMSE

## Table V

#### IBM Perth & Ideal Simulation

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_perth --noise2 ideal_sim \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_perth --noise2 ideal_sim \
    --error NRMSE

#### IBM Perth & Noisy Simulation (IBM Perth)

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_perth --noise2 noisy_sim \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_perth --noise2 noisy_sim \
    --error NRMSE

#### IBM Perth & IBM Lagos

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_lagos --noise2 real \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE

base

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_lagos --noise2 real \
    --error NRMSE

#### IBM Lagos (IBM-2) & IBM Perth (IBM-1)

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_lagos --noise1 real \
    --method2 ibm_perth --noise2 real \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE

base

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_lagos --noise1 real \
    --method2 ibm_perth --noise2 real \
    --error NRMSE

#### Ideal Simulation & IBM Perth

NCM:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 ideal_sim \
    --method2 ibm_perth --noise2 real \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE

baseline:

python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 ideal_sim \
    --method2 ibm_perth --noise2 real \
    --error NRMSE

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

One to one correspondence. Same # parameters.

For QAOA: # params = 2 * p
For VQE:  # params = (p+1) * n



## MaxCut

### QAOA

python cs_high_dim_vary_2d.py --n 6 --p 3 --ansatz qaoa --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 14 --gs 14

python cs_high_dim_vary_2d.py --n 4 --p 4 --ansatz qaoa --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 7 --gs 7

### twolocal

python cs_high_dim_vary_2d.py --n 6 --p 0 --ansatz twolocal --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 14

python cs_high_dim_vary_2d.py --n 4 --p 1 --ansatz twolocal --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 7

## partition

**Do not do**

### QAOA

python cs_high_dim_vary_2d.py --p 3 --n 6 --ansatz qaoa --problem partition --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 14 --gs 14

python cs_high_dim_vary_2d.py --p 4 --n 4 --ansatz qaoa --problem partition --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 7 --gs 7

### twolocal

python cs_high_dim_vary_2d.py --p 0 --n 6 --ansatz twolocal --problem partition --noise ideal --seed 0 --error NRMSE --repeat 100

python cs_high_dim_vary_2d.py --p 1 --n 4 --ansatz twolocal --problem partition --noise ideal --seed 0 --error NRMSE --repeat 100

## skmodel

#### QAOA

python cs_high_dim_vary_2d.py --n 6 --p 3 --ansatz qaoa --problem skmodel --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 14 --gs 14

python cs_high_dim_vary_2d.py --n 4 --p 4 --ansatz qaoa --problem skmodel --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 7 --gs 7

### twolocal

python cs_high_dim_vary_2d.py --n 6 --p 0 --ansatz twolocal --problem skmodel --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 14

python cs_high_dim_vary_2d.py --n 4 --p 1 --ansatz twolocal --problem skmodel --noise ideal --seed 0 --error NRMSE --repeat 100 --bs 7

# ============= measure sparsity of all the data we have ==========

python cs_measure_sparsity.py --p 2 --ns 16 20 24 --ansatz qaoa --problem maxcut --noise ideal --n_seeds 16

python cs_measure_sparsity.py --p 1 --ns 16 20 24 30 --ansatz qaoa --problem maxcut --noise ideal --n_seeds 16

python cs_measure_sparsity.py --p 2 --ns 12 16 20 24 --ansatz qaoa --problem skmodel --noise ideal --n_seeds 16

# Quantum Chemistry

UCCSD, twolocal, QAOA

H2, LiH

## qaoa

**Do not do**

<!-- python cs_evaluate.py --ansatz qaoa --ns 2 --p 1 --problem h2 --bs 50 --gs 50 \
    --noise ideal --seeds 1 --error NRMSE

python cs_evaluate.py --ansatz qaoa --ns 2 --p 2 --problem h2 --bs 14 --gs 14 \
    --noise ideal --seeds 1 --error NRMSE -->

## twolocal

### H2

python cs_high_dim_vary_2d.py --ansatz twolocal --n 2 --p 1 --problem h2 --bs 14 \
    --noise ideal --seed 0 --error NRMSE --repeat 100 --force_recon

### LiH

python cs_high_dim_vary_2d.py --ansatz twolocal --n 4 --p 1 --problem lih --bs 7 \
    --noise ideal --seed 0 --error NRMSE --repeat 100 --force_recon

## uccsd

### H2

python cs_high_dim_vary_2d.py --ansatz uccsd --n 2 --p 3 --problem h2 --bs 14 \
    --noise ideal --seed 0 --error NRMSE --repeat 100 --force_recon

python cs_high_dim_vary_2d.py --ansatz uccsd --n 2 --p 3 --problem h2 --bs 50 \
    --noise ideal --seed 0 --error NRMSE --repeat 100 --force_recon

### LiH

python cs_high_dim_vary_2d.py --ansatz uccsd --n 4 --p 8 --problem lih --bs 7 \
    --noise ideal --seed 0 --error NRMSE --repeat 100 --force_recon

# Revision table

| **QPU1 (Target)** | **QPU2 (Source)** | **OSCAR (20-80)** | **OSCAR+NCM (20-80)** | **OSCAR (50-50)** | **OSCAR+NCM (50-50)** | **OSCAR (80-20)** | **OSCAR+NCM (80-20)** | **OSCAR (100)** |
|:-----------------:|:-----------------:|:-----------------:|:---------------------:|:-----------------:|:---------------------:|:-----------------:|:---------------------:|:---------------:|
| Noisy Sim-I | Noisy Sim-II|       0.076       |         0.003         |       0.061       |         0.002         |       0.039       |         0.002         |      0.001      |
| Noisy Sim-II| Noisy Sim-I |       0.075       |         0.002         |       0.059       |         0.002         |       0.037       |         0.002         |      0.001      |
|     IBM Perth     |     Ideal Sim     |       1.362       |         0.299         |        0.97       |         0.265         |       0.597       |         0.223         |      0.184      |
|     IBM Perth     |     Noisy Sim (Perth Noise Model)     |       0.767       |         0.272         |       0.564       |         0.247         |       0.379       |         0.213         |      0.184      |
|     IBM Perth     |     IBM Lagos     |        0.5        |         0.424         |       0.419       |          0.36         |       0.284       |         0.262         |      0.184      |
| IBM Lagos | IBM Perth | 0.403 | 0.337 | 0.341| 0.286 | 0.266 | 0.247 | 0.222 |
| Ideal Sim | IBM Perth | 0.478 | 0.226 | 0.363 | 0.18 | 0.215 | 0.109 | 0.042 |



|      Problem           | QAOA Ansatz| TwoLocal Ansatz | UCCSD Ansatz |
|------------------------|------------|------------|-----|
| 3-reg Maxcut (n=4)     | 0.847      | 0.645      |  |
| 3-reg Maxcut (n=6)     | 0.372      | 0.0000001      |  |
| SK Problem (n=4)       | 0.847   | 0.765     |  |
| SK Problem (n=6)       | 0.372      | 0.057      |  |
| H2 (n=2)|       | 0.171     | 0.345    |
| LiH (n=4)|      | 0.678      | 0.856   |
