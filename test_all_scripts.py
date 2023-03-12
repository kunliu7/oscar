"""Naively test all `cs_*.py` scripts in the repo.

This is a naive test that just runs all the scripts in the repo.
Besides that, output should be checked manually.

scripts = [
    "cs_comp_miti.py",
    "cs_distributed.py",
    "cs_evaluate.py",
    "cs_high_dim_vary_2d.py",
    "cs_measure_sparsity.py",
    "cs_opt_on_recon_landscapes.py",
    "cs_second_optimize.py",
]

"""

import os

def check(cmd: str):
    try:
        ret = os.system(cmd)
    except Exception as e:
        print(e)

def test_comp_miti():
    cmd = "python cs_comp_miti.py"
    check(cmd)

def test_distributed():
    cmd = "python cs_distributed.py --ns 6 -p 1 --sf 0.2 --seed 1 \
    --method1 ibm_perth --noise1 real \
    --method2 ibm_perth --noise2 ideal_sim \
    --normalize linear --norm_frac 0.2 \
    --error NRMSE --ratios 0.5"
    check(cmd)

def test_evaluate():
    cmd = "python cs_evaluate.py --aim final -p 2 --ns 12 --ansatz qaoa --problem maxcut --noise depolar-0.003-0.007 --n_seeds 1 --error NRMSE"
    check(cmd)

def test_high_dim_vary_2d():
    cmd = "python cs_high_dim_vary_2d.py --n 6 --p 3 --ansatz qaoa --problem maxcut --noise ideal --seed 0 --error NRMSE --repeat 1 --bs 14 --gs 14"
    check(cmd)

def test_measure_sparsity():
    pass

def test_opt_on_recon_landscapes():
    cmd = "python cs_opt_on_recon_landscapes.py -n 16 -p 1 --seed 0 --noise ideal --opt ADAM --maxiter 10000 --init_pt 0.1 -0.1"
    check(cmd)

def test_second_optimize():
    cmd = "python cs_second_optimize.py -n 16 --noise ideal --seed_range 1 --opt COBYLA"
    check(cmd)