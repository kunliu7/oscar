import argparse
import random
import itertools
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pandas as pd
import time
from pathlib import Path
import sys, os
from scipy.spatial.distance import (
    cosine
)
from data_loader import load_grid_search_data, get_recon_landscape
from oscar.compressed_sensing import cal_recon_error

from oscar.vis import(
    vis_landscapes,
)

def recon_high_d_landscapes_by_varying_2d(
    n: int, p: int, ansatz: str, problem: str, 
    bs: int, gs: int,
    noise: str, seed: int, error_type: str,
    repeat: int, force_recon: bool = False,
):
    method = 'sv'
    miti_method = ''
    mses = []
    coss = []

    if bs == None and gs == None:
        if ansatz == 'qaoa':
            if p == 1:
                bs = 50 # beta step
                gs = 2 * bs
            elif p == 2:
                bs = 12
                gs = 15
        elif ansatz == 'twolocal':
            if p == 0:
                bs = 14
            elif p == 1:
                bs = 7
            gs = None
        else:
            raise ValueError(f"ansatz {ansatz} not supported")

    # sfs = np.arange(0.05, 0.25, 0.04)
    # sfs = np.linspace(0.05, 0.25, 5)
    sfs = [0.20]

    print(len(sfs))

    print("noise =", noise)
    print("seed =", seed)
    print("sfs =", sfs)

    cs_seed = n # ! compare horizontally

    data, data_fname, data_dir = load_grid_search_data(
        n_qubits=n, p=p, ansatz=ansatz, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
    )

    plot_range = data['plot_range']
    full_range = data['full_range']

    is_vis = False

    # 和Tianyi代码使用相同目录结构
    recon_dir = f"figs/grid_search_recon/{ansatz}/{problem}/{method}-{noise}-p={p}/seed={seed}"

    if ansatz == 'qaoa':
        d = 2 * p
    elif ansatz == 'twolocal':
        d = n * (p + 1)
    elif ansatz == 'uccsd':
        d = p # ! check
    else:
        raise ValueError(f"ansatz {ansatz} not supported")
        
    print("dimension: ", d)
    rng = default_rng(seed=42)

    recon_d = 2

    fixed_dims = rng.choice(d, size=d-recon_d, replace=False)
    print("fixed dims =", fixed_dims)

    for sf in sfs:
        for i in range(repeat): 
            # random.choice(range(len(full_range['beta'])))

            recon_fname = f"recon-{cs_seed=}-repeat_i={i}-sf={sf:.3f}-{data_fname}"
            recon_path = f"{recon_dir}/{recon_fname}"

            origin = data['data']

            random_id_on_each_dim_list = rng.choice(a=len(full_range['beta']), size=d, replace=True)
            tmp: np.ndarray = origin.copy()
            next_axis = 0
            for j in range(d):
                # print(tmp.shape)
                if j in fixed_dims:
                    tmp = tmp.take(indices=random_id_on_each_dim_list[j], axis=next_axis)
                else:
                    next_axis += 1

            origin = tmp
            print("after np.take = ", origin.shape)
            # if problem == 'partition':
            #     origin = origin / (origin.max() - origin.min())

            recon = get_recon_landscape(ansatz, p, origin, sf, force_recon, 
                recon_path, cs_seed)

            mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), error_type)
            cos = cosine(origin.reshape(-1), recon.reshape(-1))
            mses.append(mse)
            coss.append(cos)

            # ncc = cal_recon_error()
            print("RMSE: ", mse)
            print("Cosine: ", cos)
            
            base_recon_fname = os.path.splitext(recon_fname)[0]
            if is_vis:
                vis_landscapes(
                    landscapes=[origin, recon],
                    labels=["origin", "recon"],
                    full_range={
                        "beta": plot_range['beta'],
                        "gamma": plot_range['gamma'] 
                    },
                    true_optima=None,
                    title="Origin and recon",
                    save_path=f'{recon_dir}/vis/vis-{base_recon_fname}.png',
                    params_paths=[None, None]
                )
            time.sleep(1)

    print("noise =", noise)
    print("n qubits =", n)
    print("seed =", seed)
    print("sfs =", sfs)

    print("mse =", mses)
    print("cos =", coss)
    mses = np.array(mses).reshape(len(sfs), repeat)
    coss = np.array(coss).reshape(len(sfs), repeat)

    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    # timestamp = get_curr_formatted_timestamp()
    recon_error_save_dir = f"{recon_dir}/recon_error-{cs_seed=}-{repeat=}-sfs={sfs}-{data_fname}"
    print(f"recon error data save to {recon_error_save_dir}")
    np.savez_compressed(
        recon_error_save_dir,
        mses=mses,
        coss=coss,
        n_qubits=n,
        seed=seed,
        sfs=sfs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--aim', type=str, help="Your aims, vis, opt", required=True)
    # parser.add_argument('--ns', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('--n', type=int, help="#Qubits.", required=True)
    parser.add_argument('--p', type=int, help="Circuit depth.", required=True)
    parser.add_argument('--repeat', type=int, help="# of random samples.", required=True)
    parser.add_argument('--ansatz', type=str, help="Type of ansatz.", required=True)
    parser.add_argument('--noise', type=str, help="Type of noise.", required=True)
    parser.add_argument('--problem', type=str, help="Type of problem.", required=True)
    parser.add_argument('--seed', type=int, help="Seed of instance.", required=True)
    parser.add_argument('--error', type=str, help="Type of error.", required=True)
    parser.add_argument("--bs", help="Beta steps.", type=int, default=None)
    parser.add_argument("--gs", help="Gamma steps.", type=int, default=None)
    parser.add_argument("--force_recon", help="Force reconstruction and cover existing landscapes.", action="store_true")

    # parser.add_argument('--ansatz', type=str, help="Your aims, vis, opt", required=True)
    args = parser.parse_args()
    
    recon_high_d_landscapes_by_varying_2d(n=args.n, p=args.p, ansatz=args.ansatz, problem=args.problem,
            bs=args.bs, gs=args.gs,
            noise=args.noise, seed=args.seed, error_type=args.error, repeat=args.repeat,
            force_recon=args.force_recon)