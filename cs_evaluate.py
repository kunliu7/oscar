import argparse
import itertools
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
import timeit
import orqviz
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.algorithms.optimizers import (
    ADAM,
    AQGD,
    CG,
    COBYLA,
    L_BFGS_B,
    GSLS,
    GradientDescent,
    NELDER_MEAD,
    NFT,
    P_BFGS,
    POWELL,
    SLSQP,
    SPSA,
    QNSPSA,
    TNC,
    SciPyOptimizer
)
from qiskit.algorithms import VQE, NumPyMinimumEigensolver, QAOA
from qiskit.utils import QuantumInstance, algorithm_globals
# from qiskit_optimization.algorithms import (
#     MinimumEigenOptimizer,
#     RecursiveMinimumEigenOptimizer,
#     SolutionSample,
#     OptimizationResultStatus,
#     WarmStartQAOAOptimizer
# )
from qiskit import Aer
import qiskit
from qiskit.providers.aer import AerSimulator
from functools import partial
from pathlib import Path
import copy
from itertools import groupby
import timeit
import sys, os
from scipy.optimize import minimize
from scipy.spatial.distance import (
    cosine
)
from qiskit.quantum_info import Statevector
from qiskit.opflow import PrimitiveOp, PauliSumOp
from QAOAKit.n_dim_cs import recon_4D_landscape, recon_4D_landscape_by_2D
# from cs_comp_miti import _get_recon_landscape
from data_loader import load_grid_search_data, get_recon_landscape
# from QAOAKit import vis

sys.path.append('..')
from QAOAKit.noisy_params_optim import (
    get_pauli_error_noise_model,
    optimize_under_noise,
    get_depolarizing_error_noise_model,
    compute_expectation
)

from QAOAKit.vis import(
    _vis_recon_distributed_landscape,
    vis_landscape,
    vis_landscape_heatmap,
    vis_landscape_heatmap_multi_p,
    vis_landscape_multi_p,
    vis_landscape_multi_p_and_and_count_optima,
    vis_landscape_multi_p_and_and_count_optima_MP,
    vis_landscapes,
    vis_multi_landscape_and_count_optima_and_mitiq_MP,
    vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable,
    vis_two_BPs_p1_recon
)

from QAOAKit import (
    opt_angles_for_graph,
    get_fixed_angles,
    get_graph_id,
    get_graph_from_id,
    angles_to_qaoa_format,
    beta_to_qaoa_format,
    gamma_to_qaoa_format,
    angles_to_qiskit_format,
    angles_to_qtensor_format,
    get_3_reg_dataset_table,
    get_3_reg_dataset_table_row,
    get_full_qaoa_dataset_table_row,
    get_full_qaoa_dataset_table,
    get_fixed_angle_dataset_table,
    get_fixed_angle_dataset_table_row,
    qaoa_maxcut_energy,
    noisy_qaoa_maxcut_energy,
    angles_from_qiskit_format,

)
from QAOAKit.utils import (
    angles_from_qaoa_format,
    beta_shift_sector,
    gamma_shift_sector,
    get_curr_formatted_timestamp,
    load_partial_qaoa_dataset_table,
    obj_from_statevector,
    precompute_energies,
    maxcut_obj,
    isomorphic,
    load_weights_into_dataframe,
    load_weighted_results_into_dataframe,
    get_adjacency_matrix,
    brute_force,
    get_pynauty_certificate,
    get_full_weighted_qaoa_dataset_table,
    qaoa_format_to_qiskit_format,
    save_partial_qaoa_dataset_table,
    shift_parameters
)

from QAOAKit.classical import thompson_parekh_marwaha
from QAOAKit.qaoa import get_maxcut_qaoa_circuit
from QAOAKit.qiskit_interface import (
    get_maxcut_qaoa_qiskit_circuit,
    goemans_williamson,
    get_maxcut_qaoa_qiskit_circuit_unbinded_parameters
)
from QAOAKit.examples_utils import get_20_node_erdos_renyi_graphs
from QAOAKit.parameter_optimization import get_median_pre_trained_kde
from QAOAKit.compressed_sensing import (
    _executor_of_qaoa_maxcut_energy,
    cal_recon_error,
    gen_p1_landscape,
    gen_p2_landscape,
    recon_2D_landscape,
    two_D_CS_p1_recon_with_distributed_landscapes,
    two_D_CS_p1_recon_with_given_landscapes,
)
from QAOAKit.optimizer_wrapper import (
    wrap_qiskit_optimizer_to_landscape_optimizer,
    get_numerical_derivative
)

from QAOAKit.interpolate import (
    approximate_fun_value_by_2D_interpolation
)

# from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz

from scipy import interpolate

test_utils_folder = Path(__file__).parent


def recon_landscapes_varying_qubits_and_instances(
    p: int, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list, error_type: str
):
    """Section 4 ABCD
    """
    is_recon = False

    method = 'sv'
    miti_method = ''
    mses = []
    coss = []

    if p == 1:
        bs = 50 # beta step
        gs = 2 * bs
    elif p == 2:
        bs = 12
        gs = 15

    sfs = np.arange(0.01, 0.11, 0.02)
    if len(n_seeds) == 1:
        seeds = list(range(n_seeds[0]))
    elif len(n_seeds) == 2:
        seeds = list(range(n_seeds[0], n_seeds[1]))
    # if p == 1 and noise == 'ideal':
    #     n_qubits_list = [16, 20, 24, 30]
    # elif p == 1 and noise == 'depolar-0.003-0.007':
    #     n_qubits_list = [12, 16, 20]
    # elif p == 2 and noise == 'ideal':
    #     n_qubits_list = [16, 20, 24]
    # elif p == 2 and noise == 'ideal':
    #     n_qubits_list = [12, 16, 20]

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)

    for n_qubits in n_qubits_list:
        cs_seed = n_qubits # ! compare horizontally

        for seed in seeds:
            data, data_fname, data_dir = load_grid_search_data(
                n_qubits=n_qubits, p=p, problem=problem, method=method,
                noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
            )

            plot_range = data['plot_range']

            # 和Tianyi代码使用相同目录结构
            recon_dir = f"figs/grid_search_recon/{problem}/{method}-{noise}-p={p}"

            for sf in sfs:
                recon_fname = f"recon-cs_seed={cs_seed}-sf={sf:.3f}-{data_fname}"
                recon_path = f"{recon_dir}/{recon_fname}"

                origin = data['data']
                recon = get_recon_landscape(p, origin, sf, is_recon, 
                    recon_path, cs_seed)
                     
                mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), error_type)
                # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
                cos = cosine(origin.reshape(-1), recon.reshape(-1))
                mses.append(mse)
                coss.append(cos)

                # ncc = cal_recon_error()
                print("RMSE: ", mse)
                print("Cosine: ", cos)
                
                base_recon_fname = os.path.splitext(recon_fname)[0]
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

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)

    print("mse =", mses)
    print("cos =", coss)
    mses = np.array(mses)
    coss = np.array(coss)

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    # timestamp = get_curr_formatted_timestamp()
    recon_error_save_dir = f"{recon_dir}/recon_error_ns={n_qubits_list}-seeds={seeds}-sfs={sfs}-error={error_type}"
    print(f"recon error data save to {recon_error_save_dir}")
    np.savez_compressed(
        recon_error_save_dir,
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs
    )


# ! deprecated
def recon_p1_landscape_noisy_varying_qubits_and_instances():
    is_recon = False

    label = "sv-depolar-0.003-0.007"

    bs = 50 # beta step
    gs = 2 * bs
    data_dir = f"figs/gen_p1_landscape/{label}"
    mses = []
    coss = []

    n_qubits_list = [12, 16, 20]
    seeds = [0, 1]
    sfs = np.arange(0.01, 0.11, 0.02)

    for n_qubits in n_qubits_list:
        for seed in seeds:
            data = np.load(
                f"{data_dir}/{label}-n={n_qubits}-p=1-seed={seed}-{bs}-{2*bs}.npz",
                allow_pickle=True)

            timestamp = get_curr_formatted_timestamp()
            bounds = {
                'beta': data['beta_bound'],
                'gamma': data['gamma_bound']
            }

            origin = data['data']
            print(origin.shape)

            full_range = {
                'beta': np.linspace(- bounds['beta'], bounds['beta'], origin.shape[0]),
                'gamma': np.linspace(- bounds['gamma'], bounds['gamma'], origin.shape[1])
            }

            # print(full_range)

            # origin = {'ideals': landscape}
            # recon_dir = f"figs/recon/{timestamp}"
            fig_dir = f"{data_dir}/2D_CS_recon"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            for sf in sfs:
                if not is_recon:
                    # seed = seed
                    cs_seed = n_qubits
                    np.random.seed(seed=cs_seed)
                    recon = recon_2D_landscape(
                        origin=origin,
                        sampling_frac=sf
                    )

                    np.savez_compressed(f"{fig_dir}/sf{sf:.3f}_p1_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}",
                        recon=recon,
                    )
                else:
                    recon = np.load(f"{fig_dir}/sf{sf:.3f}_p1_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}.npz",
                        allow_pickle=True
                    )['recon']

                    # print(recon.shape)
                    
                mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), "MSE")
                # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
                cos = cosine(origin.reshape(-1), recon.reshape(-1))
                mses.append(mse)
                coss.append(cos)

                # ncc = cal_recon_error()
                print("RMSE: ", mse)
                print("Cosine: ", cos)
                
                vis_landscapes(
                    landscapes=[origin, recon],
                    labels=["origin", "recon"],
                    full_range={
                        "beta": full_range['beta'],
                        "gamma": full_range['gamma'] 
                    },
                    true_optima=None,
                    title="Origin and recon",
                    save_path=f'{fig_dir}/origin_and_2D_recon_sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}.png',
                    params_paths=[None, None]
                )

    print("mse = ", mses)
    print("cos = ", coss)
    mses = np.array(mses)
    coss = np.array(coss)

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    np.savez_compressed(
        f"{data_dir}/recon_error_noisy_p1",
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs
    )


# ! deprecated
def recon_p1_landscape_ideal_varying_qubits_and_instances():
    is_recon = False

    bs = 50 # beta step
    data_dir = "figs/gen_p1_landscape/sv-ideal"
    mses = []
    coss = []

    n_qubits_list = [16, 20, 24, 30]
    seeds = [0, 1, 2]
    sfs = np.arange(0.01, 0.11, 0.02)

    for n_qubits in n_qubits_list:
        for seed in seeds:
            data = np.load(
                f"{data_dir}/sv-ideal-n={n_qubits}-p=1-seed={seed}-{bs}-{2*bs}.npz",
                allow_pickle=True)

            timestamp = get_curr_formatted_timestamp()
            bounds = {
                'beta': data['beta_bound'],
                'gamma': data['gamma_bound']
            }

            origin = data['data']
            print(origin.shape)

            full_range = {
                'beta': np.linspace(- bounds['beta'], bounds['beta'], origin.shape[0]),
                'gamma': np.linspace(- bounds['gamma'], bounds['gamma'], origin.shape[1])
            }

            # print(full_range)

            # origin = {'ideals': landscape}
            # recon_dir = f"figs/recon/{timestamp}"
            fig_dir = f"{data_dir}/2D_CS_recon"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            for sf in sfs:
            # for sf in [0.05]:
                if not is_recon:
                    cs_seed = n_qubits
                    # rng = np.random.default_rng(seed=seed)
                    np.random.seed(seed=cs_seed)
                    recon = recon_2D_landscape(
                        origin=origin,
                        sampling_frac=sf
                    )

                    np.savez_compressed(f"{fig_dir}/sf{sf:.3f}_p1_bs{bs}_nQ{n_qubits}_csSeed{cs_seed}",
                        recon=recon,
                    )
                else:
                    recon = np.load(f"{fig_dir}/sf{sf:.3f}_p1_bs{bs}_nQ{n_qubits}_csSeed{cs_seed}.npz",
                        allow_pickle=True
                    )['recon']

                    # print(recon.shape)
                    
                mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), "MSE")
                # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
                cos = cosine(origin.reshape(-1), recon.reshape(-1))
                mses.append(mse)
                coss.append(cos)

                # ncc = cal_recon_error()
                print("RMSE: ", mse)
                print("Cosine: ", cos)
                
                vis_landscapes(
                    landscapes=[origin, recon],
                    labels=["origin", "recon"],
                    full_range={
                        "beta": full_range['beta'],
                        "gamma": full_range['gamma'] 
                    },
                    true_optima=None,
                    title="Origin and recon",
                    save_path=f'{fig_dir}/origin_and_2D_recon_sf{sf:.3f}_bs{bs}_nQ{n_qubits}.png',
                    params_paths=[None, None]
                )

    print("mse = ", mses)
    print("cos = ", coss)
    mses = np.array(mses)
    coss = np.array(coss)

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    np.savez_compressed(
        f"{data_dir}/recon_error_ideal_p1",
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs
    )


# ! deprecated
def recon_p2_landscape_ideal_varying_qubits_and_instances():
    is_recon = False

    bs = 12 # beta step
    gs = 15
    data_dir = "figs/gen_p1_landscape/sv-ideal"
    mses = []
    coss = []

    # n_qubits_list = [16, 20, 24]
    # seeds = [0, 1]
    # sfs = np.arange(0.01, 0.11, 0.02)
    n_qubits_list = [16, 20, 24]
    seeds = [0, 1]
    # sfs = np.arange(0.01, 0.1, 0.02)
    sfs = [0.05]
    
    for n_qubits in n_qubits_list:
        for seed in seeds:
            data = np.load(
                f"{data_dir}/sv-ideal-n={n_qubits}-p=2-seed={seed}-{bs}-{gs}.npz",
                allow_pickle=True)

            timestamp = get_curr_formatted_timestamp()
            bounds = {
                'beta': data['beta_bound'],
                'gamma': data['gamma_bound']
            }

            origin = data['data']
            print(origin.shape)

            full_range = {
                'beta': np.linspace(- bounds['beta'], bounds['beta'], origin.shape[0]),
                'gamma': np.linspace(- bounds['gamma'], bounds['gamma'], origin.shape[1])
            }

            # print(full_range)

            # origin = {'ideals': landscape}
            # recon_dir = f"figs/recon/{timestamp}"
            fig_dir = f"{data_dir}/2D_CS_recon_p2"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            for sf in sfs:
                cs_seed = seed
                if not is_recon:
                    cs_seed = n_qubits
                    np.random.seed(seed=cs_seed)
                    recon = recon_4D_landscape_by_2D(
                        origin=origin,
                        sampling_frac=sf
                    )
            
                    np.savez_compressed(f"{fig_dir}/sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}",
                        recon=recon,
                        seed=seed
                    )
                else:
                    # recon = np.load(f"{fig_dir}/sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}.npz",
                    #     allow_pickle=True
                    # )['recon']
                    
                    recon = np.load(f"{fig_dir}/sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}.npz",
                        allow_pickle=True
                    )['recon']

                    # print(recon.shape)
                    
                mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), "MSE")
                # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
                cos = cosine(origin.reshape(-1), recon.reshape(-1))
                mses.append(mse)
                coss.append(cos)

                origin_2d = origin.reshape(origin.shape[0] * origin.shape[1],
                    origin.shape[2] * origin.shape[3])
                recon_2d = recon.reshape(recon.shape[0] * recon.shape[1],
                    recon.shape[2] * recon.shape[3])

                # ncc = cal_recon_error()
                print("RMSE: ", mse)
                print("Cosine: ", cos)
                
                vis_landscapes(
                    landscapes=[origin_2d, recon_2d],
                    labels=["origin", "recon"],
                    full_range={
                        "beta": range(origin_2d.shape[0]),
                        "gamma": range(origin_2d.shape[1]) 
                    },
                    true_optima=None,
                    title="Origin and recon",
                    save_path=f'{fig_dir}/origin_and_2D_recon_sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}.png',
                    params_paths=[None, None]
                )

    print("mse = ", mses)
    print("cos = ", coss)
    mses = np.array(mses)
    coss = np.array(coss)

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    np.savez_compressed(
        f"{fig_dir}/recon_error_ideal_p2_{n_qubits_list}_{seeds}_{sfs}",
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs
    )


# ! deprecated
def recon_large_qubits_p1_landscape_top():
    is_recon = False

    beta_steps = range(25, 76, 5)
    data_dir = "figs/gen_p1_landscape/2022-10-18_n16_p1_ideal"
    mses = []
    coss = []
    for bs in beta_steps:
        data = np.load(
            f"{data_dir}/sv-ideal/sv-ideal-n=16-p=1-seed=0-{bs}-{2*bs}.npz",
            allow_pickle=True)

        timestamp = get_curr_formatted_timestamp()
        bounds = {
            'beta': data['beta_bound'],
            'gamma': data['gamma_bound']
        }

        origin = data['data']
        print(origin.shape)

        full_range = {
            'beta': np.linspace(- bounds['beta'], bounds['beta'], origin.shape[0]),
            'gamma': np.linspace(- bounds['gamma'], bounds['gamma'], origin.shape[1])
        }

        # print(full_range)

        # origin = {'ideals': landscape}
        # recon_dir = f"figs/recon/{timestamp}"
        fig_dir = f"{data_dir}/2D_CS_recon/sv-ideal"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        for sf in np.arange(0.01, 0.11, 0.02):
        # for sf in [0.05]:
            if not is_recon:
                seed = beta_steps
                np.random.seed(seed)
                recon = recon_2D_landscape(
                    origin=origin,
                    sampling_frac=sf
                )

                np.savez_compressed(f"{fig_dir}/sf{sf:.3f}_bs{bs}",
                    recon=recon,
                    seed=seed
                )
            else:
                recon = np.load(f"{fig_dir}/sf{sf:.3f}_bs{bs}.npz",
                    allow_pickle=True
                )['recon']

                # print(recon.shape)
                
            mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), "MSE")
            # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
            cos = cosine(origin.reshape(-1), recon.reshape(-1))
            mses.append(mse)
            coss.append(cos)

            # ncc = cal_recon_error()
            print("RMSE: ", mse)
            print("Cosine: ", cos)
            
            vis_landscapes(
                landscapes=[origin, recon],
                labels=["origin", "recon"],
                full_range={
                    "beta": full_range['beta'],
                    "gamma": full_range['gamma'] 
                },
                true_optima=None,
                title="Origin and recon",
                save_path=f'{fig_dir}/origin_and_2D_recon_sf{sf:.3f}_bs{bs}.png',
                params_paths=[None, None]
            )

    print("mse", mses)
    print("cos", coss)


# ! deprecated
def recon_large_qubits_p2_landscape_top():
    is_recon = True
    data_dir = "figs/gen_p2_landscape/2022-10-18_15:50:00_n6_p2_ideal"
    data = np.load(f"{data_dir}/sv-aer-gpu-n=16-p=2-seed=0-12-15.npz", allow_pickle=True)

    timestamp = get_curr_formatted_timestamp()
    bounds = {
        'beta': data['beta_bound'],
        'gamma': data['gamma_bound']
    }

    # n_pts_per_unit = {
    #     'beta': data['beta_bound'],
    #     'gamma': data['gamma_bound']
    # }

    # n_pts = {}
    # for label, bound in bounds.items():
    #     bound_len = bound[1] - bound[0]
    #     n_pts[label] = np.floor(n_pts_per_unit[label] * bound_len).astype(int)
    
    # print('bounds: ', bounds)
    # print('n_pts: ', n_pts)
    # print('n_pts_per_unit: ', n_pts_per_unit)

    # full_range = {
    #     'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
    #     'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    # }

    landscape = data['data']
    print(landscape.shape)

    # landscape = landscape.reshape(25, 50)

    full_range = {
        'beta': np.linspace(- bounds['beta'], bounds['beta'], landscape.shape[0]),
        'gamma': np.linspace(- bounds['gamma'], bounds['gamma'], landscape.shape[2])
    }
    print(full_range)

    # origin = {'ideals': landscape}
    # recon_dir = f"figs/recon/{timestamp}"
    figdir = f"{data_dir}/2D_CS_recon"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    # for sf in np.arange(0.05, 0.5, 0.03):
    for sf in [0.05]:
        # recon = two_D_CS_p1_recon_with_given_landscapes(
        #     figdir=None,
        #     origin=origin,
        #     full_range=None,
        #     sampling_frac=sf
        # )

        if not is_recon:
            recon = recon_4D_landscape_by_2D(
                origin=landscape,
                sampling_frac=sf
            )
            np.savez_compressed(f"{figdir}_sf{sf:.3f}",
                recon=recon
            )
        else:
            recon = np.load(f"{figdir}_sf{sf:.3f}.npz",
                allow_pickle=True
            )['recon']
            
        mse = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "MSE")
        # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
        cos = cosine(landscape.reshape(-1), recon.reshape(-1))
        # ncc = cal_recon_error()
        print("RMSE: ", mse)
        print("Cosine: ", cos)
        origin_2d = landscape.reshape(landscape.shape[0] * landscape.shape[1],
            landscape.shape[2] * landscape.shape[3])
        recon_2d = recon.reshape(recon.shape[0] * recon.shape[1],
            recon.shape[2] * recon.shape[3])

        vis_landscapes(
            landscapes=[origin_2d, recon_2d],
            labels=["origin", "recon"],
            full_range={
                "beta": range(origin_2d.shape[0]),
                "gamma": range(origin_2d.shape[1])
            },
            true_optima=None,
            title="Compare different ZNE configs and reconstruction",
            save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}.png',
            params_paths=[None, None]
        )


def CS_by_BPDN_p1():
    is_recon = False
    # data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    # data_dir = "figs/cnt_opt_miti/2022-08-09_16:49:38/G30_nQ8_p1"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)

    timestamp = get_curr_formatted_timestamp()
    # fig_dir = f"{data_dir}/2D_CS_recon_BPDN"
    # label = "unmitis"
    label = "mitis"
    method = "BPDN"
    # method = "BP"
    if is_recon:
        # fig_dir = f"figs/comp_cs_methods/2022-10-25_16:34:04_{method}_{label}"
        # fig_dir = "figs/comp_cs_methods/2022-10-25_16:42:04_BP_unmitis"
        pass
    else:
        fig_dir = f"figs/comp_cs_methods/{timestamp}_{method}_{label}_p1"

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    origin = data['origin'].tolist()[label]
    full_range = data['full_range'].tolist()

    mses = []
    coss = []
    for sf in np.arange(0.05, 0.5, 1):
        
        if not is_recon:
            seed = 42
            np.random.seed(seed)
            start = time.time()
            recon = recon_2D_landscape(
                origin=origin,
                sampling_frac=sf,
                method=method
            )

            end = time.time()
            print("time usage:", end - start)
            np.savez_compressed(f"{fig_dir}/sf{sf:.3f}",
                recon=recon,
                seed=seed,
                time_usage=end-start,
                sampling_frac=sf,
            )
        else:
            recon = np.load(f"{fig_dir}/sf{sf:.3f}.npz",
                allow_pickle=True
            )['recon']

            # print(recon.shape)
            
        mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), "MSE")
        # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
        cos = cosine(origin.reshape(-1), recon.reshape(-1))
        mses.append(mse)
        coss.append(cos)

        # ncc = cal_recon_error()
        print("RMSE: ", mse)
        print("Cosine: ", cos)
        
        vis_landscapes(
            landscapes=[origin.transpose(), recon.transpose()],
            labels=["origin", "recon"],
            full_range={
                "beta": full_range['beta'],
                "gamma": full_range['gamma'] 
            },
            true_optima=None,
            title="Origin and recon",
            save_path=f'{fig_dir}/origin_and_2D_recon_sf{sf:.3f}_mse{mse:.3f}_cos{cos:.3f}.png',
            params_paths=[None, None]
        )

    print("mse", mses)
    print("cos", coss)


def CS_by_BPDN_p2():
    is_recon = True

    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)

    timestamp = get_curr_formatted_timestamp()
    # fig_dir = f"{data_dir}/2D_CS_recon_BPDN"
    label = "unmitis"
    # label = "mitis"
    method = "BPDN"
    # method = "BP"
    if is_recon:
        fig_dir = "figs/recon_p2_landscape/2022-10-01_19:50:01" # BP
        # fig_dir = "figs/comp_cs_methods/2022-10-25_20:36:37_BPDN_unmitis_p2"
    else:
        fig_dir = f"figs/comp_cs_methods/{timestamp}_{method}_{label}_p2"

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    origin = data['origin'].tolist()[label]
    full_range = data['full_range'].tolist()

    mses = []
    coss = []
    for sf in np.arange(0.05, 0.5, 1):
        
        if not is_recon:
            seed = 42
            np.random.seed(seed)
            start = time.time()
            recon = recon_4D_landscape_by_2D(
                origin=origin,
                sampling_frac=sf,
                method=method
            )

            end = time.time()
            print("time usage:", end - start)
            np.savez_compressed(f"{fig_dir}/sf{sf:.3f}",
                recon=recon,
                seed=seed,
                time_usage=end-start,
                sampling_frac=sf,
            )
        else:
            recon = np.load(f"{fig_dir}/recon_p2_landscape_sf{sf:.3f}.npz",
                allow_pickle=True
            )['arr_0'] # BP

            # recon = np.load(f"{fig_dir}/sf{sf:.3f}.npz",
            #     allow_pickle=True
            # )['recon']

            # print(recon.shape)
            
        mse = cal_recon_error(origin.reshape(-1), recon.reshape(-1), "MSE")
        # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
        cos = cosine(origin.reshape(-1), recon.reshape(-1))
        mses.append(mse)
        coss.append(cos)

        origin_2d = origin.reshape(origin.shape[0] * origin.shape[1],
            origin.shape[2] * origin.shape[3])
        recon_2d = recon.reshape(recon.shape[0] * recon.shape[1],
            recon.shape[2] * recon.shape[3])

        # ncc = cal_recon_error()
        print("RMSE: ", mse)
        print("Cosine: ", cos)
        
        vis_landscapes(
            landscapes=[origin_2d, recon_2d],
            labels=["origin", "recon"],
            full_range={
                "beta": range(origin_2d.shape[0]),
                "gamma": range(origin_2d.shape[1]) 
            },
            true_optima=None,
            title="Origin and recon",
            save_path=f'{fig_dir}/origin_and_2D_recon_sf{sf:.3f}_mse{mse:.3f}_cos{cos:.3f}.png',
            params_paths=[None, None]
        )

    print("mse", mses)
    print("cos", coss)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--ns', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('-p', type=int, help="Your aims, vis, opt", required=True)
    # parser.add_argument('--method', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--noise', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--problem', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--n_seeds', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('--error', type=str, help="Your aims, vis, opt", required=True)
    args = parser.parse_args()
    
    if args.aim == 'large1':
        recon_large_qubits_p1_landscape_top()
    elif args.aim == 'large2':
        recon_large_qubits_p2_landscape_top()
    elif args.aim == 'comp1':
        CS_by_BPDN_p1()
    elif args.aim == 'comp2':
        CS_by_BPDN_p2()
    elif args.aim == 'ideal_p1':
        recon_p1_landscape_ideal_varying_qubits_and_instances()
    elif args.aim == 'noisy_p1':
        recon_p1_landscape_noisy_varying_qubits_and_instances()
    elif args.aim == 'ideal_p2':
        recon_p2_landscape_ideal_varying_qubits_and_instances()
    elif args.aim == 'noisy_p2':
        pass
    elif args.aim == 'final':
        recon_landscapes_varying_qubits_and_instances(p=args.p, problem=args.problem,
            noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns, error_type=args.error)
    else:
        raise NotImplementedError()