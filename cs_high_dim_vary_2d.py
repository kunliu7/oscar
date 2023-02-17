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
    p: int, ansatz: str, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list, error_type: str
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
    elif p == 0:
        bs = 14
        gs = 100

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
                n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
                noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
            )

            plot_range = data['plot_range']

            # 和Tianyi代码使用相同目录结构
            recon_dir = f"figs/grid_search_recon/{ansatz}/{problem}/{method}-{noise}-p={p}"

            for sf in sfs:
                recon_fname = f"recon-cs_seed={cs_seed}-sf={sf:.3f}-{data_fname}"
                recon_path = f"{recon_dir}/{recon_fname}"

                origin = data['data']
                recon = get_recon_landscape(ansatz, p, origin, sf, is_recon, 
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


def gen_heapmap_by_varying_sampling_fraction_and_beta_step(
    p: int, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list, error_type: str
):
    is_recon = False

    method = 'sv'
    miti_method = ''
    mses = []
    coss = []

    sfs = np.arange(0.01, 0.11, 0.02)
    if len(n_seeds) == 1:
        seeds = list(range(n_seeds[0]))
    elif len(n_seeds) == 2:
        seeds = list(range(n_seeds[0], n_seeds[1]))

    bss = range(25, 76, 5)

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)
    print("bss =", bss)

    # 和Tianyi代码使用相同目录结构
    recon_dir = f"figs/grid_search_recon/{problem}/{method}-{noise}-p={p}"
    for n_qubits, seed, sf, bs in itertools.product(n_qubits_list, seeds, sfs, bss):
        cs_seed = n_qubits # ! compare horizontally
        gs = 2 * bs

        data, data_fname, data_dir = load_grid_search_data(
            n_qubits=n_qubits, p=p, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
        )

        plot_range = data['plot_range']

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
        print("NRMSE: ", mse)
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

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs), len(bss))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs), len(bss))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    # timestamp = get_curr_formatted_timestamp()
    recon_error_save_dir = f"{recon_dir}/recon_error_ns={n_qubits_list}-seeds={seeds}-sfs={sfs}-bss={bss}-error={error_type}"
    print(f"recon error data save to {recon_error_save_dir}")
    np.savez_compressed(
        recon_error_save_dir,
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs,
        bss=bss,
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
    parser.add_argument('--ansatz', type=str, help="Your aims, vis, opt", required=True)
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
    elif args.aim == 'heatmap':
        gen_heapmap_by_varying_sampling_fraction_and_beta_step(p=args.p, problem=args.problem,
            noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns, error_type=args.error)
    elif args.aim == 'final':
        recon_landscapes_varying_qubits_and_instances(p=args.p, ansatz=args.ansatz, problem=args.problem,
            noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns, error_type=args.error)
    else:
        raise NotImplementedError()