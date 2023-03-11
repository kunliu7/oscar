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
from data_loader import load_grid_search_data, get_recon_landscape, load_ibm_data
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

from scipy.fftpack import dctn, idctn, fftn
from scipy.ndimage import fourier_shift
import scipy as sp

def nonzero_ratio(ls: np.ndarray, metric: str = 'DCT'):
    if metric == 'DCT':
        f = dctn(ls)
    elif metric == 'FFT':
        f = np.fft.fftn(ls)
    elif metric == 'PS':
        f = fftn(ls)
        d = len(ls.shape)
        shift = (ls.shape[i] // 2 for i in range(d))
        f = fourier_shift(f, shift)
        f = np.abs(f)**2
        f /= f.max()

    nonzero_ids = np.argwhere(np.abs(f) > 1e-6)
    nonzero_ratio = len(nonzero_ids) * 1.0 / np.prod(ls.shape)
    
    # print("Using DCT:")
    # print("original shape            =", ls.shape)
    # print("original min, max         =", ls.min(), ls.max())
    # print("after transform, shape    =", f.shape)
    # print("after transform, min, max =", f.min(), f.max())
    # print("# nonzero, nonzero_ratio  =", len(nonzero_ids), nonzero_ratio)

    return nonzero_ratio


def measure_sparsity(
    p: int, ansatz: str, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list, device: str
):
    method = 'sv'
    miti_method = ''

    labels = ['DCT', 'FFT', 'PS']
    ratios = {label: [] for label in labels}

    if p == 1:
        bs = 50 # beta step
        gs = 2 * bs
    elif p == 2:
        bs = 12
        gs = 15
    elif p == 0:
        bs = 14
        gs = 100

    if len(n_seeds) == 1:
        seeds = list(range(n_seeds[0]))
    elif len(n_seeds) == 2:
        seeds = list(range(n_seeds[0], n_seeds[1]))

    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    for n_qubits in n_qubits_list:
        for seed in seeds:
            data, data_fname, data_dir = load_grid_search_data(
                n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
                noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
            )
            
            for label in labels:
                ratio = nonzero_ratio(data['data'], metric=label)
                ratios[label].append(ratio)

    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)

    for label in labels:
        print(f"{label} ratio =", ratios[label])
        ratios[label] = np.array(ratios[label]).reshape(len(n_qubits_list), len(seeds))

    save_dir = f"figs/sparsity/"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    save_fname = f"ns={n_qubits_list}-seeds={seeds}-ansatz={ansatz}-problem={problem}-noise={noise}-{p=}"
    save_path = os.path.join(save_dir, save_fname)
    print(f"data save to {save_path}")
    np.savez_compressed(
        save_path,
        **ratios,
        labels=labels,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('--p', type=int, help="Your aims, vis, opt", required=True)
    parser.add_argument('--noise', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--problem', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--n_seeds', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('--ansatz', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--device', type=str, help="Your aims, vis, opt", default=None)
    args = parser.parse_args()
    
    measure_sparsity(p=args.p, ansatz=args.ansatz, problem=args.problem,
            noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns, device=args.device)