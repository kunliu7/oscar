import argparse
import itertools
from tkinter.messagebox import NO
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
import timeit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import PrimitiveOp
from qiskit_optimization.applications import Maxcut, SKModel, NumberPartition
from qiskit.algorithms.optimizers.optimizer import POINT
from mitiq.zne.zne import execute_with_zne
from qiskit.providers.aer.noise import NoiseModel
from qiskit.algorithms.optimizers.optimizer import Optimizer as QiskitOptimizer
from mitiq.zne.inference import (
    LinearFactory
)
from typing import Callable, List, Optional, Tuple
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

from data_loader import get_interpolation_path_filename, get_recon_landscape, get_recon_pathname, load_grid_search_data, load_optimization_path
# from QAOAKit import vis

sys.path.append('..')
from QAOAKit.noisy_params_optim import (
    get_pauli_error_noise_model,
    optimize_under_noise,
    get_depolarizing_error_noise_model,
    compute_expectation
)

from QAOAKit.vis import(
    vis_landscape,
    vis_landscape_heatmap,
    vis_landscape_heatmap_multi_p,
    vis_landscape_multi_p,
    vis_landscape_multi_p_and_and_count_optima,
    vis_landscape_multi_p_and_and_count_optima_MP,
    vis_multi_landscape_and_count_optima_and_mitiq_MP,
    vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable,
    vis_two_BPs_p1_recon,
    vis_landscapes
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
    arraylike_to_str,
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
    gen_p1_landscape,
    two_D_CS_p1_recon_with_given_landscapes,
    cal_recon_error
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

test_utils_folder = Path(__file__).parent


def get_point_val(G: nx.Graph, p: int, last_pt: np.ndarray, noise_model: NoiseModel):
    maxcut = Maxcut(G)
    problem = maxcut.to_quadratic_program()
    C, offset = problem.to_ising()

    # qinst = AerSimulator(method='statevector', noise_model=noise_model)
    qinst = AerSimulator(shots=1024, noise_model=noise_model)

    qaoa = QAOA(
        reps=p,
        quantum_instance=qinst,
    )

    qaoa._check_operator_ansatz(C)
    energy_evaluation, expectation = qaoa.get_energy_evaluation(
        C, return_expectation=True
    )

    return energy_evaluation(last_pt)


def get_minimum_by_QAOA(G: nx.Graph, p: int, qiskit_init_pt: np.ndarray,
    noise_model: NoiseModel, optimizer: QiskitOptimizer):
    """Get minimum cost value by random initialization.

    Returns:
        eigenvalue, qiskit_init_pt, params, values
    """
    maxcut = Maxcut(G)
    problem = maxcut.to_quadratic_program()
    C, offset = problem.to_ising()

    counts = []
    values = []
    params = []
    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    qinst = AerSimulator(shots=1024, noise_model=noise_model)
    # qinst = AerSimulator(method='statevector')
    # shots = 2048
    # optimizer = SPSA(maxiter=maxiter)
    # if isinstance(maxiter, int):
    #     # optimizer = SPSA(maxiter=maxiter)
    #     optimizer = COBYLA(maxiter=maxiter)
    #     print("maxiter:", maxiter)
    # else:
    #     # optimizer = SPSA()
    #     optimizer = COBYLA()
    #     print("no maxiter")
    
    print("noise model", noise_model)

    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        initial_point=qiskit_init_pt,
        quantum_instance=qinst,
        callback=cb_store_intermediate_result
    )
    # print(qaoa.ansatz)
    result = qaoa.compute_minimum_eigenvalue(C)
    eigenvalue = result.eigenvalue.real

    print("offset                 :", offset)
    print("QAOA minimum           :", eigenvalue)
    print("Real minimum (+offset) :", eigenvalue + offset)

    return eigenvalue, qiskit_init_pt, params, values


def batch_eval_opt_on_recon_ls(n: int, seed_range: List[int], noise: str, opt: str):
    p = 1
    if len(seed_range) == 1:
        seeds = list(range(seed_range[0]))
    elif len(seed_range) == 2:
        seeds = list(range(seed_range[0], seed_range[1]))
    else:
        raise NotImplementedError()

    if opt == 'ADAM':
        maxiter = 10000
    elif opt == 'COBYLA':
        maxiter = 1000
    elif opt == 'SPSA':
        maxiter = 100
    else:
        raise ValueError()

    miti_method = None
    
    intp_paths = []
    initial_points = []
    # for seed, opt in itertools.product(seeds, opts):
    for seed in seeds:
        print(f"{n=}, {seed=}, {opt=}")

        intp_path, initial_point = optimize_on_p1_reconstructed_landscape(
            n, p, seed, noise, miti_method,
            None, opt, None, maxiter, False
        )

        initial_points.append(initial_point)
        intp_paths.append(intp_path.copy())

    assert len(intp_paths) == len(seeds)
    print("n: ", n)
    print("seeds: ", seeds)
    print("opt: ", opt)

    save_dir = f"figs/second_optimize"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/opt_on_recon-{n=}-noise={noise}-seeds={arraylike_to_str(seeds)}-opt={opt}-{maxiter=}"
    print("save to", save_path)

    np.savez_compressed(
        save_path,
        save_path=save_path,
        seeds=seeds,
        initial_points=initial_points,
        intp_paths=intp_paths,
        opt=opt,
        maxiter=maxiter
    )


def optimize_on_p1_reconstructed_landscape(
    n: int, p: int, seed: int, noise: str, miti_method: str,
    initial_point: List[float], opt_name: str, lr: float, maxiter: int, is_sim: bool
) -> Tuple[float, float]:
    """

    Args:
        initial_point (List[float]): [beta, gamma]

    """
    noise_cfgs = noise.split('-')
    if noise_cfgs[0] == 'ideal':
        noise_model = None
    elif noise_cfgs[0] == 'depolar':
        noise_model = get_depolarizing_error_noise_model(float(noise_cfgs[1]), float(noise_cfgs[2]))
        print(noise_cfgs)
    else:
        raise NotImplementedError()

    problem = 'maxcut'
    method = 'sv'
    cs_seed = n
    assert p == 1
    # assert len(initial_point) == 2 * p
    sf = 0.05
    if p == 1:
        bs = 50
        gs = 100
    elif p == 2:
        bs = 12
        gs = 15
    else:
        raise NotImplementedError()
    
    data, data_fname, data_dir = load_grid_search_data(
        n_qubits=n, p=p, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
    )

    beta_bound = data['beta_bound']
    gamma_bound = data['gamma_bound']
    plot_range = data['plot_range']
    origin = data['data']

    recon_path, recon_fname, recon_dir = get_recon_pathname(p, problem, method, noise, cs_seed, sf, data_fname)
    print("tend to save to", recon_path)
    recon = get_recon_landscape(p, origin, sf, False, recon_path, cs_seed)

    G = nx.random_regular_graph(3, n, seed)
    maxcut = Maxcut(G)
    # print(maxcut.get_gset_result)
    maxcut_problem = maxcut.to_quadratic_program()
    C, offset = maxcut_problem.to_ising()

    if n <= 16:
        row = get_3_reg_dataset_table_row(G, p)
        opt_cut = row["C_opt"]
    else:
        opt_cut = None

    bounds = np.array([
        [-beta_bound, beta_bound], 
        [-gamma_bound, gamma_bound], 
    ])

    print("bounds:", bounds)
    print("landscape shape:", recon.shape)

    print("initial point:", initial_point)
    if not initial_point:
        rng = np.random.default_rng(seed)
        # initial_point = np.array(initial_point)
        beta = rng.uniform(-beta_bound, beta_bound)
        gamma = rng.uniform(-gamma_bound, gamma_bound)

        initial_point = np.array([beta, gamma])
        print(f"{seed=}, {initial_point=}, {beta_bound=}, {gamma_bound=}")

    # ---------------- data prepare -------------------

    intp_path_path, intp_path_fname, intp_path_dir = get_interpolation_path_filename(n, p, problem, method, noise, opt_name, maxiter, initial_point, seed, miti_method)
    is_data_existed = os.path.exists(intp_path_path)
    # if is_data_existed:
    #     data = np.load(f"{intp_path_path}", allow_pickle=True)
    #     return data['intp_path'][-1], data['circ_path'][-1]


    # raw_optimizer = 'SPSA'
    if opt_name == 'ADAM':
        raw_optimizer = ADAM
    elif opt_name == 'SPSA':
        raw_optimizer = SPSA
    elif opt_name == 'COBYLA':
        raw_optimizer = COBYLA
    elif opt_name == 'L_BFGS_B':
        raw_optimizer = L_BFGS_B
    else:
        raise NotImplementedError()

    opt_params = {}
    if lr:
        opt_params['lr'] = lr
    if maxiter:
        opt_params['maxiter'] = maxiter
    print("optimizer: ", opt_name)
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer
    )(
        bounds=bounds, 
        landscape=recon,
        fun_type='INTERPOLATE_QISKIT',
        # fun_type='FUN',
        # fun=None
        # fun_type='None'
        **opt_params
    )
    
    # opts = [ADAM,
    # AQGD,
    # CG,
    # COBYLA,
    # L_BFGS_B,
    # GSLS,
    # GradientDescent,
    # NELDER_MEAD,
    # NFT,
    # P_BFGS,
    # POWELL,
    # SLSQP,
    # SPSA,
    # QNSPSA,
    # TNC,
    # SciPyOptimizer]

    # initial_point = np.hstack([[1.0 for _ in range(p)], [-1.0 for _ in range(p)]])
        
    # initial_point = np.array([0.1, -0.1])
    qinst = AerSimulator() # meaningless, do not actually activate
    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        # initial_point=angles_to_qiskit_format(
        #     {"gamma": row["gamma"],
        #     "beta": row["beta"]}
        # ),
        initial_point=initial_point,
        quantum_instance=qinst,
        # callback=cb_store_intermediate_result
        )
    result = qaoa.compute_minimum_eigenvalue(C)
    # print(qaoa.optimal_params)
    print("opt_cut                     :", opt_cut)
    print("recon landscape minimum     :", result.eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))

    params = optimizer.params_path
    # vals = optimizer
    # print(params)
    print("len of params:", len(params))
    
    params = [
        # _params
        shift_parameters(_params, bounds)
        # angles_to_qaoa_format(angles_from_qiskit_format(_params))
        for _params in params
    ]
    
    
    if is_sim:
        # 做不动
        # _, _, circ_path, _ = get_minimum_by_QAOA(G, p, initial_point, None, raw_optimizer(lr=lr, maxiter=maxiter))
        circ_path = load_optimization_path(n, p, problem, method, noise, opt_name, lr, maxiter, initial_point, seed, miti_method)
        print("len of circuit simulation path:", len(circ_path))
        circ_vals = []
        for ipt, pt in enumerate(circ_path):
            print(f"\r{ipt} th / {len(circ_path)}", end="")
            circ_vals.append(get_point_val(G, p, pt, None))
            # circ_vals = [get_point_val(G, p, pt, None) for pt in circ_path]
    else:
        circ_path = None
        circ_vals = None

    print(initial_point)
    # ts = get_curr_formatted_timestamp()

    # save_dir = f"figs/opt_on_recon_landscape/{ts}"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # print(params)
    # print(circ_path)
    # save_path = f"{save_dir}/opt_on_recon-opt={opt_name}-init={list(initial_point)}-{base_recon_fname}"
    save_path = intp_path_path
    intp_path_fname_base = os.path.splitext(intp_path_fname)[0]
    print("params save to =", save_path)
    np.savez_compressed(
        save_path,
        opt_name=opt_name,
        initial_point=initial_point,
        intp_path=params, # interpolation
        # intp_vals=
        circ_path=circ_path,
        circ_vals=circ_vals
    )

    vis_landscapes(
        landscapes=[recon, origin],
        labels=["Interpolate", "Circuit Sim."],
        full_range=plot_range,
        true_optima=None,
        title="Origin and recon",
        save_path=f'{intp_path_dir}/vis-{intp_path_fname_base}.png',
        params_paths=[params, circ_path]
    )
    
    # save_dir = f"figs/second_optimize/{get_curr_formatted_timestamp()}"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # save_path = f"{save_dir}/second-optimize-{intp_path_fname_base}"
    # print("\nsecond optimize data save to", save_path)
    # np.savez_compressed(
    #     save_path,
    #     opt_name=opt_name,
    #     initial_point=initial_point,
    #     intp_path=params, # interpolation
    #     intp_path_last_pt=
    #     # intp_vals=
    #     # circ_path=circ_path,
    #     # circ_vals=circ_vals,
    #     **opt_params
    # )

    return params, initial_point


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help="Number of qubits.", required=True)
    parser.add_argument('--seed_range', type=int, nargs="+", required=True)
    parser.add_argument('--noise', type=str, required=True)
    parser.add_argument('--opt', type=str, required=True)
    args = parser.parse_args()

    batch_eval_opt_on_recon_ls(args.n, args.seed_range, args.noise, args.opt)