import argparse
import itertools
from lib2to3.pgen2.token import CIRCUMFLEX
from sqlite3 import paramstyle
from tabnanny import check
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
import timeit
import orqviz
from qiskit_optimization.algorithms import MinimumEigenOptimizer
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
from data_loader import load_grid_search_data
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
from qiskit.opflow import AerPauliExpectation, PauliExpectation
from qiskit_optimization.applications import Maxcut, SKModel, NumberPartition

test_utils_folder = Path(__file__).parent


def _get_min_e(
    G,
    p,
    C,
    initial_point, # QISKIT format
):
    # backend = AerSimulator(method=method)

    _, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # algorithm_globals.random_seed = seed
    quantum_instance = QuantumInstance(
        backend=AerSimulator(),
        shots=2048,
        # seed_simulator=seed,
        # seed_transpiler=seed,
    )

    algorithm = QAOA(
        optimizer=SPSA(),
        reps=p,
        initial_point=initial_point,
        quantum_instance=quantum_instance,
        # expectation=AerPauliExpectation() if args.aer else None,
    )
    
    # algorithm._check_operator_ansatz(C)
    result = algorithm.compute_minimum_eigenvalue(C)

    # algorithm._check_operator_ansatz(C)
    # energy_evaluation = algorithm.get_energy_evaluation(C)

    eigenvalue = result.eigenvalue
    # params = optimizer.params_path
    # print(params[0], angles_from_qiskit_format(params[0]))
    # print(counts)
    # print(qaoa.optimal_params)
    # print("opt_cut                     :", opt_cut)
    # print("offset:", offset)
    print("minimum:", eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))
    # print("len of params:", len(params))
    return eigenvalue



def _get_min_given_init_pt(
    G,
    p,
    C,
    initial_point: dict  # ! QAOA format
) -> float:

    noise_model = None
    qinst = AerSimulator()
    shots = 2048

    offset = 0
    _, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    def _partial_qaoa_energy(x):
        # TODO use unbinded circuit to speed up if necessary
        # by shots
        circuit = get_maxcut_qaoa_circuit(
            G, beta=x[p:], gamma=x[:p], # look at wrap_qiskit_optimizer_to_landscape_optimizer._fun()
            transpile_to_basis=False, save_state=False)

        energy = -_executor_of_qaoa_maxcut_energy(
            qc=circuit, G=G, noise_model=noise_model, shots=shots
        )
        print(energy, x)
        exit()
        return energy

        # by statevector
        # return -noisy_qaoa_maxcut_energy(
        #     G=G, beta=x[:p], gamma=x[p:], precomputed_energies=None, noise_model=noise_model
        # )

    # print(p)
    # raw_optimizer_clazz = SPSA
    raw_optimizer_clazz = ADAM
    
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer_clazz
    )(
        bounds=None,
        landscape=None,
        fun_type='FUN',
        fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        # lr=1e-3,
        # lr=1e-2,
        # maxiter=1000,
    )
    
    init_pt_qiskit = initial_point
    # init_pt_qiskit = angles_to_qiskit_format(angles_from_qaoa_format(**initial_point))
    print(init_pt_qiskit)
    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        initial_point=init_pt_qiskit,
        quantum_instance=qinst,
    )

    # I = Operator(np.eye(2**n_qubits))
    # I = Pauli("I" * n_qubits)
    # print(C)
    # optimizer.minimize(fun=None, x0=)
    # I = PrimitiveOp(I)
    # I = PauliSumOp.from_list([("I" * (4 * p), 1.0)])
    # result = qaoa.compute_minimum_eigenvalue(I) # useless
    result = qaoa.compute_minimum_eigenvalue(C)

    # eigenstate, eigenvalue, params_path = optimize_under_noise(
    #     G=G, init_beta_gamma=initial_point, noise_model=None, num_shots=shots, opt_method='L-BFGS-B'
    # )

    eigenvalue = result.eigenvalue
    # params = optimizer.params_path
    # print(params[0], angles_from_qiskit_format(params[0]))
    # print(counts)
    # print(qaoa.optimal_params)
    # print("opt_cut                     :", opt_cut)
    print("offset:", offset)
    print("minimum:", eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))
    # print("len of params:", len(params))
    return eigenvalue


def find_good_initial_points_on_recon_LS_and_verify_top(
    n_qubits: int, p: int, noise: str
):
    """Find good initial points on recon landscapes.

    This is to verify that recon. will give us correct information / intuition.

    qubits 24, seed 0: 'eigenvalue': (-14+0j)
    """
    method = 'sv'
    problem = 'maxcut'
    # noise = 'depolar-0.001-0.005'
    # n_qubits = 16
    cs_seed = n_qubits
    # p = 1
    sf = 0.05
    seed = 0
    if p == 2:
        bs = 12
        gs = 15
    elif p == 1:
        bs = 50
        gs = 100
    else:
        raise ValueError()

    data = load_grid_search_data(
        n_qubits=n_qubits, p=p, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed,
    )
    origin = data['data']

    algorithm_globals.massive = True
    algorithm_globals.random_seed = seed
 
    # full_ranges: list = data['full_ranges'].tolist()
    # bounds = data['bounds'].tolist()
    # n_pts_per_unit = data['n_pts_per_unit']
    # C_opt = data['C_opt']

    offset = data['offset']
    full_ranges = data['full_ranges']
    print("offset:", offset)

    if p == 2:
        # ! old path
        # recon_path = f"figs/gen_p1_landscape/sv-ideal/2D_CS_recon_p2/sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}.npz"
        # ! new path 新增seed和cs_seed
        recon_path = f"figs/gen_p1_landscape/sv-ideal/2D_CS_recon_p2/sf{sf:.3f}_bs{bs}_gs{gs}_nQ{n_qubits}_seed{seed}_csSeed{cs_seed}.npz"
        recon = np.load(f"{recon_path}")['recon']
    elif p == 1:
        recon_path = f"figs/gen_p1_landscape/sv-ideal/2D_CS_recon/sf{sf:.3f}_p1_bs{bs}_nQ{n_qubits}_csSeed{cs_seed}.npz"
        recon = np.load(f"{recon_path}")['recon']


    # get problem instance info from QAOAKit

    print("============= origin and recon 4-D LS are loaded ===============")
    print(f"shape of recon LS: {recon.shape}, origin LS: {origin.shape}")
    
    # find, say 100 points that are \epsilon-close to minimum value of recon landscape
    
    # recon = -recon
    # C_opt = -C_opt
    eps = 0.2 # 4
    # eps = 0.3 # 22
    # eps = 0.4 # 63
    # eps = 0.5 #
    
    min_recon = np.min(recon)
    print(f"minimum recon: {min_recon:.5f}, maximum recon: {np.max(recon)}")
    # print(f"C_opt: {C_opt:.5f}")

    mask = np.abs(recon - min_recon) < eps
    # print(mask)
    print(np.sum(mask == True))
    # return
    ids = np.argwhere(mask == True) # return indices of points where mask == True
    # print(ids)

    # landscape.shape = (beta, beta, gamma, gamma)
    # randomly sample some points within eps as initial points
    # print(full_ranges)
    # full_ranges = np.array(full_ranges)
    # print(ids)

    G = nx.random_regular_graph(3, n_qubits, seed)
    maxcut = Maxcut(G)
    # print(maxcut.get_gset_result)
    problem = maxcut.to_quadratic_program()
    H, offset = problem.to_ising()

    # QAOAKit' s method
    obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    opt_en = brute_force(obj, n_qubits)[0]
    print(opt_en)

    algo = NumPyMinimumEigensolver()
    # _, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
    #     G, p
    # )
    result = algo.compute_minimum_eigenvalue(operator=H, aux_operators=None)
    print(result)

    _, C, _ = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    assert H == C

    # print(H)

    # print(C)

    # return
    
    # return
    # reg3_dataset_table = get_3_reg_dataset_table()

    # row = get_full_qaoa_dataset_table_row(G, p)
    # df = reg3_dataset_table.reset_index()
    # # print(df['p_max'].value_counts())
    # # df = 
    # df = df[(df["G"] == G)]
    # for row_id, row in df.iloc[0:1].iterrows():
    #     pass
    
    # print(row)
    # G = row['G']
    # problem = Maxcut(G).to_quadratic_program()
    # H, offset = problem.to_ising()
    inits = []
    # return
    min_energies = []
    for idx in ids: # idx: tuples
        # print(idx, recon[idx[0], idx[1], idx[2], idx[3]])
        # continue
        init_beta_gamma = [0 for _ in range(2*p)]
        for i in range(2*p):
            init_beta_gamma[i] = full_ranges[i][idx[i]]

        # initial_point = {
        #     'beta': np.array(init_beta_gamma[:p]),
        #     'gamma': np.array(init_beta_gamma[p:]),
        # }
        print(init_beta_gamma)
        # format of Qiskit, p=2: gamma, beta, gamma, beta
        initial_point = np.zeros(shape=2*p)
        initial_point[::2] = init_beta_gamma[p:]
        initial_point[1::2] = init_beta_gamma[:p]
        print(initial_point)

        inits.append(initial_point)
        # return
        # initial_point = np.array(init_beta_gamma)

        # print(initial_point)
        # min_energy = _get_min_e(G, p, None, initial_point)
        min_energy = _get_min_given_init_pt(G=G, p=p, C=H, initial_point=initial_point)

        min_energies.append(min_energy)

    timestamp = get_curr_formatted_timestamp()
    save_dir = f"figs/find_init_pts_by_recon/{timestamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}/nQ{n_qubits}_p{p}_seed{seed}_csSeed{cs_seed}"
    print("save path: ", save_path)
    np.savez_compressed(
        save_path,
        ids=ids,
        initial_points=inits, # one to one correspondence to ids
        min_energies=min_energies, # one to one correspondence to ids
        min_recon=min_recon,
        eps=eps,
        # C_opt=C_opt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim', type=str, help="Your aims, vis, opt", required=True)
    # parser.add_argument('-n', type=int, help="Number of qubits", required=True)
    # parser.add_argument('-p', type=str, help="QAOA layers")

    args = parser.parse_args()
    
    if args.aim == 'find':
        find_good_initial_points_on_recon_LS_and_verify_top(
            n_qubits=16, p=2, noise="ideal"
        )
    else:
        raise NotImplementedError()