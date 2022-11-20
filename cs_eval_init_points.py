import argparse
import itertools
from sqlite3 import paramstyle
from tabnanny import check
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
import timeit

from qiskit.compiler import transpile, assemble
from qiskit.providers.aer.noise import NoiseModel
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
from data_loader import get_recon_landscape, load_grid_search_data
# from QAOAKit import vis

sys.path.append('..')
from QAOAKit.noisy_params_optim import (
    get_pauli_error_noise_model,
    optimize_under_noise,
    get_depolarizing_error_noise_model,
    compute_expectation
)

# from QAOAKit.vis import(
#     _vis_recon_distributed_landscape,
#     vis_landscape,
#     vis_landscape_heatmap,
#     vis_landscape_heatmap_multi_p,
#     vis_landscape_multi_p,
#     vis_landscape_multi_p_and_and_count_optima,
#     vis_landscape_multi_p_and_and_count_optima_MP,
#     vis_landscapes,
#     vis_multi_landscape_and_count_optima_and_mitiq_MP,
#     vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable,
#     vis_two_BPs_p1_recon
# )

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

# from QAOAKit.interpolate import (
#     approximate_fun_value_by_2D_interpolation
# )

# from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz

from scipy import interpolate
from qiskit.opflow import AerPauliExpectation, PauliExpectation
from qiskit_optimization.applications import Maxcut, SKModel, NumberPartition

from data_loader import get_recon_pathname
# test_utils_folder = Path(__file__).parent

def test_qiskit_qaoa_circuit():
    full_qaoa_dataset_table = get_3_reg_dataset_table()
    for n_qubits in [8]:
        p = 1
        df = full_qaoa_dataset_table.reset_index()
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
        for _, row in df.iterrows():
            angles1 = opt_angles_for_graph(row["G"], row["p_max"])
            G = row["G"]
            C_opt = row['C_opt']

            qiskit_angles = angles_to_qiskit_format(angles1)
            qc1, C, offset = get_maxcut_qaoa_qiskit_circuit(
                G, p, qiskit_angles
            )
            backend = AerSimulator(method="statevector")
            
            # ! why this is wrong? version issue? of QAOAAnsatz?
            qc1_tran = transpile(qc1, backend)
            # , optimization_level=0, basis_gates=["u1", "u2", "u3", "cx"])
            # qobj = assemble(qc1_tran)
            sv1 = Statevector(backend.run(qc1_tran).result().get_statevector())

            # ! the inner function takes the following form:
            inner_qaoa_angle = np.zeros_like(qiskit_angles)
            inner_qaoa_angle[:p] = qiskit_angles[1::2] # betas
            inner_qaoa_angle[p:] = qiskit_angles[::2] # gammas

            maxcut = Maxcut(G)
            problem = maxcut.to_quadratic_program()
            H, offset_ = problem.to_ising()
            print(offset_)
            assert H == C
            assert np.isclose(offset_, offset)

            get_energy = get_wrapper(p, C)
            energy = get_energy(inner_qaoa_angle)
            print(energy)

            # --------------------------------

            angles2 = angles_to_qaoa_format(
                opt_angles_for_graph(row["G"], row["p_max"])
            )
            qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
            sv2 = Statevector(backend.run(qc2).result().get_statevector())

            print(C_opt, offset)
            obj_val1 = sv1.expectation_value(C) + offset
            # print(obj_val1)
            # obj_val1 = -(sv1.expectation_value(C) + offset)
            obj_val2 = -(sv2.expectation_value(C) + offset)
            print(obj_val1, obj_val2)
            print("--------")
            # assert sv1.equiv(sv2)


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
        # print(energy, x)
        # exit()
        return energy

        # by statevector
        # return -noisy_qaoa_maxcut_energy(
        #     G=G, beta=x[:p], gamma=x[p:], precomputed_energies=None, noise_model=noise_model
        # )

    # print(p)
    raw_optimizer_clazz = SPSA
    # raw_optimizer_clazz = ADAM
    
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

    print(qaoa.ansatz.parameters)

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


def get_wrapper(p, C):
    backend = AerSimulator(method='statevector')
    qinst = QuantumInstance(
        backend=backend,
        # seed_simulator=seed,
        # seed_transpiler=seed,
    )

    algorithm = QAOA(
        SPSA(),
        reps=p,
        quantum_instance=qinst,
        # expectation=AerPauliExpectation() if args.aer else None,
    )
    algorithm._check_operator_ansatz(C)
    energy_evaluation = algorithm.get_energy_evaluation(C)
    return energy_evaluation


def var_of_grad(ls) -> list:
    grad = np.gradient(ls)

    var_of_grad = [g.var() for g in grad]
    return var_of_grad


def cal_barren_plateaus(ls) -> float:
    return np.mean(var_of_grad(ls))


def cal_min_barren_plateaus(ls) -> float:
    return np.min(var_of_grad(ls))


def cal_local_barren_plateaus(center, ls, stride: int):
    shape = ls.shape
    p = len(shape) // 2

    bounds = np.zeros(shape=(len(shape), 2))
    bounds[:, 1] = shape

    ranges = bounds.copy()

    ranges[:, 0] = np.maximum(center - stride, bounds[:, 0])
    ranges[:, 1] = np.minimum(center + stride + 1, bounds[:, 1])
    
    ranges = np.array(ranges, dtype=int)
    print("center:", center)
    print("ranges:", ranges)

    if p == 2:
        local = ls[ranges[0, 0]:ranges[0, 1],
                    ranges[1, 0]:ranges[1, 1],
                    ranges[2, 0]:ranges[2, 1],
                    ranges[3, 0]:ranges[3, 1]]

        print(f"totally we have {np.prod(local.shape)} many points, shape = {local.shape}")
    elif p == 1:
        local = ls[ranges[0, 0]:ranges[0, 1],
                    ranges[1, 0]:ranges[1, 1]]

        print(f"totally we have {np.prod(local.shape)} many points, shape = {local.shape}")
    else:
        raise NotImplementedError()


    return cal_barren_plateaus(local)
    # return cal_min_barren_plateaus(local)


def get_minimum_by_QAOA(G: nx.Graph, p: int, qiskit_init_pt: np.ndarray,
    noise_model: NoiseModel, maxiter: int):
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

    qinst = AerSimulator(shots=2048, noise_model=noise_model)
    # shots = 2048
    # optimizer = SPSA(maxiter=maxiter)
    if isinstance(maxiter, int):
        # optimizer = SPSA(maxiter=maxiter)
        optimizer = COBYLA(maxiter=maxiter)
        print("maxiter:", maxiter)
    else:
        # optimizer = SPSA()
        optimizer = COBYLA()
        print("no maxiter")
    
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


def get_random_initialization(G: nx.Graph, p: int, n_pts: int,
    noise_model: NoiseModel, beta_bound: float, gamma_bound: float, maxiter: int, random_seed: int):
    print(f"beta_bound={beta_bound}, gamma_bound={gamma_bound}, random_seed={random_seed}")
    rng = np.random.default_rng(random_seed)

    betas = [rng.uniform(-beta_bound, beta_bound, n_pts) for _ in range(p)]
    gammas = [rng.uniform(-gamma_bound, gamma_bound, n_pts) for _ in range(p)]
    inits = np.concatenate(betas + gammas)
    print(inits.shape)
    inits = inits.reshape(2*p, n_pts).T
    print("init points shape:", inits.shape)
    assert inits.shape == (n_pts, 2*p)

    results = []
    qiskit_inits = []
    for i in range(n_pts):
        print(f"----- {i}-th ------")

        init_betas_gammas = inits[i,:]

        # format of Qiskit, p=2: gamma, beta, gamma, beta
        # qiskit_init = np.zeros(shape=2*p)
        # qiskit_init[::2] = init_betas_gammas[p:] # gammas
        # qiskit_init[1::2] = init_betas_gammas[:p] # betas
        qiskit_init = init_betas_gammas
        qiskit_inits.append(qiskit_init)
        rst = get_minimum_by_QAOA(G, p, qiskit_init, noise_model, maxiter)
        results.append(rst)

    # energies = np.array(energies) 
    qiskit_inits = np.array(qiskit_inits)
    return results, qiskit_inits


def find_good_initial_points_on_recon_LS_and_verify_top(
    n_qubits: int, p: int, noise: str, p1: float, p2: float, maxiter:int, eps: List[float],
    instance_seed: int, random_seed: int, stride: int, check: bool=False
):
    """Find good initial points on recon landscapes.

    This is to verify that recon. will give us correct information / intuition.

    qubits 24, seed 0: 'eigenvalue': (-14+0j)
    """
    method = 'sv'
    problem = 'maxcut'

    if noise == "ideal":
        noise_model = None
        noise = 'ideal'
    elif noise == "depolar":
        noise_model = get_depolarizing_error_noise_model(p1, p2)
        noise = f'{noise}-{p1}-{p2}'
    elif noise == "pauli":
        noise_model = get_pauli_error_noise_model(p1)
        noise = f'{noise}-{p1}'
    else:
        raise NotImplementedError(f"Noise model {args.noise} not implemented yet")

    print("noise:", noise)
    # noise = 'ideal'
    # n_qubits = 16
    cs_seed = n_qubits
    sf = 0.05
    seed = instance_seed
    if p == 2:
        bs = 12
        gs = 15
    elif p == 1:
        bs = 50
        gs = 100
    else:
        raise ValueError("Invalid depth of QAOA")

    data, data_fname, _ = load_grid_search_data(
        n_qubits=n_qubits, p=p, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed,
    )
    origin = data['data']
    offset = data['offset']
    full_ranges = data['full_ranges']
    print("offset:", offset)

    algorithm_globals.massive = True
    algorithm_globals.random_seed = seed

    G = nx.random_regular_graph(3, n_qubits, seed)
    if n_qubits <= 16:
        row = get_3_reg_dataset_table_row(G, p)
        # print(row)
        C_opt = -row['C_opt']
        print("QAOAKit has its minimum: ", C_opt)
    else:
        print("QAOAKit does not have true optima of this graph")
        print("compared with random initialization")
        C_opt = None

    # get_minimum_by_random_init(G, None, p)

    # TODO: check
    qaoakit_angles = opt_angles_for_graph(G, p)
    qiskit_angles = angles_to_qiskit_format(qaoakit_angles)
    print("opt angle:", qiskit_angles)

    recon_path, _, _ = get_recon_pathname(p, problem, method, noise, cs_seed, sf, data_fname)
    recon = get_recon_landscape(p, origin, sf, False, recon_path, cs_seed)

    print("============= origin and recon LS are loaded ===============")
    print(f"shape of recon LS: {recon.shape}, origin LS: {origin.shape}")
    
    # global_bp = cal_barren_plateaus(recon)
    global_bp = cal_barren_plateaus(recon)
    # global_bp = cal_min_barren_plateaus(recon)
    print("global bp =", global_bp)
    
    min_recon = np.min(recon)
    print(f"minimum recon: {min_recon:.5f}, maximum recon: {np.max(recon)}")
    # print(f"C_opt: {C_opt:.5f}")

    if len(eps) == 1:
        mask = np.abs(recon - min_recon) < eps[0]
    elif len(eps) == 2:
        diff_abs = np.abs(recon - min_recon)
        mask = np.logical_and(diff_abs >= eps[0], diff_abs < eps[1])
        
    # print(mask)
    n_pts = np.sum(mask == True)
    print("number of points: ", n_pts)
    # return
    ids = np.argwhere(mask == True) # return indices of points where mask == True
    # print(ids)

    # return
    # landscape.shape = (beta, beta, gamma, gamma)
    # randomly sample some points within eps as initial points
    # print(full_ranges)
    # full_ranges = np.array(full_ranges)
    # print(ids)

    # G = nx.random_regular_graph(3, n_qubits, seed)
    maxcut = Maxcut(G)
    # print(maxcut.get_gset_result)
    problem = maxcut.to_quadratic_program()
    H, offset = problem.to_ising()

    # brute force 1, but it is not the result of QAOA
    if False:
        print("---- brute force ---")
        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
        opt_en = brute_force(obj, n_qubits)[0]
        print(opt_en)

    # brute force 2, but it is not the result of QAOA
    if False:
        print("---- NumPyMinimumEigensolver ----")
        algo = NumPyMinimumEigensolver()
        result = algo.compute_minimum_eigenvalue(operator=H, aux_operators=None)
        print(result)

    # check H generated are the same
    if False:
        print("--- check H generated are the same -----")
        _, C, _ = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
            G, p
        )

        # angles_to_qiskit_format(angles=)
        assert H == C

    print("============== start OSCAR =============")
    
    oscar_inits = []
    oscar_mins = []
    recon_init_pt_vals = []
    origin_init_pt_vals = []
    oscar_rsts = []
    ids_higher_bp = []
    for i, idx in enumerate(ids): # idx: tuples
        print(f"----------- {i}-th ----------------")
        # print(f"\rProcess: {i:>3} / {len(ids)}")
        # print(idx, recon[idx[0], idx[1], idx[2], idx[3]])
        # continue
        init_beta_gamma = [0 for _ in range(2*p)]
        for j in range(2*p):
            init_beta_gamma[j] = full_ranges[j][idx[j]]

        local_bp = cal_local_barren_plateaus(idx, recon, stride)
        print("local bp", local_bp)
        
        recon_init_pt_val = recon[tuple(idx)]
        origin_init_pt_val = origin[tuple(idx)]
        print(f"recon_init_pt_val={recon_init_pt_val}, origin_init_pt_val={origin_init_pt_val}")
        recon_init_pt_vals.append(recon_init_pt_val)
        origin_init_pt_vals.append(origin_init_pt_val)
        
        # print(init_beta_gamma)
        # format of Qiskit, p=2: gamma, beta, gamma, beta
        
        # initial_point = np.zeros(shape=2*p)
        # initial_point[::2] = init_beta_gamma[p:]
        # initial_point[1::2] = init_beta_gamma[:p]
        """Fuck
        """
        initial_point = init_beta_gamma
        print(initial_point)

        if local_bp > global_bp:
            # ids_higher_bp.append(idx)
            ids_higher_bp.append(i)

        oscar_inits.append(initial_point) 
        rst = get_minimum_by_QAOA(G, p, initial_point, noise_model, maxiter)

        print(f"oscar min={rst[0]}, init_pt={rst[1]}, actual_init_pt={rst[2][0]}, \
            actual_init_pt_val={rst[3][0]}, recon_init_pt_val={recon_init_pt_val}")

        oscar_rsts.append(rst)
        oscar_mins.append(rst[0])
        
        if check:
            break

        # print("---------------------------")

    print(f"# higher than global bp: {len(ids_higher_bp)} / {len(ids)}")
    print(ids_higher_bp)
    if isinstance(random_seed, int):
        print("============== start Random =============")
        # random_energies, random_inits = \
        #     get_random_initialization(G, p, n_pts, noise_model,
        #         data['beta_bound'], data['gamma_bound'], maxiter)
        random_rsts, random_inits = \
            get_random_initialization(G, p, n_pts if not check else 1, noise_model,
                data['beta_bound'], data['gamma_bound'], maxiter, random_seed)
        random_mins = [rst[0] for rst in random_rsts]
    else:
        random_rsts, random_inits, random_mins = None, None, None

    timestamp = get_curr_formatted_timestamp()
    save_dir = f"figs/find_init_pts_by_recon/{timestamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}/init-maxiter={maxiter}-eps={eps}-{data_fname}"
    print("initial points data saved to ", save_path)
    
    np.savez_compressed(
        save_path,
        eps=eps,
        C_opt=C_opt, # if QAOAKit does not have it, None
        offset=offset,
        maxiter=maxiter,
        
        # about reconstruction
        recon_init_pt_vals=recon_init_pt_vals,
        origin_init_pt_vals=origin_init_pt_vals,
        min_recon=min_recon,

        # OSCAR method
        oscar_rsts=oscar_rsts,
        oscar_mins=oscar_mins,
        oscar_inits=oscar_inits, # parameters, bbgg
        oscar_init_ids=ids, # indices

        # random initialization method
        random_mins=random_mins,
        random_rsts=random_rsts,
        random_inits=random_inits, # parameters, bbgg
        
        # others
        ids_higher_bp=ids_higher_bp,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help="Number of qubits", required=True)
    parser.add_argument('-p', type=int, help="QAOA layers", required=True)
    parser.add_argument('--noise', type=str, help="Noise type", required=True)
    parser.add_argument('--p1', type=float, help="Depolar")
    parser.add_argument('--p2', type=float, help="Depolar")
    parser.add_argument('--maxiter', type=int, help="Depolar", default=None)
    parser.add_argument('--check', action='store_true', default=False)
    parser.add_argument('--eps', type=float, nargs="+", required=True)
    parser.add_argument('--inst_seed', type=int, help="seed of problem instance", required=True)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--stride', type=int)

    args = parser.parse_args()
    
    # test_qiskit_qaoa_circuit()
    # exit()
    find_good_initial_points_on_recon_LS_and_verify_top(
        n_qubits=args.n, p=args.p, noise=args.noise, p1=args.p1, p2=args.p2,
        maxiter=args.maxiter, eps=args.eps,
        instance_seed=args.inst_seed, random_seed=args.random_seed, stride=args.stride, check=args.check
    )