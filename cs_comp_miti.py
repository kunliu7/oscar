import argparse
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
from qiskit.algorithms.optimizers.optimizer import POINT
from mitiq.zne.zne import execute_with_zne
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
from QAOAKit.n_dim_cs import recon_4D_landscape_by_2D
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
    CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable,
    multi_landscapes_and_cnt_optima_and_mitiq_and_MP_and_one_variable_and_CS,
    one_D_CS_p1_generate_landscape,
    gen_p1_landscape,
    one_D_CS_p1_recon_with_given_landscapes_and_varing_sampling_frac,
    recon_2D_landscape,
    two_D_CS_p1_recon_with_given_landscapes,
    _vis_one_D_p1_recon,
    p1_generate_grad,
    _executor_of_qaoa_maxcut_energy,
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

from data_loader import get_recon_landscape, get_recon_pathname, load_grid_search_data

test_utils_folder = Path(__file__).parent


def debug_existing_BP_top():
    """Find use cases. Use CS to reconstruct landscape, and judge
    the reason of slow convergence is barren plateaus or not.

    Full and reconstructed landscapes are generated.
    """
    # derive origin full landscapes
    # data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    # data_dir = "figs/cnt_opt_miti/2022-08-09_16:49:38/G30_nQ8_p1"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    
    # derive reconstructed landscape
    recon_path = f"{data_dir}/2D_CS_recon.npz"
    recon = np.load(recon_path, allow_pickle=True)['recon'].tolist()

    # prepare figdir
    timestamp = get_curr_formatted_timestamp()
    figdir = f"{data_dir}/use_cases/{timestamp}"
    if not os.path.exists(figdir):
        os.makedirs(figdir)


    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 1
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    assert row_id == 40

    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # get_maxcut_qaoa_circuit(G, beta)
    
    noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
    qinst = AerSimulator(method="statevector") 
    # qinst = QuantumInstance(
    #     backend=AerSimulator(),
    #     noise_model=noise_model,
    #     shots=2048
    # )
    # sv1 = Statevector(backend.run(qc1).result().get_statevector())
    # angles2 = angles_to_qaoa_format(
    #     opt_angles_for_graph(row["G"], row["p_max"])
    # )
    # qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
    # sv2 = Statevector(backend.run(qc2).result().get_statevector())

    bounds = data['bounds'].tolist()
    bounds = np.array([bounds['gamma'], bounds['beta']])

    print(bounds)
    print(recon['ideals'].shape)

    # t = np.arange(2500)
    # # print(t[:10])
    # recon = np.sin(t).reshape(50, 50)
    # print(recon)

    # obj_val = -(sv1.expectation_value(C) + offset)
    opt_cut = row["C_opt"]
    
    counts = []
    values = []
    params = []
    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        # print('fuck')
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    initial_angles = {
        "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1])]),
        "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1])])
    }

    print(initial_angles)

    label = 'unmitis'
    recon_params_path_dict = {}
    origin_params_path_dict = {label: []}

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence
    
    # optimizer = SPSA()
    optimizer = ADAM()

    # optimizer = L_BFGS_B()
    # optimizer = AQGD()
    # optimizer = GradientDescent()
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
    
    optimizer_name = optimizer.__class__.__name__

    # initial_point = np.hstack([[1.0 for _ in range(p)], [-1.0 for _ in range(p)]])

    # true QAOA and true optimizer on real quantum circuits
    # qaoa = VQE(
    #     ansatz=qc1,
    #     optimizer=optimizer,
    #     initial_point=angles_to_qiskit_format(
    #         {"gamma": row["gamma"],
    #         "beta": row["beta"]}
    #     ),
    #     # initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
    #     quantum_instance=qinst,
    #     callback=cb_store_intermediate_result
    # )
    qaoa_angles = angles_to_qaoa_format(
            {"gamma": row["gamma"],
            "beta": row["beta"]}
        )

    eigenstate, eigenvalue, params = optimize_under_noise(
        G=G, 
        init_beta_gamma=np.concatenate(
            (initial_angles['beta'], initial_angles['gamma']),
        ),
        # init_beta_gamma=np.concatenate(
        #     (qaoa_angles['beta'], qaoa_angles['gamma'])
        # ),
        noise_model=None,
        num_shots=2048,
        opt_method='Nelder-Mead'
    )

    print(eigenvalue, eigenstate)
    # qaoa = QAOA(
    #     optimizer=optimizer,
    #     reps=p,
    #     initial_point=angles_to_qiskit_format(
    #         {"gamma": row["gamma"],
    #         "beta": row["beta"]}
    #     ),
    #     # initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
    #     quantum_instance=qinst,
    #     callback=cb_store_intermediate_result
    # )

    # qaoa.print_settings()

    # I = Operator(np.eye(2**n_qubits)) # , input_dims=n_qubits)
    # I = Pauli("I" * n_qubits)
    # print(C)
    # I = PrimitiveOp(I)
    # print(I)
    # result = qaoa.compute_minimum_eigenvalue(I)
    # result = qaoa.compute_minimum_eigenvalue(C)
    print(params[0], angles_from_qiskit_format(params[0]))
    # print(counts)
    # print(qaoa.optimal_params)
    print("opt_cut                     :", opt_cut)
    # print("recon landscape minimum     :", result.eigenvalue)
    # print("QAOA energy + offset        :", - (result.eigenvalue + offset))
    print("recon landscape minimum     :", eigenvalue)
    print("QAOA energy + offset        :", - (eigenvalue + offset))

    # params = optimizer.params_path


    # print(params)
    print("len of params:", len(params))
    # print("len of params:", len(optimizer.params_path))
    # opt_point = params[-1].copy()
    shifted_params = []
    for _param in params:
        _param = angles_from_qiskit_format(_param)

        _param = angles_to_qaoa_format(_param)

        # _param = shift_parameters(
        #     np.concatenate((_param['gamma'], _param['beta'])),
        #     bounds
        # )
        _param = [_param['gamma'][0], _param['beta'][0]]

        shifted_params.append(_param)
    
    params = shifted_params
    origin_params_path_dict[label] = params 
    print(len(params))
    # print(params)

    # record parameters
    np.savez_compressed(f"{figdir}/use_case_data",
        # initial_point=initial_angles,
        initial_angles=initial_angles,
        params_path=params
    )

    true_optima = np.concatenate([
        gamma_to_qaoa_format(row["gamma"]),
        beta_to_qaoa_format(row["beta"]),
    ])
    
    true_optima = shift_parameters(true_optima, bounds)

    _vis_one_D_p1_recon(
        origin_dict=origin,
        recon_dict=recon,
        full_range=data['full_range'].tolist(),
        bounds=data['bounds'].tolist(),
        true_optima=true_optima,
        title=f'{optimizer_name} with sampling fraction {sf:.3f}',
        save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}_{optimizer_name}.png',
        recon_params_path_dict=recon_params_path_dict,
        origin_params_path_dict=origin_params_path_dict
    )


# def minimize_on_qaoa_angles(
#     fun: Callable[[POINT], float], 
#     x0: POINT, 
#     jac: Optional[Callable[[POINT], POINT]] = None,
#     bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizerResult:
#     """Override existing optimizer's minimize function

#     """
#     # print(self.callback)
#     # print(super().callback)
#     res = super().minimize(query_fun_value_from_landscape, x0, jac, bounds)
#     print('res', res)
#     print(jac)
#     return res


def debug_existing_BP_top_2():
    """Find use cases. Use CS to reconstruct landscape, and judge
    the reason of slow convergence is barren plateaus or not.

    Full and reconstructed landscapes are generated.
    """
    # derive origin full landscapes
    # data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    # data_dir = "figs/cnt_opt_miti/2022-08-09_16:49:38/G30_nQ8_p1"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    
    # derive reconstructed landscape
    recon_path = f"{data_dir}/2D_CS_recon.npz"
    recon = np.load(recon_path, allow_pickle=True)['recon'].tolist()

    # prepare figdir
    timestamp = get_curr_formatted_timestamp()
    figdir = f"{data_dir}/use_cases/{timestamp}"
    if not os.path.exists(figdir):
        os.makedirs(figdir)


    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 1
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    assert row_id == 40

    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # get_maxcut_qaoa_circuit(G, beta)
    
    noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
    # qinst = AerSimulator(method="statevector") 
    qinst = QuantumInstance(
        backend=AerSimulator(),
        noise_model=noise_model,
        shots=2048
    )
    # sv1 = Statevector(backend.run(qc1).result().get_statevector())
    # angles2 = angles_to_qaoa_format(
    #     opt_angles_for_graph(row["G"], row["p_max"])
    # )
    # qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
    # sv2 = Statevector(backend.run(qc2).result().get_statevector())

    bounds = data['bounds'].tolist()
    bounds = np.array([bounds['gamma'], bounds['beta']])

    print(bounds)
    print(recon['ideals'].shape)

    # t = np.arange(2500)
    # # print(t[:10])
    # recon = np.sin(t).reshape(50, 50)
    # print(recon)

    # obj_val = -(sv1.expectation_value(C) + offset)
    opt_cut = row["C_opt"]
    
    counts = []
    values = []
    params = []
    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        # print('fuck')
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    initial_angles = {
        "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1])]),
        "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1])])
    }
    
    initial_angles = {
        "gamma": np.array([1.403]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.45]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.57]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.62]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.62]),
        "beta": np.array([0.78])
    }
    
    # initial_angles = {
    #     "gamma": np.array([1.62]),
    #     "beta": np.array([0.0])
    # }
    

    print(initial_angles)

    label = 'unmitis'
    recon_params_path_dict = {}
    origin_params_path_dict = {label: []}

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence
    
    # optimizer = SPSA()
    # optimizer = ADAM()
    shots = 2048

    def _partial_qaoa_energy(x): # qaoa angles
        # angles = angles_from_qiskit_format(x)
        # angles = angles_to_qaoa_format(angles)

        # x = np.concatenate([angles['gamma'], angles['beta']])
        # print("!!!", x)

        circuit = get_maxcut_qaoa_circuit(
        G, beta=[x[1]], gamma=[x[0]],
        # transpile_to_basis=True, save_state=False)
        transpile_to_basis=False, save_state=False)

        miti = execute_with_zne(
            circuit,
            executor=partial(
                _executor_of_qaoa_maxcut_energy, G=G, noise_model=noise_model, shots=shots),
        )
        
        return miti

    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        # SPSA
        ADAM
        # L_BFGS_B
    #     # COBYLA
    )(
        bounds=None,
        landscape=None,
        fun_type='FUN',
        fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        # lr=1e-2,
        # beta_1=0.9,
        # beta_2=0.99,
        # noise_factor=1e-8,
        # eps=1e-10,
        # eps=1e-3,
    )

    # optimizer = L_BFGS_B()
    # optimizer = AQGD()
    # optimizer = GradientDescent()
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
    
    optimizer_name = optimizer.__class__.__name__

    # initial_point = np.hstack([[1.0 for _ in range(p)], [-1.0 for _ in range(p)]])

    # true QAOA and true optimizer on real quantum circuits
    # qaoa = VQE(
    #     ansatz=qc1,
    #     optimizer=optimizer,
    #     initial_point=angles_to_qiskit_format(
    #         {"gamma": row["gamma"],
    #         "beta": row["beta"]}
    #     ),
    #     # initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
    #     quantum_instance=qinst,
    #     callback=cb_store_intermediate_result
    # )
    qaoa_angles = angles_to_qaoa_format(
        {"gamma": row["gamma"],
        "beta": row["beta"]}
    )

    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        # initial_point=angles_to_qiskit_format(
        #     {"gamma": row["gamma"],
        #     "beta": row["beta"]}
        # ),
        initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
        quantum_instance=qinst,
        callback=cb_store_intermediate_result
    )

    qaoa.print_settings()

    # I = Operator(np.eye(2**n_qubits)) # , input_dims=n_qubits)
    # I = Pauli("I" * n_qubits)
    # print(C)
    # I = PrimitiveOp(I)
    # print(I)
    # result = qaoa.compute_minimum_eigenvalue(I)
    result = qaoa.compute_minimum_eigenvalue(C)

    params = optimizer.params_path
    # print(params[0], angles_from_qiskit_format(params[0]))
    # print(counts)
    # print(qaoa.optimal_params)
    print("opt_cut                     :", opt_cut)
    print("recon landscape minimum     :", result.eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))
    print("len of params:", len(params))
    
    shifted_params = []
    for _param in params:
        # _param = angles_from_qiskit_format(_param)

        # _param = angles_to_qaoa_format(_param)

        _param = shift_parameters(
            np.array([_param[0], _param[1]]), # gamma, beta
            bounds # bounds = np.array([bounds['gamma'], bounds['beta']])
        )
        # _param = [_param['gamma'][0], _param['beta'][0]]

        shifted_params.append(_param)
    
    params = shifted_params
    origin_params_path_dict[label] = params 
    print(len(params))
    # print(params)

    # record parameters
    np.savez_compressed(f"{figdir}/use_case_data",
        # initial_point=initial_angles,
        initial_angles=initial_angles,
        params_path=params
    )

    true_optima = np.concatenate([
        gamma_to_qaoa_format(row["gamma"]),
        beta_to_qaoa_format(row["beta"]),
    ])
    
    true_optima = shift_parameters(true_optima, bounds)

    _vis_one_D_p1_recon(
        origin_dict=origin,
        recon_dict=recon,
        full_range=data['full_range'].tolist(),
        bounds=data['bounds'].tolist(),
        true_optima=true_optima,
        title=f'{optimizer_name} with sampling fraction {sf:.3f}',
        save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}_{optimizer_name}.png',
        recon_params_path_dict=recon_params_path_dict,
        origin_params_path_dict=origin_params_path_dict
    )


def debug_existing_BP_top_3():
    """Find use cases. Use CS to reconstruct landscape, and judge
    the reason of slow convergence is barren plateaus or not.

    Full and reconstructed landscapes are generated.
    """
    # derive origin full landscapes
    # data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    # data_dir = "figs/cnt_opt_miti/2022-08-09_16:49:38/G30_nQ8_p1"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    
    # derive reconstructed landscape
    recon_path = f"{data_dir}/2D_CS_recon.npz"
    recon = np.load(recon_path, allow_pickle=True)['recon'].tolist()

    # prepare figdir
    timestamp = get_curr_formatted_timestamp()
    figdir = f"{data_dir}/use_cases/{timestamp}"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 1
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    assert row_id == 40

    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # get_maxcut_qaoa_circuit(G, beta)
    
    noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
    # qinst = AerSimulator(method="statevector") 
    qinst = QuantumInstance(
        backend=AerSimulator(),
        noise_model=noise_model,
        shots=2048
    )
    # sv1 = Statevector(backend.run(qc1).result().get_statevector())
    # angles2 = angles_to_qaoa_format(
    #     opt_angles_for_graph(row["G"], row["p_max"])
    # )
    # qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
    # sv2 = Statevector(backend.run(qc2).result().get_statevector())

    bounds = data['bounds'].tolist()
    bounds = np.array([bounds['gamma'], bounds['beta']])

    print(bounds)
    print(recon['ideals'].shape)

    # t = np.arange(2500)
    # # print(t[:10])
    # recon = np.sin(t).reshape(50, 50)
    # print(recon)

    # obj_val = -(sv1.expectation_value(C) + offset)
    opt_cut = row["C_opt"]
    
    counts = []
    values = []
    params = []
    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        # print('fuck')
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    initial_angles = {
        "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1])]),
        "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1])])
    }
    
    initial_angles = {
        "gamma": np.array([1.403]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.45]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.57]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.62]),
        "beta": np.array([0.679])
    }
    
    initial_angles = {
        "gamma": np.array([1.62]),
        "beta": np.array([0.78])
    }
    
    initial_angles = {
        "gamma": np.array([1.62]),
        "beta": np.array([0.0])
    }
    
    initial_angles = {
        "gamma": np.array([1.62]),
        "beta": np.array([-0.5])
    }
    
    initial_angles = {
        "gamma": np.array([-1.62]),
        "beta": np.array([0.5])
    }
    
    # lr=1e-2
    initial_angles = {
        "gamma": np.array([-1.62]),
        "beta": np.array([0.4])
    }
    
    initial_angles = {
        "gamma": np.array([-0.7]),
        "beta": np.array([-0.45])
    }
    

    print(initial_angles)

    label = 'unmitis'
    recon_params_path_dict = {}
    origin_params_path_dict = {label: []}

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence
    
    # optimizer = SPSA()
    # optimizer = ADAM()

    def _partial_qaoa_energy(x):
        # angles = angles_from_qiskit_format(x)
        # angles = angles_to_qaoa_format(angles)

        # x = np.concatenate([angles['gamma'], angles['beta']])
        # print("!!!", x)
        return -noisy_qaoa_maxcut_energy(
            G=G, beta=[x[1]], gamma=[x[0]], precomputed_energies=None, noise_model=None
        )

    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        # SPSA
        ADAM
        # L_BFGS_B
    #     # COBYLA
    )(
        bounds=None,
        landscape=None,
        fun_type='FUN',
        fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        maxiter=1000,
        # tol=1e-7,
        lr=5e-2,
        # lr=1e-2,
        # beta_1=0.9,
        # beta_2=0.99,
        # noise_factor=1e-8,
        # eps=1e-10,
        # eps=1e-3,
        # amsgrad=True
    )

    # optimizer = L_BFGS_B()
    # optimizer = AQGD()
    # optimizer = GradientDescent()
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
    
    optimizer_name = optimizer.__class__.__name__

    # initial_point = np.hstack([[1.0 for _ in range(p)], [-1.0 for _ in range(p)]])

    # true QAOA and true optimizer on real quantum circuits
    # qaoa = VQE(
    #     ansatz=qc1,
    #     optimizer=optimizer,
    #     initial_point=angles_to_qiskit_format(
    #         {"gamma": row["gamma"],
    #         "beta": row["beta"]}
    #     ),
    #     # initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
    #     quantum_instance=qinst,
    #     callback=cb_store_intermediate_result
    # )
    qaoa_angles = angles_to_qaoa_format(
        {"gamma": row["gamma"],
        "beta": row["beta"]}
    )

    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        # initial_point=angles_to_qiskit_format(
        #     {"gamma": row["gamma"],
        #     "beta": row["beta"]}
        # ),
        initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
        quantum_instance=qinst,
        callback=cb_store_intermediate_result
    )

    qaoa.print_settings()

    # I = Operator(np.eye(2**n_qubits)) # , input_dims=n_qubits)
    # I = Pauli("I" * n_qubits)
    # print(C)
    # I = PrimitiveOp(I)
    # print(I)
    # result = qaoa.compute_minimum_eigenvalue(I)
    result = qaoa.compute_minimum_eigenvalue(C)

    params = optimizer.params_path
    # print(params[0], angles_from_qiskit_format(params[0]))
    # print(counts)
    # print(qaoa.optimal_params)
    print("opt_cut                     :", opt_cut)
    print("recon landscape minimum     :", result.eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))
    print("len of params:", len(params))
    
    shifted_params = []
    for _param in params:
        # _param = angles_from_qiskit_format(_param)

        # _param = angles_to_qaoa_format(_param)

        _param = shift_parameters(
            np.array([_param[0], _param[1]]), # gamma, beta
            bounds # bounds = np.array([bounds['gamma'], bounds['beta']])
        )
        # _param = [_param['gamma'][0], _param['beta'][0]]

        shifted_params.append(_param)
    
    params = shifted_params
    origin_params_path_dict[label] = params 
    print(len(params))
    # print(params)

    # record parameters
    np.savez_compressed(f"{figdir}/use_case_data",
        # initial_point=initial_angles,
        initial_angles=initial_angles,
        params_path=params
    )

    true_optima = np.concatenate([
        gamma_to_qaoa_format(row["gamma"]),
        beta_to_qaoa_format(row["beta"]),
    ])
    
    true_optima = shift_parameters(true_optima, bounds)

    _vis_one_D_p1_recon(
        origin_dict=origin,
        recon_dict=recon,
        full_range=data['full_range'].tolist(),
        bounds=data['bounds'].tolist(),
        true_optima=true_optima,
        title=f'{optimizer_name} with sampling fraction {sf:.3f}',
        save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}_{optimizer_name}.png',
        recon_params_path_dict=recon_params_path_dict,
        origin_params_path_dict=origin_params_path_dict
    )


def optimize_on_p1_reconstructed_landscape():
    # data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    # data_dir = "figs/cnt_opt_miti/2022-08-09_16:49:38/G30_nQ8_p1"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    existed = True
    # existed = False
    # if not existed:
    timestamp = get_curr_formatted_timestamp()
    figdir = f"{data_dir}/2D_CS_recon"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 1
    sf = 0.05
    # recon = two_D_CS_p1_recon_with_given_landscapes(
    #     figdir=figdir,
    #     origin=data['origin'].tolist(),
    #     sampling_frac=sf
    # )
    # np.savez_compressed(f'{figdir}', recon=recon)

    # return
    recon = np.load(f'{figdir}.npz', allow_pickle=True)['recon'].tolist()
    # print(recon)

    df = reg3_dataset_table.reset_index()
    # print(df)
    # print(df.columns)
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass
    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )
    backend = AerSimulator(method="statevector")
    # sv1 = Statevector(backend.run(qc1).result().get_statevector())
    # angles2 = angles_to_qaoa_format(
    #     opt_angles_for_graph(row["G"], row["p_max"])
    # )
    # qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
    # sv2 = Statevector(backend.run(qc2).result().get_statevector())

    # optimizer = GradientDescent()
    bounds = data['bounds'].tolist()

    bounds = np.array([bounds['gamma'], bounds['beta']])

    print(bounds)
    print(recon['ideals'].shape)

    # t = np.arange(2500)
    # # print(t[:10])
    # recon = np.sin(t).reshape(50, 50)
    # print(recon)

    # obj_val = -(sv1.expectation_value(C) + offset)
    opt_cut = row["C_opt"]
    
    counts = []
    values = []
    params = []
    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        # print('fuck')
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    initial_angles = {
        "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1])]),
        "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1])])
    }
    recon_params_path_dict = {}
    origin_params_path_dict = {}

    for label_type in ['origin', 'recon']:
        for label in origin.keys():
            if label_type == 'origin':
                landscape = origin[label]
            else:
                landscape = recon[label]

            counts = []
            values = []
            params = []

            optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
                SPSA
                # ADAM
                # L_BFGS_B
            #     # COBYLA
            )(
                bounds=bounds, 
                landscape=landscape,
                # fun_type='INTERPOLATE'
                fun_type='FUN',
                fun=None
                # fun_type='None'
            )
            
            # optimizer = L_BFGS_B()

            optimizer_name = 'SPSA'
            # optimizer_name = "ADAM"
            # optimizer_name = "L_BFGS_B"
            # optimizer_name = "COBYLA"

            # optimizer = SPSA()
            # optimizer = ADAM()
            # optimizer = SPSA()
            # optimizer = AQGD()
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
            qaoa = QAOA(
                optimizer=optimizer,
                reps=p,
                # initial_point=angles_to_qiskit_format(
                #     {"gamma": row["gamma"],
                #     "beta": row["beta"]}
                # ),
                initial_point=angles_to_qiskit_format(angles_from_qaoa_format(**initial_angles)),
                quantum_instance=backend,
                # callback=cb_store_intermediate_result
                )
            result = qaoa.compute_minimum_eigenvalue(C)
            # print(qaoa.optimal_params)
            print("opt_cut                     :", opt_cut)
            print("recon landscape minimum     :", result.eigenvalue)
            print("QAOA energy + offset:", - (result.eigenvalue + offset))

            params = optimizer.params_path
            # print(params)
            print("len of params:", len(params))
            # print("len of params:", len(optimizer.params_path))
            # opt_point = params[-1].copy()
            params = [
                # _params
                shift_parameters(_params, bounds)
                # angles_to_qaoa_format(angles_from_qiskit_format(_params))
                for _params in params
            ]
            # params.insert(0, np.concatenate([initial_angles['gamma'], initial_angles['beta']]))
            # params.insert(0, angles_to_qaoa_format(angles_from_qiskit_format(qaoa.optimal_params)))

            if label_type == 'origin':
                origin_params_path_dict[label] = params
            else: 
                recon_params_path_dict[label] = params

    true_optima = np.concatenate([
        gamma_to_qaoa_format(row["gamma"]),
        beta_to_qaoa_format(row["beta"]),
    ])
    
    true_optima = shift_parameters(true_optima, bounds)

    _vis_one_D_p1_recon(
        origin_dict=origin,
        recon_dict=recon,
        full_range=data['full_range'].tolist(),
        bounds=data['bounds'].tolist(),
        true_optima=true_optima,
        title=f'{optimizer_name} with sampling fraction {sf:.3f}',
        save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}_{optimizer_name}.png',
        recon_params_path_dict=recon_params_path_dict,
        origin_params_path_dict=origin_params_path_dict
    )


def gen_p1_landscape_with_other_miti_top():
    """Qiskit==0.37.2, mitiq==0.18.0.

        https://mitiq.readthedocs.io/en/stable/changelog.html
    """

    MAX_NUM_GRAPHS_PER_NUM_QUBITS = 10
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    
    for n_qubits in range(8, 9, 2): # [1, 16]
    # for n_qubits in range(4, 5): # [1, 16]
        for p in range(1, 2): # [1, 11]
            start_time = time.time()
            df = reg3_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.iloc[1: 2] # row_id = 40


            # df = df.iloc[0: MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            # df = df.iloc[MAX_NUM_GRAPHS_PER_NUM_QUBITS:2*MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            
            print(f"n_qubits={n_qubits}")
            print(f"num of graphs={len(df)}")
            print(f"p={p}")

            p1Q = 0.001
            p2Q = 0.005

            # p1Q = 0.003
            # p2Q = 0.007

            print(f"depolarizing error, p1Q={p1Q}, p2Q={p2Q}")
            noise_model = get_depolarizing_error_noise_model(p1Q=p1Q, p2Q=p2Q)
            for row_id, row in df.iterrows():
                
                print(f"handling graph with row_id={row_id}")
                angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

                # print(row["beta"])
                # print(row["gamma"])
                # print(angles)
                # print(row["p_max"])
                # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                C_opt = row["C_fixed"]
                print("C_fixed", C_opt)
                G = row["G"]

                figdir = f'figs/gen_p1_landscape/{signature}/G{row_id}_nQ{n_qubits}_p{p}_depolar{p1Q}_{p2Q}_zneLinear'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}.png")
                plt.cla()

                mitigation_params = {
                    'factory': LinearFactory(scale_factors=[1.0, 2.0])
                }

                print(mitigation_params)

                gen_p1_landscape(
                    G=G,
                    p=p,
                    figdir=figdir, 
                    # beta_opt=beta_to_qaoa_format(angles["beta"]),
                    # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                    beta_opt=angles["beta"],
                    gamma_opt=angles["gamma"],
                    noise_model=noise_model,
                    params_path=[],
                    C_opt=C_opt,
                    mitigation_params=mitigation_params,
                    # n_shots=2048,
                    # n_pts_per_unit=2
                )

                print(" ================ ")

            end_time = time.time()
            print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return


def cal_multi_errors(a, b):
    diff = {}
    a = a.reshape(-1)
    b = b.reshape(-1)
    # diff['L2-norm'] = np.linalg.norm(a - b)
    diff['SqrtMSE'] = cal_recon_error(a, b, 'MSE')
    # diff['1-NCC'] = 1 - cal_recon_error(a, b, "CROSS_CORRELATION")
    diff['COS'] = cosine(a, b)
    return diff


def cal_gap(C_opt, full, recon):
    min_full = np.max(full)
    min_recon = np.max(recon)
    print("C_opt: ", C_opt)
    print(f"min_full: {min_full}, C_opt - min_full: {C_opt - min_full}")
    print(f"min_miti_recon: {min_recon}, C_opt - min_recon: {C_opt - min_recon}")


# def _get_recon_landscape(p: int, origin: np.ndarray, sampling_frac: float, is_reconstructed: bool,
#     recon_save_path: str, cs_seed: int
# ) -> np.ndarray:
#     save_dir = os.path.dirname(recon_save_path)
#     is_recon = os.path.exists(recon_save_path)
#     if not is_recon:
#         np.random.seed(cs_seed)
#         if p == 1:
#             recon = recon_2D_landscape(
#                 origin=origin,
#                 sampling_frac=sampling_frac
#             )
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             np.savez_compressed(recon_save_path, recon=recon, sampling_frac=sampling_frac) 
#         elif p == 2:
#             recon = recon_4D_landscape_by_2D(
#                 origin=origin,
#                 sampling_frac=sampling_frac
#             )
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             np.savez_compressed(recon_save_path, recon=recon, sampling_frac=sampling_frac)
#         print("not exists, save to", save_dir)
#     else:
#         recon = np.load(recon_save_path, allow_pickle=True)['recon']
#         print("read from", save_dir)
    
#     return recon


# def check_improvement_of_mitigation(noisy, miti, metrics):

def smoothness(ls) -> list:
    """
    sd(diff(x))/abs(mean(diff(x)))
    sd(diff(y))/abs(mean(diff(y)))
    """

    grad = np.gradient(ls)
    std_of_grad = [g.std() for g in grad]
    abs_mean = [np.abs(g.mean()) for g in grad]

    std_of_grad = np.array(std_of_grad)
    abs_mean = np.array(abs_mean)

    smoothness = std_of_grad / (abs_mean + 0.01)
    return smoothness.mean()


def metric_smoothness(ls1, ls2) -> float:
    return smoothness(ls1) - smoothness(ls2)


def metric_variance(ls1, ls2) -> float:
    margin = np.var(ls1) - np.var(ls2)
    return margin


def var_of_grad(ls) -> list:
    grad = np.gradient(ls)

    var_of_grad = [g.var() for g in grad]
    return var_of_grad


def metric_barren_plateaus(ls1, ls2) -> float:
    var_grad1 = var_of_grad(ls1)
    var_grad2 = var_of_grad(ls2)

    mean1 = np.mean(var_grad1)
    mean2 = np.mean(var_grad2)
    return mean1 - mean2

# def metric(ls1, ls2) -> Tuple[bool, float]:

def compare_by_matrics(ls1, ls2, metrics):
    metric_vals = [met(ls1, ls2) for met in metrics]
    metric_vals = np.array(metric_vals)
    return metric_vals


def compare_with_ideal_landscapes(ideal, ls1, ls2):
    diff1 = ideal - ls1
    diff2 = ideal - ls2

    ids = np.abs(diff1) < np.abs(diff2)     # indices that Richardson is closer to ideal than linear

    # print("n_pts =", np.sum(ids))
    print("abs(diff1) <  abs(diff2), num of such points =", np.sum(ids == True))
    print("abs(diff1) >= abs(diff2), num of such points =", np.sum(ids == False))
    
    print("var diff1 =", np.var(diff1))
    print("var diff2 =", np.var(diff2))

    print("var abs(diff1) =", np.var(np.abs(diff1)))
    print("var abs(diff2) =", np.var(np.abs(diff2)))




def configurate(ideal, noisy, miti):
    pass


def benchmark():
    pass


def vis_case_compare_mitigation_method(check: bool=False):
    is_reconstructed = True 
    is_test = check

    method = 'sv'
    problem = 'maxcut'
    miti_method1 = 'zne-RichardsonFactory'
    miti_method2 = 'zne-LinearFactory'

    # noise-3
    p1 = 0.001
    p2 = 0.02
    n_qubits = 16

    noise = f'depolar-{p1}-{p2}'

    # n_qubits = 16
    cs_seed = n_qubits
    p = 1
    sf = 0.05
    seed = 0
    if p == 2:
        bs = 12
        gs = 15
    elif p == 1:
        bs = 50
        gs = 100
    else:
        raise ValueError("Invalid depth of QAOA")

    if is_test:
        ideal_data1, ideal_data_fname, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method1
        )
        ideal_data2, ideal_data_fname, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method2
        )
        full_range = ideal_data1['full_range']
        vis_landscapes(
            # landscapes=[origin['unmitis'], miti1, miti2, miti1_recon, miti2_recon],
            landscapes=[ideal_data1['data'], ideal_data2['data']],
            labels=[miti_method1, miti_method2],
            full_range=full_range,
            true_optima=None,
            title="Compare different ZNE configs and reconstruction",
            save_path="paper_figs/debug_miti.png",
            params_paths=[None, None]
        )
        return
    else:
        ideal_data, _, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, problem=problem, method=method,
            noise='ideal', beta_step=bs, gamma_step=gs, seed=seed
        )
        
        noisy_data, _, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed
        )

    full_range = ideal_data['full_range']

    ideal = ideal_data['data']
    noisy = noisy_data['data']

    # offset = ideal_data['offset']
    # full_ranges = data['full_ranges']
    # print("offset:", offset)

    # derive origin full landscape
    # data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    # data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    # origin = data['origin'].tolist()
    # full_range = data['full_range'].tolist()
    # miti1 = origin['mitis']
    # C_opt = data['C_opt']
    
    if not is_reconstructed:
        timestamp = get_curr_formatted_timestamp()
    else:
        timestamp = "2022-11-07_13:55:52_OK" # TODO

    # -------- derive miti1 data

    miti1_data, miti1_data_fname, _ = load_grid_search_data(
        n_qubits=n_qubits, p=p, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method1
    )
    miti1 = miti1_data['data']
    # mitigation_method1 = miti1_data['mitigation_method']
    # mitigation
    print(miti1_data['mitigation_method'], miti1_data['mitigation_config'])
    # print(mitigation_config1)

    # exit()
    recon1_path, _, _ = get_recon_pathname(p, problem, method, noise, cs_seed, sf, miti1_data_fname)
    # recon1_path = f"figs/recon_p2_landscape/{timestamp}/recon-sf={sf:.3f}-cs_seed={cs_seed}-{miti1_data_fname}"
    miti1_recon = get_recon_landscape(p, miti1, sf, is_reconstructed, recon1_path, cs_seed)

    # -------- derive miti2 data

    print("\n\n")

    miti2_data, miti2_data_fname, _ = load_grid_search_data(
        n_qubits=n_qubits, p=p, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method2
    )
    miti2 = miti2_data['data']

    recon2_path, _, _ = get_recon_pathname(p, problem, method, noise, cs_seed, sf, miti2_data_fname)
    # recon2_path = f"figs/recon_p2_landscape/{timestamp}/recon-sf={sf:.3f}-cs_seed={cs_seed}-{miti2_data_fname}"
    miti2_recon = get_recon_landscape(p, miti2, sf, is_reconstructed, recon2_path, cs_seed)
    
    print(miti2_data['mitigation_method'], miti2_data['mitigation_config'])
    
    # --------------- compare MSE, NCC and Cosine distance -----------

    # metrics = ['MSE', 'NCC', 'COS']
    # metrics = {"MSE": 0, "NCC": 0, "COS": 0}

    # diff1 = cal_multi_errors(miti1, miti1_recon)
    # diff2 = cal_multi_errors(miti2, miti2_recon)
    print("")
    print(f"ideal and {miti_method1} error")
    diff1 = cal_multi_errors(miti1, ideal)
    print(diff1)
    
    print(f"ideal and recon-{miti_method1} error")
    diff1 = cal_multi_errors(miti1_recon, ideal)
    print(diff1)

    print("")
    
    print(f"ideal and {miti_method2}")
    diff2 = cal_multi_errors(miti2, ideal)
    print(diff2)
    
    print(f"ideal and recon-{miti_method2}")
    diff2 = cal_multi_errors(miti2_recon, ideal)
    print(diff2)

    print("")
    
    print(f"ls1: Origin {miti_method1}, ls2: Origin {miti_method2}")
    print("diff1 = ideal - ls1, diff2 = ideal - ls2")
    compare_with_ideal_landscapes(ideal, miti1, miti2)
    
    print("")
    print(f"ls1: Recon {miti_method1}, ls2: Recon {miti_method2}")
    print("diff1 = ideal - ls1, diff2 = ideal - ls2")
    compare_with_ideal_landscapes(ideal, miti1_recon, miti2_recon)
    # save_path = f"figs/recon_2D"
    
    print("\n----- (1) Configuring  ZNE mitigation with OSCAR -----")
    metrics = [metric_barren_plateaus, metric_variance, metric_smoothness]
    print("metrics:", [m.__name__ for m in metrics])
    
    diff_origin = compare_by_matrics(miti1, miti2, metrics)
    diff_recon = compare_by_matrics(miti1_recon, miti2_recon, metrics)

    print(f"Origin {miti_method1}'s - Origin {miti_method2}'s:", diff_origin)
    print(f"Recon  {miti_method1}'s - Recon {miti_method2}'s:", diff_recon)
    
    print("\n----- (2) Benchmarking ZNE mitigation with OSCAR -----")
    # metrics = [metric_barren_plateaus, metric_variance]
    print("metrics:", [m.__name__ for m in metrics])
    # improvement of miti1
    imp_origin = compare_by_matrics(miti1, noisy, metrics)
    imp_recon  = compare_by_matrics(miti1_recon, noisy, metrics)
    
    print(f"Origin {miti_method1}'s - unmiti's:", imp_origin)
    print(f"Recon  {miti_method1}'s - unmiti's:", imp_recon)

    print("")
    # np.savez_compressed(
    #     miti1=miti1,
    #     miti2=miti2,
    #     ideal=ideal,
    #     miti1_recon=miti1_recon,
    #     miti2_recon=miti2_recon,
    #     recon1_path=recon1_path,
    #     recon2_path=recon2_path,
    # )

    # --------------- gap ------------

    # cal_gap(C_opt, miti1, miti1_recon)
    # cal_gap(C_opt, miti2, miti2_recon)

    # reg3_dataset_table = get_3_reg_dataset_table()
    # df = reg3_dataset_table.reset_index()
    # n_qubits = 8
    # p = 1
    # sf = 0.05
    # df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    # for row_id, row in df.iloc[1:2].iterrows():
    #     pass
    # assert row_id == 40

    vis_landscapes(
        # landscapes=[origin['unmitis'], miti1, miti2, miti1_recon, miti2_recon],
        landscapes=[miti1, miti2, miti1_recon, miti2_recon],
        labels=[miti_method1, miti_method2, f"recon-{miti_method1}", f"recon-{miti_method2}"],
        full_range=full_range,
        true_optima=None,
        title="Compare different ZNE configs and reconstruction",
        save_path="paper_figs/case3_debug_after_miti.png",
        params_paths=[None, None, None, None]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', default=False)
    args = parser.parse_args()
    vis_case_compare_mitigation_method(args.check)