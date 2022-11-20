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


def gradient_based_stuck():
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
    recon_path = f"{data_dir}/2D_CS_recon_sf0.050.npz"
    recon = np.load(recon_path, allow_pickle=True)['recon'].tolist()

    # prepare figdir
    timestamp = get_curr_formatted_timestamp()
    figdir = f"figs/opt_on_recon_landscape/{timestamp}"
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
    
    # noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
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
    
    # initial_angles = {
    #     "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1])]),
    #     "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1])])
    # }
    
    # initial_angles = {'gamma': np.array([-1.97728443]), 'beta': np.array([0.03140856])}
    print(initial_angles)

    label = 'unmitis'
    recon_params_path_dict = {}
    origin_params_path_dict = {label: []}

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence
    
    # optimizer = SPSA()
    # optimizer = ADAM()
    # raw_optimizer = ADAM
    raw_optimizer = SPSA

    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer
        # SPSA
        # ADAM
        # L_BFGS_B
    #     # COBYLA
    )(
        bounds=bounds,
        landscape=recon['unmitis'],
        fun_type='INTERPOLATE',
        # fun=_partial_qaoa_energy,
        fun=None
        
        # parameter of raw optimizer
        # lr=1e-2,
        # beta_1=0.9,
        # beta_2=0.99,
        # noise_factor=1e-8,
        # eps=1e-10,
        # eps=1e-3,
   )

    optimizer_name = raw_optimizer.__name__

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

        # _param = shift_parameters(
        #     np.array([_param[0], _param[1]]), # gamma, beta
        #     bounds # bounds = np.array([bounds['gamma'], bounds['beta']])
        # )
        # _param = [_param['gamma'][0], _param['beta'][0]]

        shifted_params.append(_param)
    
    params = shifted_params
    origin_params_path_dict[label] = params 
    print(len(params))
    # print(params)

    # record parameters
    np.savez_compressed(f"{figdir}/data",
        # initial_point=initial_angles,
        initial_angles=initial_angles,
        params_path=params
    )

    true_optima = np.concatenate([
        gamma_to_qaoa_format(row["gamma"]),
        beta_to_qaoa_format(row["beta"]),
    ])
    
    true_optima = shift_parameters(true_optima, bounds)

    recon_params_path_dict = {
        'mitis': params
    }

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
    diff['L2-norm'] = np.linalg.norm(a - b)
    diff['MSE'] = cal_recon_error(a, b, 'MSE')
    diff['1-NCC'] = 1 - cal_recon_error(a, b, "CROSS_CORRELATION")
    diff['COS'] = cosine(a, b)
    return diff


def vis_case_compare_mitigation_method():
    is_reconstructed = True

    # derive origin full landscape
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    full_range = data['full_range'].tolist()
    miti1 = origin['mitis']

    # ----------- tmp start
    # sf = 0.05
    # miti1_recon = two_D_CS_p1_recon_with_given_landscapes(
    #     figdir=None,
    #     origin=origin,
    #     full_range=None,
    #     sampling_frac=sf
    # )
    # recon1_path = f"{data_dir}/2D_CS_recon_sf{sf:.3f}"
    # np.savez_compressed(recon1_path, recon=miti1_recon)

    # return

    # ----------- tmp end

    # derive reconstructed landscape
    sf = 0.05
    recon_path = f"{data_dir}/2D_CS_recon_sf{sf:.3f}.npz"
    recon = np.load(recon_path, allow_pickle=True)['recon'].tolist()
    miti1_recon = recon['mitis']

    data2_dir = "figs/gen_p1_landscape/2022-10-08_16:52:53/G40_nQ8_p1_depolar0.001_0.005_zneLinear"
    miti2 = np.load(f"{data2_dir}/data.npz", allow_pickle=True)['origin'].tolist()['mitis']

    if not is_reconstructed:
        miti2_recon = two_D_CS_p1_recon_with_given_landscapes(
            figdir=None,
            origin={ "mitis": miti2 },
            full_range=None,
            sampling_frac=sf
        )
        recon2_path = f"{data2_dir}/2D_CS_recon_sf{sf:.3f}"
        np.savez_compressed(recon2_path, recon=miti2_recon, sampling_frac=sf)
        miti2_recon = miti2_recon['mitis']
    else:
        recon2_path = f"{data2_dir}/2D_CS_recon_sf{sf:.3f}.npz"
        miti2_recon = np.load(recon2_path, allow_pickle=True)['recon'].tolist()['mitis']

    # --------------- compare MSE, NCC and Cosine distance -----------

    # metrics = ['MSE', 'NCC', 'COS']
    # metrics = {"MSE": 0, "NCC": 0, "COS": 0}

    diff1 = cal_multi_errors(miti1, miti1_recon)
    diff2 = cal_multi_errors(miti2, miti2_recon)
    print(diff1)
    print(diff2)

    vis_landscapes(
        # landscapes=[origin['unmitis'], miti1, miti2, miti1_recon, miti2_recon],
        landscapes=[miti1, miti2, miti1_recon, miti2_recon],
        labels=["ZNE RichardsonFactory", "ZNE LinearFactory", "ZNE RichardsonFactory Recon", "ZNE LinearFactory Recon"],
        full_range=full_range,
        true_optima=None,
        title="Compare different ZNE configs and reconstruction",
        save_path="paper_figs/case3.png",
        params_paths=[None, None, None, None]
    )


if __name__ == "__main__":
    # debug_existing_BP_top()
    # debug_existing_BP_top_2()
    # debug_existing_BP_top_3()
    gradient_based_stuck()
    # optimize_on_p1_reconstructed_landscape()
    # gen_p1_landscape_with_other_miti_top()
    # vis_case_compare_mitigation_method()