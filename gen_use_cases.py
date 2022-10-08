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
    _executor_of_qaoa_maxcut_energy
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


def CS_on_google_data():
    data_types = ['sk', '3reg', 'grid']
    # data_types = ['sk']
    # data_types = ['3reg']
    # data_types = ['grid']
    timestamp = get_curr_formatted_timestamp()
    for data_type in data_types:
        data_path = f"google_data/Google_Data/google_{data_type}.npz"

        data = np.load(data_path, allow_pickle=True)

        betas = data['betas']
        gammas = data['gammas']
        energies = data['energies']
        bounds = data['bounds'].tolist()
        # print(bounds)

        df = pd.DataFrame(data={'Beta': betas, 'Gamma': gammas, 'E': energies})
        df = df.pivot(index='Gamma', columns='Beta', values='E')

        landscape = np.array(df)
        print("landscape shape: ", landscape.shape)

        figdir = f"./google_data/{timestamp}"
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        # print(np.linspace(bounds['beta'][0], bounds['beta'][1], landscape.shape[1]))
        full_range = {
            "beta": np.linspace(bounds['beta'][0], 
                bounds['beta'][1],
                landscape.shape[1]),
            "gamma": np.linspace(bounds['gamma'][0], bounds['gamma'][1], landscape.shape[0]),
        }

        recons = []
        times = []
        for sf in np.arange(0.05, 0.5, 0.03):
            start = timeit.timeit()
            recon = two_D_CS_p1_recon_with_given_landscapes(
                figdir=figdir,
                origin={'unmitis': landscape},
                full_range=full_range,
                sampling_frac=sf
            )
            end = timeit.timeit()
            print(f"start: {start}, end: {end}")
            times.append(end - start)
            recons.append(recon)

        np.savez_compressed(f"{figdir}/recons_{data_type}",
            recons=recons, origin=landscape, times=times)


def get_grid_points(bounds, n_samples_along_axis):
    # from qaoa format to angles to qiskit format
    xs = []
    qaoa_angles = []
    # for gamma in np.linspace(bounds['gamma'][0], bounds['gamma'][1], 10):
    #     for beta in np.linspace(bounds['beta'][0], bounds['beta'][1], 10):
    for beta in np.linspace(bounds['beta'][0], bounds['beta'][1], 10):
        for gamma in np.linspace(bounds['gamma'][0], bounds['gamma'][1], 10):
            qaoa_angles.append({
                'gamma': [gamma],
                'beta': [beta]
            })
            angles = angles_from_qaoa_format(
                gamma=np.array([gamma]),
                beta=np.array([beta])
            )
            # print(angles)
            x = angles_to_qiskit_format(angles)
            xs.append(x)

    xs = np.array(xs)
    # print(xs.shape)
    # print("var of points: ", np.var(xs, axis=0))
    return xs, qaoa_angles


if __name__ == "__main__":
    debug_existing_BP_top()