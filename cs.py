from random import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
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
    one_D_CS_p1_recon_with_given_landscapes_and_varing_sampling_frac,
    two_D_CS_p1_recon_with_given_landscapes,
    _vis_one_D_p1_recon,
    p1_generate_grad,
    _mitiq_executor_of_qaoa_maxcut_energy
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


def one_D_CS_p1_generate_landscapes():
    """Count optima for regular 3 graphs with p>=3.

    In QAOAKit (or more specifically, https://github.com/danlkv/fixed-angle-QAOA),
    use fixed angles to solve 3 regular graphs.
    These approximation ratios are well-bounded, so we take them as results.
    """

    MAX_NUM_GRAPHS_PER_NUM_QUBITS = 10
    count_rst_df = pd.DataFrame(
        columns=[
            'row_id', 'G', 'pynauty_cert',
            'n_qubits', 'p',
            'miti_n_opt_list',
            'unmiti_n_opt_list',
            'has_opt',
            'C_opt'
        ]
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    for n_qubits in range(10, 11, 2): # [1, 16]
    # for n_qubits in range(4, 5): # [1, 16]
        for p in range(1, 2): # [1, 11]
            start_time = time.time()
            df = reg3_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.iloc[0: 2]
            # df = df.iloc[0: MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            # df = df.iloc[MAX_NUM_GRAPHS_PER_NUM_QUBITS:2*MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            
            print(f"n_qubits={n_qubits}")
            print(f"num of graphs: {len(df)}")
            print(f"p={p}")
            for row_id, row in df.iterrows():
                
                print(f"handling {row_id}")
                angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

                # print(row["beta"])
                # print(row["gamma"])
                # print(angles)
                # print(row["p_max"])
                # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                C_opt = row["C_fixed"]
                print("C_fixed", C_opt)
                G = row["G"]

                figdir = f'figs/cnt_opt_miti/{signature}/G{row_id}_nQ{n_qubits}_p{p}'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}.png")
                plt.cla()

                # n_optima_list = vis_landscape_multi_p_and_and_count_optima(
                miti_n_opt_list, unmiti_n_opt_list = \
                    one_D_CS_p1_generate_landscape(
                    # multi_landscapes_and_cnt_optima_and_mitiq_and_MP_and_one_variable_and_CS(
                # miti_n_opt_list, unmiti_n_opt_list = vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable(
                        G=G,
                        p=p,
                        figdir=figdir, 
                        # beta_opt=beta_to_qaoa_format(angles["beta"]),
                        # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                        beta_opt=angles["beta"],
                        gamma_opt=angles["gamma"],
                        noise_model=None,
                        params_path=[],
                        C_opt=C_opt
                )

                print('miti_n_opt_list', miti_n_opt_list)
                print('unmiti_n_opt_list', unmiti_n_opt_list)

                # count_rst_df = count_rst_df.append({
                #     'row_id': row_id,
                #     'G': G,
                #     'pynauty_cert': row['pynauty_cert'],
                #     'n_qubits': n_qubits,
                #     'p': p,
                #     'miti_n_opt_list': miti_n_opt_list,
                #     'unmiti_n_opt_list': unmiti_n_opt_list,
                #     'has_opt': False,
                #     'C_opt': C_opt,
                # }, ignore_index=True)

                # print(count_rst_df)
                # count_rst_df.to_pickle(f"cnt_opt_miti_df/{signature}_cnt_opt_fixed_angles.p")
                print(" ================ ")

            end_time = time.time()
            print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return

def count_optima_of_fixed_angles_3reg_graphs_one_variable_CS_sampling_frac():
    """Count optima for regular 3 graphs with p>=3.

    In QAOAKit (or more specifically, https://github.com/danlkv/fixed-angle-QAOA),
    use fixed angles to solve 3 regular graphs.
    These approximation ratios are well-bounded, so we take them as results.
    """

    MAX_NUM_GRAPHS_PER_NUM_QUBITS = 10
    count_rst_df = pd.DataFrame(
        columns=[
            'row_id', 'G', 'pynauty_cert',
            'n_qubits', 'p',
            'miti_n_opt_list',
            'unmiti_n_opt_list',
            'has_opt',
            'C_opt'
        ]
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sf in np.arange(0.05, 0.3, 0.03):
            for n_qubits in range(8, 9, 2): # [1, 16]
            # for n_qubits in range(4, 5): # [1, 16]
                for p in range(2, 3): # [1, 11]
                    start_time = time.time()
                    df = reg3_dataset_table.reset_index()
                    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
                    df = df.iloc[0: 1]
                    # df = df.iloc[0: MAX_NUM_GRAPHS_PER_NUM_QUBITS]
                    # df = df.iloc[MAX_NUM_GRAPHS_PER_NUM_QUBITS:2*MAX_NUM_GRAPHS_PER_NUM_QUBITS]
                    
                    print(f"n_qubits={n_qubits}")
                    print(f"num of graphs: {len(df)}")
                    print(f"p={p}")
                    print(f"sampling frac={sf}")
                    for row_id, row in df.iterrows():
                        
                        print(f"handling {row_id}")
                        angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

                        # print(row["beta"])
                        # print(row["gamma"])
                        # print(angles)
                        # print(row["p_max"])
                        # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                        C_opt = row["C_fixed"]
                        print("C_fixed", C_opt)
                        G = row["G"]

                        figdir = f'figs/cnt_opt_miti/{signature}/G{row_id}_nQ{n_qubits}_p{p}_sf{sf:.3f}'
                        
                        if not os.path.exists(figdir):
                            os.makedirs(figdir)

                        nx.draw_networkx(G)
                        plt.title(f"")
                        plt.savefig(f"{figdir}/G{row_id}.png")
                        plt.cla()

                        # n_optima_list = vis_landscape_multi_p_and_and_count_optima(
                        miti_n_opt_list, unmiti_n_opt_list = \
                            multi_landscapes_and_cnt_optima_and_mitiq_and_MP_and_one_variable_and_CS(
                        # miti_n_opt_list, unmiti_n_opt_list = vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable(
                                G=G,
                                p=p,
                                figdir=figdir, 
                                # beta_opt=beta_to_qaoa_format(angles["beta"]),
                                # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                                beta_opt=angles["beta"],
                                gamma_opt=angles["gamma"],
                                noise_model=None,
                                params_path=[],
                                C_opt=C_opt,
                                executor=executor,
                                sampling_frac=sf
                            )

                        print('miti_n_opt_list', miti_n_opt_list)
                        print('unmiti_n_opt_list', unmiti_n_opt_list)

                        # count_rst_df = count_rst_df.append({
                        #     'row_id': row_id,
                        #     'G': G,
                        #     'pynauty_cert': row['pynauty_cert'],
                        #     'n_qubits': n_qubits,
                        #     'p': p,
                        #     'miti_n_opt_list': miti_n_opt_list,
                        #     'unmiti_n_opt_list': unmiti_n_opt_list,
                        #     'has_opt': False,
                        #     'C_opt': C_opt,
                        # }, ignore_index=True)

                        # print(count_rst_df)
                        # count_rst_df.to_pickle(f"cnt_opt_miti_df/{signature}_cnt_opt_fixed_angles.p")
                        print(" ================ ")

                    end_time = time.time()
                    print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return

def count_optima_of_fixed_angles_3reg_graphs_one_variable_CS():
    """Count optima for regular 3 graphs with p>=3.

    In QAOAKit (or more specifically, https://github.com/danlkv/fixed-angle-QAOA),
    use fixed angles to solve 3 regular graphs.
    These approximation ratios are well-bounded, so we take them as results.
    """

    MAX_NUM_GRAPHS_PER_NUM_QUBITS = 10
    count_rst_df = pd.DataFrame(
        columns=[
            'row_id', 'G', 'pynauty_cert',
            'n_qubits', 'p',
            'miti_n_opt_list',
            'unmiti_n_opt_list',
            'has_opt',
            'C_opt'
        ]
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for n_qubits in range(8, 9, 2): # [1, 16]
        # for n_qubits in range(4, 5): # [1, 16]
            for p in range(2, 3): # [1, 11]
                start_time = time.time()
                df = reg3_dataset_table.reset_index()
                df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
                df = df.iloc[7: MAX_NUM_GRAPHS_PER_NUM_QUBITS]
                # df = df.iloc[0: 1]
                # df = df.iloc[MAX_NUM_GRAPHS_PER_NUM_QUBITS:2*MAX_NUM_GRAPHS_PER_NUM_QUBITS]
                
                print(f"n_qubits={n_qubits}")
                print(f"num of graphs: {len(df)}")
                print(f"p={p}")
                for row_id, row in df.iterrows():
                    
                    print(f"handling {row_id}")
                    angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

                    # print(row["beta"])
                    # print(row["gamma"])
                    # print(angles)
                    # print(row["p_max"])
                    # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                    C_opt = row["C_fixed"]
                    print("C_fixed", C_opt)
                    G = row["G"]

                    figdir = f'figs/cnt_opt_miti/{signature}/G{row_id}_nQ{n_qubits}_p{p}'
                    
                    if not os.path.exists(figdir):
                        os.makedirs(figdir)

                    nx.draw_networkx(G)
                    plt.title(f"")
                    plt.savefig(f"{figdir}/G{row_id}.png")
                    plt.cla()

                    # n_optima_list = vis_landscape_multi_p_and_and_count_optima(
                    miti_n_opt_list, unmiti_n_opt_list = \
                        multi_landscapes_and_cnt_optima_and_mitiq_and_MP_and_one_variable_and_CS(
                    # miti_n_opt_list, unmiti_n_opt_list = vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable(
                            G=G,
                            p=p,
                            figdir=figdir, 
                            # beta_opt=beta_to_qaoa_format(angles["beta"]),
                            # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                            beta_opt=angles["beta"],
                            gamma_opt=angles["gamma"],
                            noise_model=None,
                            params_path=[],
                            C_opt=C_opt,
                            executor=executor,
                            sampling_frac=1/3
                        )

                    print('miti_n_opt_list', miti_n_opt_list)
                    print('unmiti_n_opt_list', unmiti_n_opt_list)

                    # count_rst_df = count_rst_df.append({
                    #     'row_id': row_id,
                    #     'G': G,
                    #     'pynauty_cert': row['pynauty_cert'],
                    #     'n_qubits': n_qubits,
                    #     'p': p,
                    #     'miti_n_opt_list': miti_n_opt_list,
                    #     'unmiti_n_opt_list': unmiti_n_opt_list,
                    #     'has_opt': False,
                    #     'C_opt': C_opt,
                    # }, ignore_index=True)

                    # print(count_rst_df)
                    # count_rst_df.to_pickle(f"cnt_opt_miti_df/{signature}_cnt_opt_fixed_angles.p")
                    print(" ================ ")

                end_time = time.time()
                print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return


def one_D_CS_p1_recon_with_given_landscapes_top():
    data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)

    figdir = f"{data_dir}/1D_CS_recon_by_cvxpy"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for sf in np.arange(0.5, 0.55, 0.03):
        recon = one_D_CS_p1_recon_with_given_landscapes_and_varing_sampling_frac(
            figdir=figdir,
            origin=data['origin'].tolist(),
            full_range=data['full_range'].tolist(),
            n_pts=data['n_pts'].tolist(),
            sampling_frac=sf,
            alpha=0.1
        )


# ================== 2-D CS =================

def two_D_CS_p1_recon_with_given_landscapes_top():
    data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)

    figdir = f"{data_dir}/2D_CS_recon"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for sf in np.arange(0.05, 0.5, 0.03):
        recon = two_D_CS_p1_recon_with_given_landscapes(
            figdir=figdir,
            origin=data['origin'].tolist(),
            sampling_frac=sf
        )

# ================== optimize on 2-D CS reconstructed landscape ==============

def optimize_on_p1_reconstructed_landscape():
    # data_dir = "figs/cnt_opt_miti/2022-08-08_19:48:31"
    # data_dir = "figs/cnt_opt_miti/2022-08-09_16:49:38/G30_nQ8_p1"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()

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
                # SPSA
                ADAM
                # L_BFGS_B
            #     # COBYLA
            )(
                bounds=bounds, 
                landscape=landscape,
                fun_type='INTERPOLATE'
                # fun_type='None'
            )
            
            # optimizer = L_BFGS_B()

            # optimizer_name = 'SPSA'
            optimizer_name = "ADAM"
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


# ================== gradient ==============

def approximate_grad_by_2D_interpolation_top():

    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()

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
    
    bounds = data['bounds'].tolist()
    bounds = np.array([bounds['gamma'], bounds['beta']])
    
    x = np.array([[0.1, 0.3], [0.2, 0.4]])
    guess = approximate_fun_value_by_2D_interpolation(
        x=x,
        landscape=origin['ideals'],
        bounds=bounds
    )
    print(guess)
    # print(recon)

    # df = reg3_dataset_table.reset_index()
    # df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    # for row_id, row in df.iloc[1:2].iterrows():
    #     pass



def p1_generate_grad_top():
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    for n_qubits in range(8, 9, 2): # [1, 16]
    # for n_qubits in range(4, 5): # [1, 16]
        for p in range(1, 2): # [1, 11]
            start_time = time.time()
            df = reg3_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.iloc[1: 2]
            # df = df.iloc[0: MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            # df = df.iloc[MAX_NUM_GRAPHS_PER_NUM_QUBITS:2*MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            
            print(f"n_qubits={n_qubits}")
            print(f"num of graphs: {len(df)}")
            print(f"p={p}")
            for row_id, row in df.iterrows():
                
                print(f"handling {row_id}")
                angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

                # print(row["beta"])
                # print(row["gamma"])
                # print(angles)
                # print(row["p_max"])
                # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                C_opt = row["C_fixed"]
                print("C_fixed", C_opt)
                G = row["G"]

                figdir = f'figs/cnt_opt_miti/{signature}/G{row_id}_nQ{n_qubits}_p{p}_grad'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}.png")
                plt.cla()

                p1_generate_grad(
                    G=G,
                    p=p,
                    figdir=figdir, 
                    # beta_opt=beta_to_qaoa_format(angles["beta"]),
                    # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                    beta_opt=angles["beta"],
                    gamma_opt=angles["gamma"],
                    noise_model=None,
                    params_path=[],
                    C_opt=C_opt
                )

                print(" ================ ")

            end_time = time.time()
            print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return


def compare_with_original_grad_top():
    reg3_dataset_table = get_3_reg_dataset_table()

    grad_data_dir = "figs/cnt_opt_miti/2022-09-01_22:20:04/G40_nQ8_p1_grad"
    data_dir = "figs/cnt_opt_miti/2022-08-10_10:14:03/G40_nQ8_p1"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    grad_data = np.load(f"{grad_data_dir}/grad_data.npz", allow_pickle=True)

    grads = grad_data['grads'].tolist()
    origin = data['origin'].tolist()

    n_test = 1000

    n_qubits = 8
    p = 1
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
    # qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
    #     G, p
    # )

    bounds = data['bounds'].tolist()
    bounds = np.array([bounds['gamma'], bounds['beta']])
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        # SPSA
        ADAM
        # L_BFGS_B
    #     # COBYLA
    )(
        bounds=bounds, 
        landscape=origin['ideals'],
        fun_type='INTERPOLATE'
    )

    # eps = 1e-10
    # eps = 0.01
    eps = 0.1
    
    def _get_ideal_energy_by_qc(x):
        x = optimizer.qiskit_format_to_qaoa_format_arr(x)
        qc = get_maxcut_qaoa_circuit(
            G, gamma=x[:p], beta=x[p:], 
            # transpile_to_basis=True, save_state=False
        )
        energy = _mitiq_executor_of_qaoa_maxcut_energy(
            qc=qc,
            G=G,
            is_noisy=False,
            shots=2048
        )
        # energy = qaoa_maxcut_energy(G, gamma=x[:p], beta=x[p:])
        return -energy
        
    jac_appro = get_numerical_derivative(
        optimizer.approximate_fun_value, eps
    )

    total_norm = 0
    for i in range(n_test):
        x = np.random.rand(2)
        # print("x", x)

        # optimizer.approximate_fun_value(x)
        jac_ideal = get_numerical_derivative(
            _get_ideal_energy_by_qc, eps
        )

        grad_ideal = jac_ideal(x)
        grad_appro = jac_appro(x)

        # print(grad_ideal, grad_appro)
        norm = ((grad_ideal - grad_appro)**2).mean()
        print(f"norm={norm}")
        total_norm += norm

    total_norm /= n_test
    print("total norm", total_norm)

    # approximate_grad_by_2D_interpolation_top
    

    # _vis_one_D_p1_recon(
    #     origin_dict=origin,
    #     recon_dict=recon,
    #     full_range=data['full_range'].tolist(),
    #     bounds=data['bounds'].tolist(),
    #     true_optima=true_optima,
    #     title=f'{optimizer_name} with sampling fraction {sf:.3f}',
    #     save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}_{optimizer_name}.png',
    #     recon_params_path_dict=recon_params_path_dict,
    #     origin_params_path_dict=origin_params_path_dict
    # )

    pass

if __name__ == "__main__":
    # test_qiskit_qaoa_circuit()
    # test_noisy_qaoa_maxcut_energy()
    # test_optimization_method()
    # test_qiskit_qaoa_circuit_optimization()
    # test_removing_edges()
    # count_optimals()
    # count_optima_of_fixed_angles_3reg_graphs()
    # count_optima_of_specific_3reg_graph(nQ_range=[4,17])
    # print_graphs()
    # count_optima_of_fixed_angles_3reg_graphs()
    # count_optima_of_fixed_angles_3reg_graphs_one_variable_CS()
    # count_optima_of_fixed_angles_3reg_graphs_one_variable_CS_sampling_frac()
    # one_D_CS_p1_generate_landscapes()
    # one_D_CS_p1_recon_with_given_landscapes_top()
    # two_D_CS_p1_recon_with_given_landscapes_top()

    # optimize_on_p1_reconstructed_landscape()
    compare_with_original_grad_top()
    # p1_generate_grad_top()
    # approximate_grad_by_2D_interpolation_top()