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
    save_partial_qaoa_dataset_table
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
    multi_landscapes_and_cnt_optima_and_mitiq_and_MP_and_one_variable_and_CS
)


# from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz

test_utils_folder = Path(__file__).parent


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
        for sf in np.arange(0.1, 0.3, 0.03):
            for n_qubits in range(8, 9, 2): # [1, 16]
            # for n_qubits in range(4, 5): # [1, 16]
                for p in range(2, 3): # [1, 11]
                    start_time = time.time()
                    df = reg3_dataset_table.reset_index()
                    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
                    # df = df.iloc[0: 1]
                    df = df.iloc[0: MAX_NUM_GRAPHS_PER_NUM_QUBITS]
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
    count_optima_of_fixed_angles_3reg_graphs_one_variable_CS_sampling_frac()