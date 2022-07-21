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
    vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable
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

# from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz

test_utils_folder = Path(__file__).parent


def count_optima():
    """Count optimals for regular 3 graphs, only for p=1,2

    For p>=3, we do not have optimal parameters. We need to use fixed angles methods.
    """
    MAX_NUM_GRAPHS_PER_NUM_QUBITS = 10
    count_rst_df = pd.DataFrame(
        columns=['row_id', 'G', 'n_qubits', 'p', 'n_optima_list', 'has_opt']
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    signature = get_curr_formatted_timestamp()
    for n_qubits in range(4, 17): # only to 16
        for p in range(3, 7):
            df = reg3_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.dropna(how='any', subset=["C_opt", "beta", "gamma"])
            df = df.iloc[:MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            
            print(f"n_qubits={n_qubits}")
            print(f"num of graphs: {len(df)}")
            for row_id, row in df.iterrows():
                cert = row_id

                # print(type(row["beta"]))
                # is_valid_params = (True in np.isnan(row["beta"])) \
                #     and (True in np.isnan(row["gamma"]))
                # if not is_valid_params:
                #     continue

                # print(row["beta"])
                # print(row["gamma"])
                # print(row["p_max"])
                # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                has_opt = True # not np.isnan(row["C_opt"])
                if has_opt:
                    C_opt = row["C_opt"]
                else:
                    C_opt = row["C_fixed"]
                print("C_opt", C_opt)
                # C_opt = row["C_opt"]
                # if np.isnan(row["C_opt"]):
                #     C_opt = 
                
                print(f"handling {cert}")
                G = row["G"]
                
                # p = len(row["beta"])
                figdir = f'figs/count_optima/{signature}/G{cert}_nQ{n_qubits}_p{p}'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{cert}.png")
                plt.cla()

                n_optima_list = vis_landscape_multi_p_and_and_count_optima(
                    G=row["G"],
                    p=p,
                    figdir=figdir, 
                    beta_opt=beta_to_qaoa_format(row["beta"]),
                    gamma_opt=gamma_to_qaoa_format(row["gamma"]),
                    noise_model=None,
                    params_path=[],
                    C_opt=C_opt
                )

                count_rst_df = count_rst_df.append({
                    'row_id': cert,
                    'n_qubits': n_qubits,
                    'p': p,
                    'n_optima_list': n_optima_list,
                    'has_opt': has_opt,
                }, ignore_index=True)

                # print(count_rst_df)
                count_rst_df.to_pickle(f"{signature}_count_optima.p")
                print(" ================ ")

    return


def count_optima_of_fixed_angles_3reg_graphs_one_variable():
    """Count optima for regular 3 graphs with p>=3.

    In QAOAKit (or more specifically, https://github.com/danlkv/fixed-angle-QAOA),
    use fixed angles to solve 3 regular graphs.
    These approximation ratios are well-bounded, so we take them as results.
    """

    # df = get_3_reg_dataset_table().sample(n=10).reset_index()

    # for _, row in df.iterrows():
    #     angles = angles_to_qaoa_format(get_fixed_angles(3, row["p_max"]))
    #     assert np.isclose(
    #         qaoa_maxcut_energy(row["G"], angles["beta"], angles["gamma"]),
    #         row["C_fixed"],
    #         rtol=1e-4,
    #     )

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
        for n_qubits in range(8, 17, 2): # [1, 16]
        # for n_qubits in range(4, 5): # [1, 16]
            for p in range(2, 3): # [1, 11]
                start_time = time.time()
                df = reg3_dataset_table.reset_index()
                df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
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

                    figdir = f'figs/cnt_opt_miti/{signature}/G{row_id}_nQ{n_qubits}_p{p}'
                    
                    if not os.path.exists(figdir):
                        os.makedirs(figdir)

                    nx.draw_networkx(G)
                    plt.title(f"")
                    plt.savefig(f"{figdir}/G{row_id}.png")
                    plt.cla()

                    # n_optima_list = vis_landscape_multi_p_and_and_count_optima(
                    miti_n_opt_list, unmiti_n_opt_list = vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable(
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
                            executor=executor
                    )

                    print('miti_n_opt_list', miti_n_opt_list)
                    print('unmiti_n_opt_list', unmiti_n_opt_list)

                    count_rst_df = count_rst_df.append({
                        'row_id': row_id,
                        'G': G,
                        'pynauty_cert': row['pynauty_cert'],
                        'n_qubits': n_qubits,
                        'p': p,
                        'miti_n_opt_list': miti_n_opt_list,
                        'unmiti_n_opt_list': unmiti_n_opt_list,
                        'has_opt': False,
                        'C_opt': C_opt,
                    }, ignore_index=True)

                    # print(count_rst_df)
                    count_rst_df.to_pickle(f"cnt_opt_miti_df/{signature}_cnt_opt_fixed_angles.p")
                    print(" ================ ")

                end_time = time.time()
                print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

                # 4h for nQ=16, p=6
                # 100 min for nQ=12, p=6

    return


def count_optima_of_fixed_angles_3reg_graphs():
    """Count optima for regular 3 graphs with p>=3.

    In QAOAKit (or more specifically, https://github.com/danlkv/fixed-angle-QAOA),
    use fixed angles to solve 3 regular graphs.
    These approximation ratios are well-bounded, so we take them as results.
    """

    # df = get_3_reg_dataset_table().sample(n=10).reset_index()

    # for _, row in df.iterrows():
    #     angles = angles_to_qaoa_format(get_fixed_angles(3, row["p_max"]))
    #     assert np.isclose(
    #         qaoa_maxcut_energy(row["G"], angles["beta"], angles["gamma"]),
    #         row["C_fixed"],
    #         rtol=1e-4,
    #     )

    MAX_NUM_GRAPHS_PER_NUM_QUBITS = 10
    count_rst_df = pd.DataFrame(
        columns=[
            'row_id', 'G', 'pynauty_cert',
            'n_qubits', 'p',
            'miti_n_opt_list',
            'unmiti_n_opt_list',
            'has_opt'
        ]
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    for n_qubits in range(8, 17, 2): # [1, 16]
    # for n_qubits in range(4, 5): # [1, 16]
        for p in range(2, 7): # [1, 11]
            start_time = time.time()
            df = reg3_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
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
                # print(row["p_max"])
                # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                C_opt = row["C_fixed"]
                print("C_fixed", C_opt)
                G = row["G"]
                
                # p = len(row["beta"])
                figdir = f'figs/cnt_opt_miti/{signature}/G{row_id}_nQ{n_qubits}_p{p}'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}.png")
                plt.cla()

                # n_optima_list = vis_landscape_multi_p_and_and_count_optima(
                miti_n_opt_list, unmiti_n_opt_list = vis_multi_landscape_and_count_optima_and_mitiq_MP(
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

                count_rst_df = count_rst_df.append({
                    'row_id': row_id,
                    'G': G,
                    'pynauty_cert': row['pynauty_cert'],
                    'n_qubits': n_qubits,
                    'p': p,
                    'miti_n_opt_list': miti_n_opt_list,
                    'unmiti_n_opt_list': unmiti_n_opt_list,
                    'has_opt': False,
                }, ignore_index=True)

                # print(count_rst_df)
                count_rst_df.to_pickle(f"cnt_opt_miti_df/{signature}_cnt_opt_fixed_angles.p")
                print(" ================ ")

            end_time = time.time()
            print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

            # 4h for nQ=16, p=6
            # 100 min for nQ=12, p=6

    return


def count_optima_of_specific_3reg_graph(nQ_range):
    """Count optima for specific regular 3 graph.

    When graph is fixed, we could vary p and analyze n_optima_list.

    In QAOAKit (or more specifically, https://github.com/danlkv/fixed-angle-QAOA),
    use fixed angles to solve 3 regular graphs.
    These approximation ratios are well-bounded, so we take them as results.
    """
    
    assert len(nQ_range) == 2
    assert nQ_range[0] % 2 == 0

    df_path = "count_optima_for_one_graph_df"
    if not os.path.exists(df_path):
        os.makedirs(df_path)
        
    
    count_rst_df = pd.DataFrame(
        columns=['row_id', 'G', 'n_qubits', 'p', 'n_optima_list', 'has_opt']
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    signature = get_curr_formatted_timestamp()
    
    print("read 3 reg dataset OK")
    # print("use fixed angles to calculate n_optima")
    print("n qubits range: ", nQ_range)
    # print("p range: ", p_range)
    print("timestamp: ", signature)

    for n_qubits in range(*nQ_range, 2):
        reg3 = reg3_dataset_table.reset_index()
        print(" ================= ")
        print(f"handling nQ={n_qubits}")

        # ! what to use existing cert
        # df = pd.read_pickle("count_optima_for_one_graph_df/2022-06-01_21:12:14_count_optima_for_one_graph.p")
        # df = df[df['n_qubits'] == n_qubits]
        df = reg3[(reg3["n"] == n_qubits)]

        if len(df) == 0:
            print(f"no qualified graph for nQ={n_qubits}")
            # assert False
            continue
        
        # randomly choose one of the graph with given node
        _row = df.sample(n=1).iloc[0]
        G = _row["G"]
        # _row = df.iloc[0]
        # cert = _row["row_id"]
        
        for p in range(2, 7):    
            print("--------")
            row = get_3_reg_dataset_table_row(G, p)
            cert = get_pynauty_certificate(G)

            # ! what to use existing cert
            # row = reg3_dataset_table.loc[(cert, p)]
            # G = row['G']

            # print(f"num of graphs: {len(df)}")
            print(f"p={p}")

            if p in [1, 2]:
                angles = {
                    "beta": beta_to_qaoa_format(row["beta"]),
                    "gamma": gamma_to_qaoa_format(row["gamma"]),
                }
                C_opt = row["C_opt"]
            else:
                angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))
                C_opt = row["C_fixed"]

            # print(row["beta"])
            # print(row["gamma"])
            # print(row["p_max"])
            # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

            print("C_opt", C_opt)
            
            figdir = f'figs/count_optima_of_one_graph/{signature}/nQ{n_qubits}_p{p}'
            
            if not os.path.exists(figdir):
                os.makedirs(figdir)

            nx.draw_networkx(G)
            plt.title(f"")
            plt.savefig(f"{figdir}/G.png")
            plt.cla()

            n_optima_list = vis_landscape_multi_p_and_and_count_optima(
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

            count_rst_df = count_rst_df.append({
                'row_id': cert,
                'n_qubits': n_qubits,
                'p': p,
                'n_optima_list': n_optima_list,
                'has_opt': False,
            }, ignore_index=True)

            # print(count_rst_df)
            count_rst_df.to_pickle(f"{df_path}/{signature}_count_optima_for_one_graph.p")
            print(" ================ ")

    return


def print_graphs():
    count_rst_df = pd.DataFrame(
        columns=[
            'row_id',
            'combined_G',
            'original_G',
            'n_connect',
            'original_cert',
            'combined_cert',
            'n_qubits',
            'p',
            'original_n_optima_list',
            'combined_n_optima_list',
            'has_opt']
    )

    # reg3_dataset_table = get_3_reg_dataset_table()
    reg3_dataset_table = get_full_qaoa_dataset_table()
    # df = reg3_dataset_table.reset_index()
    signature = get_curr_formatted_timestamp()
    # n_qubits = 4
    # p = 2

    n_connect = 2
    # df = df[(df['n']==n_qubits) & (df['p_max']==p)]
    # df = df[:10]
    
    
    start_time = time.time()

    # for n_qubits in range(3, 9):
    #     for p in range(2, 7):
    for n_qubits in range(3, 5):
        for p in range(2, 4):
            df = reg3_dataset_table.reset_index()
            df = df[(df['n']==n_qubits) & (df['p_max']==p)]
            df = df[:10]
            print(f"n_qubits={n_qubits}")
            print(f"num of graphs: {len(df)}")
            print(f"p={p}")
            for row_id, original_row in df.iterrows():
                figdir = f'figs/compare/{signature}/G{row_id}_nQ={n_qubits}_p={p}'
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                G = original_row['G']
                mapping = dict()
                for i in range(n_qubits):
                    # mirror relabeling
                    mapping[i] = 2 * n_qubits - i - 1

                    # mapping[i] = i + n_qubits

                copy = nx.relabel_nodes(G, mapping)
                
                original_G = G.copy()
                G = nx.compose(G, copy)
                # TODO: randomly pick n_connect many nodes
                for i in range(n_connect):
                    G.add_edge(i, i + n_qubits)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}_combined.png")
                plt.cla()
                
                nx.draw_networkx(copy)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}_copy_relabeled.png")
                plt.cla()

                try:
                    # tmp = get_3_reg_dataset_table_row(G, p)
                    combined_row = get_full_qaoa_dataset_table_row(G, p)
                    # print(tmp)
                except:
                    print("Graph not found in the regular dataset table")
                    continue
                
                
                # angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))
                # C_opt = combined_row["C_opt"]
                # print("C_opt", C_opt)
                
                combined_figdir = f"{figdir}/combined"
                combined_n_optima_list = vis_landscape_multi_p_and_and_count_optima_MP(
                    G=G,
                    p=p,
                    figdir=combined_figdir, 
                    # beta_opt=beta_to_qaoa_format(angles["beta"]),
                    # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                    beta_opt=beta_to_qaoa_format(combined_row["beta"]),
                    gamma_opt=gamma_to_qaoa_format(combined_row["gamma"]),
                    noise_model=None,
                    params_path=[],
                    C_opt=combined_row['C_opt']
                )

                original_figdir = f"{figdir}/original"
                original_n_optima_list = vis_landscape_multi_p_and_and_count_optima_MP(
                    G=original_G,
                    p=p,
                    figdir=original_figdir, 
                    # beta_opt=beta_to_qaoa_format(angles["beta"]),
                    # gamma_opt=gamma_to_qaoa_format(angles["gamma"]),
                    beta_opt=beta_to_qaoa_format(original_row["beta"]),
                    gamma_opt=gamma_to_qaoa_format(original_row["gamma"]),
                    noise_model=None,
                    params_path=[],
                    C_opt=original_row['C_opt']
                )
                
                print('original_n_optima_list', original_n_optima_list)
                print('combined_n_optima_list', combined_n_optima_list)

                count_rst_df = count_rst_df.append({
                    'row_id': row_id,
                    'combined_G': G,
                    'original_G': original_G,
                    'n_connect': n_connect,
                    'original_cert': get_pynauty_certificate(original_G),
                    'combined_cert': get_pynauty_certificate(G),
                    'n_qubits': n_qubits,
                    'p': p,
                    'original_n_optima_list': original_n_optima_list,
                    'combined_n_optima_list': combined_n_optima_list,
                    'has_opt': False,
                }, ignore_index=True)

                # print(count_rst_df)
                count_rst_df.to_pickle(f"compare_df/{signature}_nQ={n_qubits}_p={p}.p")
                print(" ================ ")

        # return

    end_time = time.time()
    print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")
    return


def compare_symmetry():
    

    # ps = [2, 3, 4, 5, 6]
    
    count_rst_df = pd.DataFrame(
        columns=['row_id', 'G', 'pynauty_cert', 'n_qubits', 'p', 'n_optima_list', 'has_opt']
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    
    start_time = time.time()
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    
    
    print(f"n_qubits={n_qubits}")
    print(f"num of graphs: {len(df)}")
    print(f"p={p}")
    for row_id, row in df.iterrows():
        
        print(f"handling {row_id}")
        angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

        # print(row["beta"])
        # print(row["gamma"])
        # print(row["p_max"])
        # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

        C_opt = row["C_fixed"]
        print("C_fixed", C_opt)
        G = row["G"]
        
        # p = len(row["beta"])
        figdir = f'figs/count_optima/{signature}/G{row_id}_nQ{n_qubits}_p{p}'
        
        if not os.path.exists(figdir):
            os.makedirs(figdir)

        nx.draw_networkx(G)
        plt.title(f"")
        plt.savefig(f"{figdir}/G{row_id}.png")
        plt.cla()

        # n_optima_list = vis_landscape_multi_p_and_and_count_optima(
        n_optima_list = vis_landscape_multi_p_and_and_count_optima_MP(
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

        print('n_optima_list', n_optima_list)

        count_rst_df = count_rst_df.append({
            'row_id': row_id,
            'G': G,
            'pynauty_cert': row['pynauty_cert'],
            'n_qubits': n_qubits,
            'p': p,
            'n_optima_list': n_optima_list,
            'has_opt': False,
        }, ignore_index=True)

        # print(count_rst_df)
        count_rst_df.to_pickle(f"count_optima_dataframe/{signature}_count_optima_fixed_angles.p")
        print(" ================ ")

    end_time = time.time()
    print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

        # 4h for nQ=16, p=6
        # 100 min for nQ=12, p=6

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
    count_optima_of_fixed_angles_3reg_graphs_one_variable()