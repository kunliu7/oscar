from cProfile import label
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# import pytest
from itertools import groupby
import timeit
import sys, os
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector
from sympy import print_maple_code
# from QAOAKit import vis

sys.path.append('..')
from QAOAKit.noisy_params_optim import (
    get_pauli_error_noise_model,
    optimize_under_noise,
    get_depolarizing_error_noise_model,
)

from QAOAKit.vis import(
    vis_landscape,
    vis_landscape_heatmap,
    vis_landscape_heatmap_multi_p,
    vis_landscape_multi_p,
    vis_landscape_multi_p_and_and_count_optima
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

from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz

test_utils_folder = Path(__file__).parent


def count_optimals():
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
        columns=['row_id', 'G', 'n_qubits', 'p', 'n_optima_list', 'has_opt']
    )
    
    reg3_dataset_table = get_3_reg_dataset_table()
    print("read 3 reg dataset OK")
    print("use fixed angles to calculate n_optima")
    signature = get_curr_formatted_timestamp()
    for n_qubits in range(4, 17): # [1, 16]
        for p in range(3, 4): # [1, 11]
            df = reg3_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.iloc[:MAX_NUM_GRAPHS_PER_NUM_QUBITS]
            
            print(f"n_qubits={n_qubits}")
            print(f"num of graphs: {len(df)}")
            print(f"p={p}")
            for row_id, row in df.iterrows():
                cert = row_id
                print(f"handling {cert}")
                angles = angles_to_qaoa_format(get_fixed_angles(d=3, p=p))

                # print(row["beta"])
                # print(row["gamma"])
                # print(row["p_max"])
                # print(row["C_opt"], row["C_{true opt}"], row["C_fixed"])

                C_opt = row["C_fixed"]
                print("C_fixed", C_opt)
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
                count_rst_df.to_pickle(f"{signature}_count_optima_fixed_angles.p")
                print(" ================ ")

    return


def test_vis_depolarizing():
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()

    p1Qs = [0.001, 0.002, 0.005]
    p2Qs = [0.005, 0.01, 0.02]
    for p1Q, p2Q in zip(p1Qs, p2Qs):
        signature = get_curr_formatted_timestamp()
        # p1Q = 0.005
        # p2Q = 0.02
        noise_model = get_depolarizing_error_noise_model(p1Q, p2Q)
        # noise_model = None
        cnt = 0
        # for n_qubits in [3, 4]:
        for n_qubits in [8]:
            print(f"now: p1={p1Q}, p2={p2Q}, nQubits={n_qubits}")
            p = 2
            df = full_qaoa_dataset_table.reset_index()
            # print(df["n"].max())
            # exit()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.head(3)
            print("total # circuit", len(df))
            for _, row in df.iterrows():
                C_noisy = noisy_qaoa_maxcut_energy(
                    row["G"],
                    beta_to_qaoa_format(row["beta"]),
                    gamma_to_qaoa_format(row["gamma"]),
                    noise_model=noise_model
                )
                diff = abs(row["C_opt"] - C_noisy)
                print(row["C_opt"], C_noisy, diff)
                # vis_landscape(row["G"], "test")
                # print(type(row["graph_id"]))
                # vis_landscape_heatmap(row["G"], f'id{cnt}', 
                #     beta_to_qaoa_format(row["beta"])[0],
                #     gamma_to_qaoa_format(row["gamma"])[0])
                
                vis_landscape_multi_p(
                    row["G"],
                    f'figs/{signature}_p1Q{p1Q}_p2Q{p2Q}/G{cnt}_nQubit{n_qubits}_{diff:.2}', 
                    beta_to_qaoa_format(row["beta"]),
                    gamma_to_qaoa_format(row["gamma"]),
                    noise_model
                )
                # return
                cnt += 1

                # assert np.isclose(
                #     row["C_opt"], C_noisy
                # )



def test_vis_pauli():
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()

    p1s = [0.001, 0.005, 0.01]
    timestamp = get_curr_formatted_timestamp()
    for p1 in p1s:
        # p1Q = 0.005
        # p2Q = 0.02
        noise_model = get_pauli_error_noise_model(p1)
        # noise_model = None
        cnt = 0
        # for n_qubits in [3, 4]:
        for n_qubits in [8]:
            print(f"now: p1={p1}, nQubits={n_qubits}")
            p = 2
            df = full_qaoa_dataset_table.reset_index()
            df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
            df = df.head(3)
            print("total # circuit", len(df))
            signature = f"{timestamp}_pauliXZ{p1}_nQ{n_qubits}_p{p}"
            for _, row in df.iterrows():
                C_noisy = noisy_qaoa_maxcut_energy(
                    row["G"],
                    beta_to_qaoa_format(row["beta"]),
                    gamma_to_qaoa_format(row["gamma"]),
                    noise_model=noise_model
                )
                diff = abs(row["C_opt"] - C_noisy)
                print(row["C_opt"], C_noisy, diff)
                # vis_landscape(row["G"], "test")
                # print(type(row["graph_id"]))
                # vis_landscape_heatmap(row["G"], f'id{cnt}', 
                #     beta_to_qaoa_format(row["beta"])[0],
                #     gamma_to_qaoa_format(row["gamma"])[0])
                
                vis_landscape_multi_p(
                    G=row["G"],
                    figdir=f'figs/{signature}/G{cnt}_{diff:.2}', 
                    beta_opt=beta_to_qaoa_format(row["beta"]),
                    gamma_opt=gamma_to_qaoa_format(row["gamma"]),
                    noise_model=None,
                    params_path=[]
                )
                # return
                cnt += 1

                # assert np.isclose(
                #     row["C_opt"], C_noisy
                # )

    
def test_noisy_qaoa_maxcut_energy():
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()

    noise_model = get_depolarizing_error_noise_model(0.001, 0.02)
    # noise_model = None
    for n_qubits in [3, 4]:
        p = 3
        df = full_qaoa_dataset_table.reset_index()
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
        for _, row in df.iterrows():
            C_noisy = noisy_qaoa_maxcut_energy(
                row["G"],
                beta_to_qaoa_format(row["beta"]),
                gamma_to_qaoa_format(row["gamma"]),
                noise_model=noise_model
            )
            print(row["C_opt"], C_noisy, abs(row["C_opt"] - C_noisy))

            # assert np.isclose(
            #     row["C_opt"], C_noisy
            # )


def test_qiskit_qaoa_circuit_old():
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()
    for n_qubits in [3, 4]:
        p = 1
        df = full_qaoa_dataset_table.reset_index()
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
        for _, row in df.iterrows():
            print("========")
            # backend = AerSimulator(method="statevector")
            angles2 = angles_to_qaoa_format(
                opt_angles_for_graph(row["G"], row["p_max"])
            )
            # qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
            # noise_model = get_depolarizing_error_noise_model(0.001, 0.02)
            noise_model = None

            rst = optimize_under_noise(row["G"], [0.0, 0.0], noise_model, 1e6)
            # rst = optimize_under_noise(row["G"], np.vstack([angles2["beta"], angles2["gamma"]]), noise_model, 1e4)

            assert rst.success == True
            print(-rst.fun)
            print(row["C_opt"])
            print(rst.x)
            print(beta_shift_sector(np.array([rst.x[0]])),
                gamma_shift_sector(np.array([rst.x[1]])))
            print(angles2["beta"], angles2["gamma"])
            # print(row["C_"])

            
            # sv2 = Statevector(backend.run(
            #     qc2
            #     , noise_model=noise_model
            # ).result().get_statevector())
            # print(sv2)
            # assert sv1.equiv(sv2)


def test_qiskit_qaoa_circuit_optimization():
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()
    signature = get_curr_formatted_timestamp()
    cnt = 0

    noise_model_type = ''
    # noise_model_type = 'DEPOLAR'
    # noise_model_type = 'PAULI'
    
    for n_qubits in [8]:
        p = 2
        df = full_qaoa_dataset_table.reset_index()
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
        # df = df.head(3)
        df = df.iloc[1:4]
        for _, row in df.iterrows():
            print("============================")
            angles1 = opt_angles_for_graph(row["G"], row["p_max"])
            G = row["G"]
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
            optimizer = L_BFGS_B()
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

            # obj_val = -(sv1.expectation_value(C) + offset)
            opt_cut = row["C_opt"]
            # print('obj_val', obj_val)
            
            counts = []
            values = []
            params = []
            def cb_store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                params.append(parameters)

            if noise_model_type == 'PAULI':
                p_error = 0.01
                noise_model = get_pauli_error_noise_model(p_error)
                noise_sign = f'pauli_p{p_error}'
            elif noise_model_type == 'DEPOLAR':
                # p1 = 0.01
                # p2 = 0.005
                p1 = 0.001
                p2 = 0.005
                noise_model = get_depolarizing_error_noise_model(p1, p2)
                noise_sign = f'depolar_p1e{p1}_p2e{p2}'
            else:
                noise_model = None
                noise_sign = 'noiseless'

            print('noise:', noise_sign)
            # initial_point = np.hstack([[1.0 for _ in range(p)], [-1.0 for _ in range(p)]])
            initial_point = [1.0 for _ in range(2*p)]
            if False:
            # if False:
                counts = []
                values = []
                params = []
                qaoa = QAOA(
                    optimizer,
                    reps=p,
                    # initial_point=angles_to_qiskit_format(angles1),
                    initial_point=initial_point,
                    quantum_instance=backend,
                    callback=cb_store_intermediate_result)
                result = qaoa.compute_minimum_eigenvalue(C)
                print(qaoa.optimal_params)
                print("opt_cut             :", opt_cut)
                print("QAOA energy         :", result.eigenvalue)
                print("QAOA energy + offset:", - (result.eigenvalue + offset))

                opt_point = params[-1].copy()
                params = [
                    _params
                    # angles_to_qaoa_format(angles_from_qiskit_format(_params))
                    for _params in params
                ]
                params.insert(0, angles_to_qaoa_format(angles_from_qiskit_format(initial_point)))
                params.insert(0, angles_to_qaoa_format(angles_from_qiskit_format(qaoa.optimal_params)))
                # vis_landscape_multi_p(
                #     row["G"],
                #     f'figs/test_opt_method/{signature}_nQ{n_qubits}_p{p}_{noise_sign}/G{cnt}',
                #     beta_to_qaoa_format(row["beta"]),
                #     gamma_to_qaoa_format(row["gamma"]),
                #     noise_model,
                #     # params
                #     # angles_to_qaoa_format(angles1)
                #     # [params[-1]]
                #     params
                # )

                print(angles_to_qaoa_format(angles_from_qiskit_format(opt_point)))
                print(angles_to_qaoa_format(angles1))
                print(angles_to_qaoa_format(angles_from_qiskit_format(qaoa.optimal_params)))

                # qc2, C, offset = get_maxcut_qaoa_qiskit_circuit(
                #     G, p, qaoa.optimal_params
                # )

                # backend = AerSimulator(method="statevector")
                # sv = Statevector(backend.run(qc2).result().get_statevector())
                # obj_val = -(sv.expectation_value(C) + offset)
                # print(obj_val)



            if True:
                counts = []
                values = []
                params = []   
                # vqe_optimizer = MinimumEigenOptimizer(vqe)
                # result = vqe_optimizer.solve(C)
                # print(qc1.parameters)
                # noise_model = get_depolarizing_error_noise_model()
                # if True:
                
                qc2 = qc1.copy()

                qinst = QuantumInstance(
                    backend=backend,
                    noise_model=noise_model
                )
                vqe = VQE(qc1,
                    optimizer=optimizer,
                    # initial_point=[a + 1.0 for a in angles_to_qiskit_format(angles1)],
                    initial_point=[1.0 for _ in range(2*p)],
                    callback=cb_store_intermediate_result,
                    quantum_instance=qinst
                )
                result = vqe.compute_minimum_eigenvalue(C)
                exp_cut = -(result.eigenvalue.real + offset)
                
                print("opt_cut            :", opt_cut)
                print('VQE energy         :', result.eigenvalue.real)
                print('VQE energy + offset:', exp_cut)

                diff = abs(exp_cut - opt_cut)

                params = [
                    # _params
                    angles_to_qaoa_format(angles_from_qiskit_format(_params))
                    for _params in params
                ]

                
                vis_landscape_multi_p(
                        row["G"],
                        f'figs/test_opt_method/{signature}_nQ{n_qubits}_p{p}_{noise_sign}/G{cnt}_diff{diff:.2}', 
                        beta_to_qaoa_format(row["beta"]),
                        gamma_to_qaoa_format(row["gamma"]),
                        noise_model,
                        params
                )

                # print(vqe.optimal_params)
                # print(qc2.parameters)
                # qc2.bind_parameters()

                # sv = Statevector(backend.run(qc2).result().get_statevector())
                # obj_val = -(sv.expectation_value(C) + offset)
                # print(obj_val)


            
            # assert sv1.equiv(sv2)
            cnt += 1


def test_removing_edges():
    """Remove edges of graph, to see if there is any change of symmetry
    
    Will the landscape be more sparse, or will it lost symmetry?
    """
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()
    signature = get_curr_formatted_timestamp()
    cnt = 0

    noise_model_type = ''
    # noise_model_type = 'DEPOLAR'
    # noise_model_type = 'PAULI'
    graph_id = 9
    
    for n_qubits in [9]:
        p = 2
        df = full_qaoa_dataset_table.reset_index()
        # df = load_partial_qaoa_dataset_table(n_qubits)
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
        print(f"# of graphs: {len(df)}")
        df = df.loc[[297427, 297439, 297440, 297441, 297443]]
        edges = [(7, 8), (7, 8), (3, 8), (7, 8), (7, 8)]
        # df = df.loc[[297440, 297441, 297443]]
        # edges = [(3, 8), (7, 8), (7, 8)]
        # df = df.iloc[graph_id:graph_id] # TODO for test, one only
        # save_partial_qaoa_dataset_table(df, n_qubits)
        # exit()
        cnt = -1
        for row_idx, row in df.iterrows():
            cnt += 1
            print(f"============ {row_idx} ================")
            angles1 = opt_angles_for_graph(row["G"], row["p_max"])
            original_G = row["G"]
            print("before remove edges: ", original_G.edges)
            fig_path = f"figs/remove_edge/{signature}_id{row_idx}_nQ{n_qubits}_p{p}"
            # vis_landscape_multi_p(
            #     original_G,
            #     f'{fig_path}/origin',
            #     beta_to_qaoa_format(row["beta"]),
            #     gamma_to_qaoa_format(row["gamma"]),
            #     None,
            #     []
            # )
            # nx.draw(original_G)
            # nx.draw_networkx(original_G, nx.spring_layout(original_G))
            nx.draw_networkx(original_G)
            # nx.draw_networkx_nodes(original_G, nx.spring_layout(original_G))
            # nx.draw_networkx_edge_labels(original_G, pos=nx.spring_layout(original_G))
            # plt.savefig(f"id{row_idx}_tobecut_{edge}.png")
            # exit()
            # print(G.edges)
            # edge = G.edges.data()[0]
            edge = edges[cnt]
            plt.title(f"id{row_idx}_tobecut{edge}.png")
            plt.savefig(f"id{row_idx}_tobecut_{edge}.png")
            plt.cla()
            # for edge_idx, edge in enumerate(original_G.edges):
            G = original_G.copy()
                # print(edge)
            G.remove_edge(*edge)
            print(f"remove edges: {edge}")
            # print("after remove edges: ", G.edges)

            # vis_landscape_multi_p(
            #     G,
            #     f'{fig_path}/remove_{edge}', 
            #     beta_to_qaoa_format(row["beta"]),
            #     gamma_to_qaoa_format(row["gamma"]),
            #     None,
            #     []
            # )
            continue
                
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
            optimizer = L_BFGS_B()
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

            # obj_val = -(sv1.expectation_value(C) + offset)
            opt_cut = row["C_opt"]
            # print('obj_val', obj_val)
            
            counts = []
            values = []
            params = []
            def cb_store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                params.append(parameters)

            if noise_model_type == 'PAULI':
                p_error = 0.01
                noise_model = get_pauli_error_noise_model(p_error)
                noise_sign = f'pauli_p{p_error}'
            elif noise_model_type == 'DEPOLAR':
                # p1 = 0.01
                # p2 = 0.005
                p1 = 0.001
                p2 = 0.005
                noise_model = get_depolarizing_error_noise_model(p1, p2)
                noise_sign = f'depolar_p1e{p1}_p2e{p2}'
            else:
                noise_model = None
                noise_sign = 'noiseless'

            print('noise:', noise_sign)
            # initial_point = np.hstack([[1.0 for _ in range(p)], [-1.0 for _ in range(p)]])
            initial_point = [1.0 for _ in range(2*p)]
            if False:
            # if False:
                counts = []
                values = []
                params = []
                qaoa = QAOA(
                    optimizer,
                    reps=p,
                    # initial_point=angles_to_qiskit_format(angles1),
                    initial_point=initial_point,
                    quantum_instance=backend,
                    callback=cb_store_intermediate_result)
                result = qaoa.compute_minimum_eigenvalue(C)
                print(qaoa.optimal_params)
                print("opt_cut             :", opt_cut)
                print("QAOA energy         :", result.eigenvalue)
                print("QAOA energy + offset:", - (result.eigenvalue + offset))

                opt_point = params[-1].copy()
                params = [
                    _params
                    # angles_to_qaoa_format(angles_from_qiskit_format(_params))
                    for _params in params
                ]
                params.insert(0, angles_to_qaoa_format(angles_from_qiskit_format(initial_point)))
                params.insert(0, angles_to_qaoa_format(angles_from_qiskit_format(qaoa.optimal_params)))
                # vis_landscape_multi_p(
                #     row["G"],
                #     f'figs/test_opt_method/{signature}_nQ{n_qubits}_p{p}_{noise_sign}/G{cnt}',
                #     beta_to_qaoa_format(row["beta"]),
                #     gamma_to_qaoa_format(row["gamma"]),
                #     noise_model,
                #     # params
                #     # angles_to_qaoa_format(angles1)
                #     # [params[-1]]
                #     params
                # )

                print(angles_to_qaoa_format(angles_from_qiskit_format(opt_point)))
                print(angles_to_qaoa_format(angles1))
                print(angles_to_qaoa_format(angles_from_qiskit_format(qaoa.optimal_params)))

                # qc2, C, offset = get_maxcut_qaoa_qiskit_circuit(
                #     G, p, qaoa.optimal_params
                # )

                # backend = AerSimulator(method="statevector")
                # sv = Statevector(backend.run(qc2).result().get_statevector())
                # obj_val = -(sv.expectation_value(C) + offset)
                # print(obj_val)



            if True:
                counts = []
                values = []
                params = []   
                # vqe_optimizer = MinimumEigenOptimizer(vqe)
                # result = vqe_optimizer.solve(C)
                # print(qc1.parameters)
                # noise_model = get_depolarizing_error_noise_model()
                # if True:
                
                qc2 = qc1.copy()

                qinst = QuantumInstance(
                    backend=backend,
                    noise_model=noise_model
                )
                vqe = VQE(qc1,
                    optimizer=optimizer,
                    # initial_point=[a + 1.0 for a in angles_to_qiskit_format(angles1)],
                    initial_point=[1.0 for _ in range(2*p)],
                    callback=cb_store_intermediate_result,
                    quantum_instance=qinst
                )
                result = vqe.compute_minimum_eigenvalue(C)
                exp_cut = -(result.eigenvalue.real + offset)
                
                print("opt_cut            :", opt_cut)
                print('VQE energy         :', result.eigenvalue.real)
                print('VQE energy + offset:', exp_cut)

                diff = abs(exp_cut - opt_cut)

                params = [
                    # _params
                    angles_to_qaoa_format(angles_from_qiskit_format(_params))
                    for _params in params
                ]

                
                vis_landscape_multi_p(
                        row["G"],
                        f'figs/test_opt_method/{signature}_nQ{n_qubits}_p{p}_{noise_sign}/G{cnt}_diff{diff:.2}', 
                        beta_to_qaoa_format(row["beta"]),
                        gamma_to_qaoa_format(row["gamma"]),
                        noise_model,
                        params
                )

                # print(vqe.optimal_params)
                # print(qc2.parameters)
                # qc2.bind_parameters()

                # sv = Statevector(backend.run(qc2).result().get_statevector())
                # obj_val = -(sv.expectation_value(C) + offset)
                # print(obj_val)


            
            # assert sv1.equiv(sv2)
            cnt += 1


if __name__ == "__main__":
    # test_qiskit_qaoa_circuit()
    # test_noisy_qaoa_maxcut_energy()
    # test_optimization_method()
    # test_qiskit_qaoa_circuit_optimization()
    # test_removing_edges()
    # count_optimals()
    count_optima_of_fixed_angles_3reg_graphs()
