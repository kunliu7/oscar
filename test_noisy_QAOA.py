import networkx as nx
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
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
    WarmStartQAOAOptimizer
)
from qiskit import Aer
import qiskit
from qiskit.providers.aer import AerSimulator
from functools import partial
from pathlib import Path
import copy
import pytest
from itertools import groupby
import timeit
import sys, os
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector
# from QAOAKit import vis

sys.path.append('..')
from QAOAKit.noisy_params_optim import (
    optimize_under_noise,
    get_depolarizing_error_noise_model,
)

from QAOAKit.vis import(
    vis_landscape,
    vis_landscape_heatmap,
    vis_landscape_heatmap_multi_p,
    vis_landscape_multi_p
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
    obj_from_statevector,
    precompute_energies,
    maxcut_obj,
    isomorphic,
    load_weights_into_dataframe,
    load_weighted_results_into_dataframe,
    get_adjacency_matrix,
    brute_force,
    get_pynauty_certificate,
    get_full_weighted_qaoa_dataset_table
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


def test_vis():
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
    
    for n_qubits in [3, 4]:
        p = 1
        df = full_qaoa_dataset_table.reset_index()
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
        for _, row in df.iterrows():
            print("============================")
            angles1 = opt_angles_for_graph(row["G"], row["p_max"])
            G = row["G"]
            qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
                G, p, angles_to_qiskit_format(angles1)
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
            print('opt_cut', opt_cut)
            
            counts = []
            values = []
            params = []
            def cb_store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                params.append(parameters)

            qaoa = QAOA(
                optimizer,
                reps=p,
                initial_point=[1.0 for _ in range(2*p)],
                quantum_instance=backend,
                callback=cb_store_intermediate_result)
            result = qaoa.compute_minimum_eigenvalue(C)
            print("QAOA energy:", result.eigenvalue)
            print("QAOA energy, + offset:", - (result.eigenvalue + offset))

            # vqe_optimizer = MinimumEigenOptimizer(vqe)
            # result = vqe_optimizer.solve(C)
            # print(qc1.parameters)
            # vqe = VQE(qc1,
            #     optimizer=optimizer,
            #     # initial_point=angles_to_qiskit_format(angles1),
            #     initial_point=[0.0 for _ in range(2*p)],
            #     callback=cb_store_intermediate_result,
            #     quantum_instance=backend
            # )
            # result = vqe.compute_minimum_eigenvalue(C)
            # print(values)
            print(params)
            # result = NumPyMinimumEigensolver(Heisenberg_op).run()
            # result = vqe.get_optimal_cost()
            # print('VQE energy:', result.eigenvalue.real)
            # print('VQE energy, + offset:', -(result.eigenvalue.real + offset))

            vis_landscape_multi_p(
                    row["G"],
                    f'figs/test_opt_method/{signature}_nQ{n_qubits}_p{p}/G{cnt}', 
                    beta_to_qaoa_format(row["beta"]),
                    gamma_to_qaoa_format(row["gamma"]),
                    None,
                    params
                )
            # qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=[0.0, 1.0])
            # ws_qaoa = WarmStartQAOAOptimizer(
            #     pre_solver=CplexOptimizer(), relax_for_pre_solver=True, qaoa=qaoa_mes, epsilon=0.0
            # )
            # assert sv1.equiv(sv2)
            cnt += 1

if __name__ == "__main__":
    # test_qiskit_qaoa_circuit()
    # test_noisy_qaoa_maxcut_energy()
    # test_optimization_method()
    test_qiskit_qaoa_circuit_optimization()