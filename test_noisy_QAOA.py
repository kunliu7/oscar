import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
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
from QAOAKit import vis

from QAOAKit.noisy_params_optim import (
    optimize_under_noise,
    get_depolarizing_error_noise_model
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

sys.path.append('..')
from QAOAKit.classical import thompson_parekh_marwaha
from QAOAKit.qaoa import get_maxcut_qaoa_circuit
from QAOAKit.qiskit_interface import (
    get_maxcut_qaoa_qiskit_circuit,
    goemans_williamson,
)
from QAOAKit.examples_utils import get_20_node_erdos_renyi_graphs
from QAOAKit.parameter_optimization import get_median_pre_trained_kde

from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz

test_utils_folder = Path(__file__).parent


def test_vis():
    signature = get_curr_formatted_timestamp()
    full_qaoa_dataset_table = get_full_qaoa_dataset_table()
    
    p1Q = 0.001
    p2Q = 0.005
    noise_model = get_depolarizing_error_noise_model(p1Q, p2Q)
    # noise_model = None
    cnt = 0
    for n_qubits in [3, 4]:
        p = 2
        df = full_qaoa_dataset_table.reset_index()
        df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
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
                f'figs/{signature}_p1Q{p1Q}_p2Q{p2Q}/G{cnt}_{diff:.2}', 
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


def test_qiskit_qaoa_circuit():
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




if __name__ == "__main__":
    test_qiskit_qaoa_circuit()
    # test_noisy_qaoa_maxcut_energy()
    # test_vis()