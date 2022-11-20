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
from qiskit_optimization.applications import Maxcut, SKModel, NumberPartition
from qiskit.algorithms.optimizers.optimizer import POINT
from mitiq.zne.zne import execute_with_zne
from qiskit.providers.aer.noise import NoiseModel
from qiskit.algorithms.optimizers.optimizer import Optimizer as QiskitOptimizer
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

from data_loader import get_interpolation_path_filename, get_recon_landscape, load_grid_search_data
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
    gen_p1_landscape,
    two_D_CS_p1_recon_with_given_landscapes,
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


def get_minimum_by_QAOA(G: nx.Graph, p: int, qiskit_init_pt: np.ndarray,
    noise_model: NoiseModel, optimizer: QiskitOptimizer):
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

    # qinst = AerSimulator(shots=2048, noise_model=noise_model)
    qinst = AerSimulator(method='statevector')
    # shots = 2048
    # optimizer = SPSA(maxiter=maxiter)
    # if isinstance(maxiter, int):
    #     # optimizer = SPSA(maxiter=maxiter)
    #     optimizer = COBYLA(maxiter=maxiter)
    #     print("maxiter:", maxiter)
    # else:
    #     # optimizer = SPSA()
    #     optimizer = COBYLA()
    #     print("no maxiter")
    
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


def optimize_on_p1_reconstructed_landscape(
    n: int, p: int, seed: int, noise: str, miti_method: str,
    initial_point: List[float], opt_name: str, lr: float, maxiter: int
):
    """

    Args:
        initial_point (List[float]): [beta, gamma]

    """
    problem = 'maxcut'
    method = 'sv'
    cs_seed = n
    assert p == 1
    assert len(initial_point) == 2 * p
    sf = 0.05
    if p == 1:
        bs = 50
        gs = 100
    elif p == 2:
        bs = 12
        gs = 15
    else:
        raise NotImplementedError()

    data, data_fname, data_dir = load_grid_search_data(
        n_qubits=n, p=p, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
    )

    recon_dir = f"figs/grid_search_recon/{problem}/{method}-{noise}-p={p}"
    plot_range = data['plot_range']

    recon_fname = f"recon-cs_seed={cs_seed}-sf={sf:.3f}-{data_fname}"
    recon_path = f"{recon_dir}/{recon_fname}"

    origin = data['data']
    recon = get_recon_landscape(p, origin, sf, False, recon_path, cs_seed)

    G = nx.random_regular_graph(3, n, seed)
    maxcut = Maxcut(G)
    # print(maxcut.get_gset_result)
    maxcut_problem = maxcut.to_quadratic_program()
    C, offset = maxcut_problem.to_ising()

    if n <= 16:
        row = get_3_reg_dataset_table_row(G, p)
        print("============================")
        # angles1 = opt_angles_for_graph(row["G"], row["p_max"])
        # qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        #     G, p
        # )
        # backend = AerSimulator(method="statevector")
        # sv1 = Statevector(backend.run(qc1).result().get_statevector())
        # angles2 = angles_to_qaoa_format(
        #     opt_angles_for_graph(row["G"], row["p_max"])
        # )
        # qc2 = get_maxcut_qaoa_circuit(row["G"], angles2["beta"], angles2["gamma"])
        # sv2 = Statevector(backend.run(qc2).result().get_statevector())

    # optimizer = GradientDescent()
    bounds = np.array([
        [-data['beta_bound'], data['beta_bound']], 
        [-data['gamma_bound'], data['gamma_bound']], 
    ])

    print("bounds:", bounds)
    print("landscape shape:", origin.shape)

    opt_cut = row["C_opt"]
    
    counts = []
    values = []
    params = []
    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        # print('fuck')
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    # initial_angles = {
    #     "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1])]),
    #     "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1])])
    # }

    # raw_optimizer = 'SPSA'
    if opt_name == 'ADAM':
        raw_optimizer = ADAM
    elif opt_name == 'SPSA':
        raw_optimizer = SPSA
    elif opt_name == 'COBYLA':
        raw_optimizer = COBYLA
    elif opt_name == 'L_BFGS_B':
        raw_optimizer = L_BFGS_B
    else:
        raise NotImplementedError()

    print("optimizer: ", opt_name)
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer
    )(
        bounds=bounds, 
        landscape=origin,
        fun_type='INTERPOLATE_QISKIT',
        # fun_type='FUN',
        # fun=None
        # fun_type='None'
        # lr=lr,
        maxiter=maxiter
    )
    
    # optimizer = L_BFGS_B()
    # optimizer_name = raw_optimizer.__class__.__name__
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
    print("initial point:", initial_point)
    initial_point = np.array(initial_point)
    # initial_point = np.array([0.1, -0.1])
    qinst = AerSimulator() # meaningless, do not actually activate
    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        # initial_point=angles_to_qiskit_format(
        #     {"gamma": row["gamma"],
        #     "beta": row["beta"]}
        # ),
        initial_point=initial_point,
        quantum_instance=qinst,
        # callback=cb_store_intermediate_result
        )
    result = qaoa.compute_minimum_eigenvalue(C)
    # print(qaoa.optimal_params)
    print("opt_cut                     :", opt_cut)
    print("recon landscape minimum     :", result.eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))

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


    # true_optima = np.concatenate([
    #     gamma_to_qaoa_format(row["gamma"]),
    #     beta_to_qaoa_format(row["beta"]),
    # ])
    base_recon_fname = os.path.splitext(recon_fname)[0]

    vis_landscapes(
        landscapes=[origin, recon],
        labels=["Interpolate", "Circuit Sim."],
        full_range=plot_range,
        true_optima=None,
        title="Origin and recon",
        save_path=f'{recon_dir}/vis-{base_recon_fname}.png',
        params_paths=[params, None]
    )

    if n < 16: 
        # true_optima = shift_parameters(true_optima, bounds)
        _, _, circ_path, _ = get_minimum_by_QAOA(G, p, initial_point, None, raw_optimizer(maxiter=maxiter))

        vis_landscapes(
            landscapes=[recon, origin],
            labels=["Interpolate", "Circuit Sim."],
            full_range=plot_range,
            true_optima=None,
            title="Origin and recon",
            save_path=f'{recon_dir}/vis-{base_recon_fname}.png',
            params_paths=[params, circ_path]
        )
    else:
        circ_path = None

    print(initial_point)
    intp_path_path, _, _ = get_interpolation_path_filename(n, p, problem, method, noise, opt_name, maxiter, initial_point, seed, miti_method)
    ts = get_curr_formatted_timestamp()
    # save_dir = f"figs/opt_on_recon_landscape/{ts}"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # print(params)
    # print(circ_path)
    # save_path = f"{save_dir}/opt_on_recon-opt={opt_name}-init={list(initial_point)}-{base_recon_fname}"
    save_path = intp_path_path
    print("params save to =", save_path)
    np.savez_compressed(
        save_path,
        opt_name=opt_name,
        initial_point=initial_point,
        intp_path=params, # interpolation
        circ_path=circ_path
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
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ns', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('-n', type=int, help="Your aims, vis, opt", required=True)
    parser.add_argument('-p', type=int, help="Your aims, vis, opt", required=True)
    # parser.add_argument('--method', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--noise', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--miti', type=str, help="Your aims, vis, opt", default=None)
    parser.add_argument('--seed', type=int, help="Your aims, vis, opt", required=True)
    parser.add_argument('--lr', type=float, help="Your aims, vis, opt", default=None)
    parser.add_argument('--maxiter', type=int, help="Your aims, vis, opt", default=None)
    parser.add_argument('--init_pt', type=float, nargs="+", help="[beta, gamma]", required=True)
    # parser.add_argument('--error', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--check', action="store_true", help="Your aims, vis, opt", default=False)
    parser.add_argument('--opt', type=str, required=True)
    args = parser.parse_args()

    optimize_on_p1_reconstructed_landscape(
        args.n, args.p, args.seed, args.noise, args.miti, args.init_pt,
        args.opt, args.lr, args.maxiter
    )