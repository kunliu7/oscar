import argparse
import itertools
from lib2to3.pgen2.token import CIRCUMFLEX
from sqlite3 import paramstyle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
import timeit
import orqviz
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
from scipy.spatial.distance import (
    cosine
)
from qiskit.quantum_info import Statevector
from qiskit.opflow import PrimitiveOp, PauliSumOp
from QAOAKit.n_dim_cs import recon_4D_landscape, recon_4D_landscape_by_2D
# from QAOAKit import vis

sys.path.append('..')
from QAOAKit.noisy_params_optim import (
    get_pauli_error_noise_model,
    optimize_under_noise,
    get_depolarizing_error_noise_model,
    compute_expectation
)

from QAOAKit.vis import(
    _vis_recon_distributed_landscape,
    vis_landscape,
    vis_landscape_heatmap,
    vis_landscape_heatmap_multi_p,
    vis_landscape_multi_p,
    vis_landscape_multi_p_and_and_count_optima,
    vis_landscape_multi_p_and_and_count_optima_MP,
    vis_landscapes,
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
    qaoa_format_to_qiskit_format,
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
    _executor_of_qaoa_maxcut_energy,
    cal_recon_error,
    gen_p1_landscape,
    gen_p2_landscape,
    two_D_CS_p1_recon_with_distributed_landscapes,
    two_D_CS_p1_recon_with_given_landscapes,
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

from scipy import interpolate

test_utils_folder = Path(__file__).parent


def check_QAOAAnsatz_bound_p1():
    p = 1
    # n_pts_per_unit = 16 # 15 * 15
    n_pts_per_unit = {
        "beta": 16,
        "gamma": 16
    }
    
    # p == 2
    bounds = {'beta': np.array([-np.pi/4, np.pi/4]),
              'gamma': np.array([-np.pi/2, np.pi/2])}
    
    n_pts = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit[label] * bound_len).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    print('n_pts_per_unit: ', n_pts_per_unit)

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    # in order of beta1 beta2 gamma1 gamma2
    full_ranges = []
    for _ in range(p):
        full_ranges.append(full_range['beta'].copy())
    
    for _ in range(p):
        full_ranges.append(full_range['gamma'].copy())

    params = []
    # Input : arr1 = [1, 2, 3] 
    # arr2 = [5, 6, 7] 
    # Output : [(1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)] 
    for betas_gammas in itertools.product(*full_ranges):
        param = (
            betas_gammas[:p], # beta
            betas_gammas[p:] # gamma
        )
        params.append(param)

    # ----------- use meshgrid ------------

    x1, x2 = np.meshgrid(*full_ranges, sparse=False, indexing='ij') # 行优先meshgrid
    x1 = x1.reshape(-1) # 行优先reshape
    x2 = x2.reshape(-1)
    for i in range(x1.shape[0]):
        assert x1[i] == params[i][0][0]
        assert x2[i] == params[i][1][0]

    print('num points for p = 1: ', len(full_range['beta']) * len(full_range['gamma']))


def check_QAOAAnsatz_bound_p2():
    p = 2
    # n_pts_per_unit = 16 # 15 * 15
    n_pts_per_unit = {
        "beta": 16,
        "gamma": 10
    }
    
    # p == 2
    bounds = {'beta': np.array([-np.pi/8, np.pi/8]),
              'gamma': np.array([-np.pi/4, np.pi/4])}
    
    n_pts = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit[label] * bound_len).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    print('n_pts_per_unit: ', n_pts_per_unit)

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    # in order of beta1 beta2 gamma1 gamma2
    full_ranges = []
    for _ in range(p):
        full_ranges.append(full_range['beta'].copy())
    
    for _ in range(p):
        full_ranges.append(full_range['gamma'].copy())

    params = []
    # Input : arr1 = [1, 2, 3] 
    # arr2 = [5, 6, 7] 
    # Output : [(1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)] 
    for beta2_gamma2 in itertools.product(*full_ranges):
        param = (
            beta2_gamma2[:p], # beta
            beta2_gamma2[p:] # gamma
        )
        params.append(param)
    
    # -------- check with meshgrid ---------
    x0, x1, x2, x3 = np.meshgrid(*full_ranges, sparse=False, indexing='ij')
    x0 = x0.reshape(-1)
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    x3 = x3.reshape(-1)

    for i in range(x1.shape[0]):
        # beta
        assert x0[i] == params[i][0][0]
        assert x1[i] == params[i][0][1]

        # gamma
        assert x2[i] == params[i][1][0]
        assert x3[i] == params[i][1][1]

    # print(full_ranges)
    n_pts = (len(full_range['beta']) ** 2) * (len(full_range['gamma']) ** 2)
    print('num points: ', n_pts)


def check_QAOAAnsatz_bound():
    p = 2
    # n_pts_per_unit = 16 # 15 * 15
    n_pts_per_unit = {
        "beta": 16,
        "gamma": 16
    }
    n_shots = 1024

    # qaoa bounds
    # bounds_qaoa = {'beta': np.array([-np.pi/8, np.pi/4]),
    #           'gamma': np.array([-np.pi/2, np.pi/1])}
    # bounds_qaoa = {'beta': np.array([-np.pi/8, np.pi/4]),
    #           'gamma': np.array([-np.pi/2, np.pi/1])}
    # bounds_qaoa = {'beta': np.array([-np.pi/4, np.pi/4]),
    #                'gamma': np.array([-np.pi/4, np.pi/4])}
    
    # p == 2
    bounds_qaoa = {'beta': np.array([-np.pi/8, np.pi/8]),
                   'gamma': np.array([-np.pi/8, np.pi/8])}
    
    bounds_qiskit = {
        'beta': np.array([-np.pi/4, np.pi/4]), # do not change beta for p == 1
        'gamma': np.array([-np.pi/8, np.pi/8])
    }

    # bounds_qaoa = {'beta': np.array([-np.pi/8, np.pi/])}
    
    # qiskit bounds
    # bounds_qiskit = {'beta': np.array([-np.pi/8, np.pi/8]),
    #           'gamma': np.array([-np.pi/4, np.pi/4])}

    # print(angles_from_qaoa_format)
    print(bounds_qaoa)
    qiskit1 = qaoa_format_to_qiskit_format(gamma=bounds_qaoa['gamma'], beta=bounds_qaoa['beta'])
    qiskit2 = angles_to_qiskit_format(angles_from_qaoa_format(gamma=bounds_qaoa['gamma'], beta=bounds_qaoa['beta']))

    # bounds_qiskit = qaoa_format_to_qiskit_format()
    # print(bounds_qiskit)

    assert (qiskit1 == qiskit2).all()
    print(qiskit1)
    print(qiskit2)

    print(angles_to_qaoa_format(angles_from_qiskit_format(qiskit1)))

    # bounds = bounds_qiskit
    bounds = bounds_qaoa

    n_pts = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit[label] * bound_len).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    print('n_pts_per_unit: ', n_pts_per_unit)

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    # in order of beta1 beta2 gamma1 gamma2
    full_ranges = []
    for _ in range(p):
        full_ranges.append(full_range['beta'].copy())
    
    for _ in range(p):
        full_ranges.append(full_range['gamma'].copy())

    params = []
    for beta2_gamma2 in itertools.product(*full_ranges):
        param = (
            beta2_gamma2[:p], # beta
            beta2_gamma2[p:] # gamma
        )
        params.append(param)
    
    # -------- check with meshgrid ---------
    x0, x1, x2, x3 = np.meshgrid(*full_ranges, sparse=False, indexing='ij')
    print(x1.shape)
    for i in range(x1.shape[1]):
        # beta
        assert x0[0][i] == params[i][0][0]
        assert x1[0][i] == params[i][0][1]

        # gamma
        assert x2[0][i] == params[i][1][0]
        assert x3[0][i] == params[i][1][1]

    # print(full_ranges)
    n_pts = (len(full_range['beta']) ** 2) * (len(full_range['gamma']) ** 2)
    print('num points: ', n_pts)



def gen_p2_landscape_top():
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
        for p in range(2, 3): # [1, 11]
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

                figdir = f'figs/gen_p2_landscape/{signature}/G{row_id}_nQ{n_qubits}_p{p}_depolar{p1Q}_{p2Q}'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}.png")
                plt.cla()

                gen_p2_landscape(
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
                    n_pts_per_unit=36, # test
                    n_shots=1024,
                    bounds = {'beta': [-np.pi/8, np.pi/8], 'gamma': [-np.pi/8, np.pi/8]},
                )

                print(" ================ ")

            end_time = time.time()
            print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return



# ================== 2-D CS =================

def two_D_CS_p1_recon_with_given_landscapes_top():
    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)

    figdir = f"{data_dir}/2D_CS_recon"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for sf in np.arange(0.05, 0.5, 0.03):
        recon = two_D_CS_p1_recon_with_given_landscapes(
            origin=data['origin'].tolist(),
            sampling_frac=sf
        )


def reconstruct_p2_landscape_top():
    """Reconstructed with two noisy simulations
    """

    is_existing_recon = True 
    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()

    print(origin['ideals'].shape)

    # return

    if not is_existing_recon:
        signature = get_curr_formatted_timestamp()
        recon_dir = f"figs/recon_p2_landscape/{signature}"
        if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)
    else:
        # recon_dir = "figs/recon_distributed_landscape/2022-09-30_14:34:08"
        # recon_dir = "figs/recon_p2_landscape/2022-10-01_19:15:39" # 0.01
        recon_dir = "figs/recon_p2_landscape/2022-10-01_19:50:01" # 0.05

    # --------- data prepared OK -----------

    # sfs = np.arange(0.05, 0.011, 0.05)
    sfs = [0.05]

    errors = []
    for sf in sfs:
        landscape = origin['ideals']

        if not is_existing_recon:
            recon = recon_4D_landscape_by_2D(
                origin=landscape,
                sampling_frac=sf
            )
            
            np.savez_compressed(f"{recon_dir}/recon_p2_landscape_sf{sf:.3f}", recon)
        else:
            recon = np.load(f"{recon_dir}/recon_p2_landscape_sf{sf:.3f}.npz")['arr_0']
            print(recon.shape)

        
        shape_4d = landscape.shape.copy()

        origin_4d = landscape.copy()

        origin_2d = landscape.reshape(landscape.shape[0] * landscape.shape[1],
            landscape.shape[2] * landscape.shape[3])
        recon_2d = recon.reshape(recon.shape[0] * recon.shape[1],
            recon.shape[2] * recon.shape[3])

        # ------------------ visualize all the combs ------------------

        # origin_4d = origin_2d.reshape(*shape_4d)
            
        landscapes = [] # landscapes in different combinations of indices

        # _ls = np.transpose(origin_4d, (0, 1, 2, 3))
        # landscapes.append(_ls)
        
        # _ls = np.transpose(origin_4d, (0, 1, 2, 3))
        # landscapes.append(_ls)

        _vis_recon_distributed_landscape(
            landscapes=[origin_2d, recon_2d],
            labels=['full', 'recon'],
            full_range={'beta': range(225), 'gamma': range(225)},
            bounds=None,
            true_optima=None,
            title=f'reconstruct landscape, sampling fraction: {sf:.3f}',
            save_path=f'{recon_dir}/recon_p2_landscape_{sf:.3f}.png'
        )

        # error = np.linalg.norm(recon - origin['ideals'])
        error1 = cal_recon_error(recon.reshape(-1), origin['ideals'].reshape(-1), "MSE")
        error2 = cosine(recon.reshape(-1), origin['ideals'].reshape(-1))
        print(f"============ error for {sf:.3f}: MSE={error1:.3f}, Cosine={error2:.3f} =============")
        errors.append((error1, error2))

    return


def test_4D_CS():
    t = np.linspace(0, 1, 10000)
    origin = np.cos(2 * 97 * t * np.pi).reshape(10, 10, 10, 10)
    recon = recon_4D_landscape_by_2D(
        origin=origin,
        sampling_frac=0.05
    )

    error = np.linalg.norm(origin - recon)

    print(error)


def vis_p2_landscape_by_PCA(loss_function, opt_paths: list):
    from orqviz.pca import get_pca, perform_2D_pca_scan, plot_pca_landscape, plot_optimization_trajectory_on_pca
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes.reshape(-1)
    parameter_trajectory = opt_paths[0] + opt_paths[1]
    pca_object = get_pca(parameter_trajectory)
    pca_scan_result = perform_2D_pca_scan(pca_object, loss_function, n_steps_x=50, verbose=True)

    for i in range(2):
        plot_pca_landscape(pca_scan_result, pca_object, fig=fig, ax=axes[i])
        plot_optimization_trajectory_on_pca(opt_paths[i], pca_object, ax=axes[i])
    
        axes[i].legend()

    plt.savefig("paper_figs/p2_recon_landscape.png", bbox_inches='tight')


def vis_p2_landscape(loss_function, opt_paths):
    p = 2
    # bounds = {'beta': [-np.pi/2, np.pi/2],
    #             'gamma': [-np.pi/2, np.pi/2]}
    bounds = {'beta': [-np.pi/8, np.pi/8],
                'gamma': [-np.pi/8, np.pi/8]}
    # n_shots: int=2048
    n_pts_per_unit = 64

    n_pts = {}
    # n_samples = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit * bound_len).astype(int)
        # n_samples[label] = np.ceil(n_pts_per_unit * bound_len * sampling_frac).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    # print('n_samples: ', n_samples)
    # print('alpha: ', alpha)
    print('n_pts_per_unit: ', n_pts_per_unit)
    # sample P points from N randomly

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    # in order of beta1 beta2 gamma1 gamma2
    # full_ranges = []
    # for ip in range(p):
    #     full_ranges.append(full_range['beta'].copy())
    
    # for ip in range(p):
    #     full_ranges.append(full_range['gamma'].copy())

    # params = []
    # # former 2 are betas, latter 2 are gammas
    # for beta2_gamma2 in itertools.product(*full_ranges):
    #     param = beta2_gamma2
    #     params.append(param)

    # params = np.array(params)
    # print("all params: ", params.shape)
    vis_p2_landscape_by_PCA(loss_function, opt_paths)


def vis_p2_landscape_top():
    sf = 0.05
    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()

    recon_dir = "figs/recon_p2_landscape/2022-10-01_19:50:01" # 0.05
    recon = np.load(f"{recon_dir}/recon_p2_landscape_sf{sf:.3f}.npz")['arr_0']

    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 2
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    assert row_id == 41

    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # get_maxcut_qaoa_circuit(G, beta)
    
    # noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
    noise_model = None
    shots = 2048
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
    print(recon.shape)

    opt_cut = row["C_opt"]

    initial_angles = {
        "gamma": np.array([np.random.uniform(bounds[0][0], bounds[0][1], p)]),
        "beta": np.array([np.random.uniform(bounds[1][0], bounds[1][1], p)])
    }
    
    print("initial points: ", initial_angles)

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence
    
    # optimizer = SPSA()
    # optimizer = ADAM()

    def _partial_qaoa_energy(x):
        # angles = angles_from_qiskit_format(x)
        # angles = angles_to_qaoa_format(angles)

        # x = np.concatenate([angles['gamma'], angles['beta']])
        # circuit = get_maxcut_qaoa_circuit(
        #     G, beta=[x[1]], gamma=[x[0]],
        #     # transpile_to_basis=True, save_state=False)
        #     transpile_to_basis=False, save_state=False)

        # return _executor_of_qaoa_maxcut_energy(
        #     qc=circuit, G=G, noise_model=noise_model, shots=shots
        # )
        # print(x)
        return -noisy_qaoa_maxcut_energy(
            G=G, beta=x[:p], gamma=x[p:], precomputed_energies=None, noise_model=noise_model
        )

    # vis_p2_landscape(_partial_qaoa_energy)

    # return
    # ---------- vis ------------

    # raw_optimizer_clazz = SPSA
    raw_optimizer_clazz = ADAM
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer_clazz
    )(
        bounds=None,
        landscape=None,
        fun_type='FUN',
        fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        # lr=1e-3, # ! before
        # lr=1e-2, # ! after
        # maxiter=100,
        # beta_1=0.9,
        # beta_2=0.99,
        # noise_factor=1e-8,
        # eps=1e-10,
        # eps=1e-3,
    )

    optimizer_name = raw_optimizer_clazz.__name__

    # opt path for ADAM
    opt_path1 = np.load(
        # "figs/opt_on_p2_landscape/2022-10-11_15:51:03/ADAM.npz",
        "figs/opt_on_p2_landscape/2022-10-11_16:27:29/ADAM.npz", # 0.0.0.0
        allow_pickle=True)['params_path'].tolist()

    # opt_path_SPSA = np.load(
    opt_path2 = np.load(
        # "figs/opt_on_p2_landscape/2022-10-11_15:51:47/SPSA.npz",
        # "figs/opt_on_p2_landscape/2022-10-11_16:01:42/ADAM.npz",
        # "figs/opt_on_p2_landscape/2022-10-11_16:10:48/ADAM.npz", 
        # "figs/opt_on_p2_landscape/2022-10-11_16:25:32/ADAM.npz", # initial points:  {'gamma': array([0.31962453, 0.47234702]), 'beta': array([0.20990108, 0.68302982])}
        "figs/opt_on_p2_landscape/2022-10-11_19:06:12/ADAM.npz",
        allow_pickle=True)['params_path'].tolist()

    opt_paths = [opt_path1, opt_path2]
    # print("opt path len: ", len(opt_path))

    vis_p2_landscape(_partial_qaoa_energy, opt_paths)


def opt_on_p2_landscape_top():
    sf = 0.05
    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()

    recon_dir = "figs/recon_p2_landscape/2022-10-01_19:50:01" # 0.05
    recon = np.load(f"{recon_dir}/recon_p2_landscape_sf{sf:.3f}.npz")['arr_0']

    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 2
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    assert row_id == 41

    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # get_maxcut_qaoa_circuit(G, beta)
    
    # noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
    noise_model = None
    shots = 2048
    qinst = AerSimulator(method="statevector") 

    bounds = data['bounds'].tolist()
    bounds = np.array([bounds['gamma'], bounds['beta']])

    print(bounds)
    print(recon.shape)

    opt_cut = row["C_opt"]
    
    initial_angles = {
        "gamma": np.array(np.random.uniform(bounds[0][0], bounds[0][1], p)),
        "beta": np.array(np.random.uniform(bounds[1][0], bounds[1][1], p))
    }

    initial_angles = {
        "gamma": np.array([.0, .0]),
        "beta": np.array([.0, .0])
    }

    initial_angles = {
        'gamma': np.array([0.31962453, 0.47234702]),
        'beta': np.array([0.20990108, 0.68302982])
    }
    
    initial_angles = {
        'gamma': np.array([-0.31962453, 0.47234702]),
        'beta': np.array([-0.20990108, 0.68302982])
    }

    initial_angles = {
        'gamma': np.array([-1, -1]),
        'beta': np.array([-1, -1])
    }

    initial_angles = {
        'gamma': np.array([0.5, 0.5]),
        'beta': np.array([0.5, 0.5])
    }
    
    # initial_angles = {
    #     "gamma": np.array([1.0, 1.0]),
    #     "beta": np.array([1.0, 1.0])
    # }
    
    print("initial_angles: ", initial_angles)

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence

    def _partial_qaoa_energy(x):
        # angles = angles_from_qiskit_format(x)
        # angles = angles_to_qaoa_format(angles)

        # x = np.concatenate([angles['gamma'], angles['beta']])
        # circuit = get_maxcut_qaoa_circuit(
        #     G, beta=[x[1]], gamma=[x[0]],
        #     # transpile_to_basis=True, save_state=False)
        #     transpile_to_basis=False, save_state=False)

        # return _executor_of_qaoa_maxcut_energy(
        #     qc=circuit, G=G, noise_model=noise_model, shots=shots
        # )
        # print(x)
        return -noisy_qaoa_maxcut_energy(
            G=G, beta=x[:p], gamma=x[p:], precomputed_energies=None, noise_model=noise_model
        )

    # raw_optimizer_clazz = SPSA
    raw_optimizer_clazz = ADAM
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer_clazz
    )(
        bounds=None,
        landscape=None,
        fun_type='FUN',
        fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        # lr=1e-3,
        # lr=1e-2,
        maxiter=1000,
        # beta_1=0.9,
        # beta_2=0.99,
        # noise_factor=1e-8,
        # eps=1e-10,
        # eps=1e-3,
    )

    optimizer_name = raw_optimizer_clazz.__name__
    
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
        # callback=cb_store_intermediate_result
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
    print(len(params))
    # print(params)
    # vis_p2_landscape(_partial_qaoa_energy, params)

    # record parameters
    timestamp = get_curr_formatted_timestamp()
    figdir = f"figs/opt_on_p2_landscape/{timestamp}"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print("save to ", figdir)
    # opt_path_save_path = f"{}"
    np.savez_compressed(f"{figdir}/{optimizer_name}",
        # initial_point=initial_angles,
        initial_angles=initial_angles,
        params_path=params
    )

    return 


def vis_p2_recon_landscape_top():
    sf = 0.05
    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    full_ranges: list = data['full_ranges'].tolist()
    bounds = data['bounds'].tolist()
    n_pts_per_unit = data['n_pts_per_unit']

    recon_dir = "figs/recon_p2_landscape/2022-10-01_19:50:01" # 0.05
    recon = np.load(f"{recon_dir}/recon_p2_landscape_sf{sf:.3f}.npz")['arr_0']

    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 2
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    assert row_id == 41

    print("============================")
    angles1 = opt_angles_for_graph(row["G"], row["p_max"])
    G = row["G"]
    print('row_id:', row_id)
    qc1, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    # get_maxcut_qaoa_circuit(G, beta)
    
    # noise_model = get_depolarizing_error_noise_model(0.001, 0.005)
    noise_model = None
    shots = 2048
    qinst = AerSimulator(method="statevector") 

    bounds = data['bounds'].tolist()
    bounds_arr = np.array([bounds['gamma'], bounds['gamma'], bounds['beta'], bounds['beta']])

    print(bounds)
    print(recon.shape)

    opt_cut = row["C_opt"]
    
    # initial_angles = {
    #     "gamma": np.array(np.random.uniform(bounds[0][0], bounds[0][1], p)),
    #     "beta": np.array(np.random.uniform(bounds[1][0], bounds[1][1], p))
    # }

    initial_angles = {
        "gamma": np.array([.0, .0]),
        "beta": np.array([.0, .0])
    }

    initial_angles = {
        'gamma': np.array([0.31962453, 0.47234702]),
        'beta': np.array([0.20990108, 0.68302982])
    }
    
    # initial_angles = {
    #     "gamma": np.array([1.0, 1.0]),
    #     "beta": np.array([1.0, 1.0])
    # }
    
    print("initial_angles: ", initial_angles)

    # find the case that CS reconstructed landscape will display BP
    # when slow convergence


    # raw_optimizer_clazz = SPSA
    raw_optimizer_clazz = ADAM
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer_clazz
    )(
        bounds=bounds_arr,
        landscape=origin['ideals'],
        # fun_type='NONE',
        fun_type='INTERPOLATE',
        # fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        # lr=1e-3,
        # lr=1e-2,
        maxiter=1000,
        # beta_1=0.9,
        # beta_2=0.99,
        # noise_factor=1e-8,
        # eps=1e-10,
        # eps=1e-3,
    )

    optimizer_name = raw_optimizer_clazz.__name__

    # x1, x2, x3, x4 = \
    #     np.meshgrid(*full_ranges, indexing='ij') #, sparse=True)

    # print(x1.shape)
    # print(origin['ideals'].shape)

    # print(tuple(full_ranges))
    # print(x1)
    interp = interpolate.RegularGridInterpolator(
        points=tuple(full_ranges),
        values=origin['ideals'],
        bounds_error=False,
        fill_value=0.0,
        method="linear"
    )

    def _partial_qaoa_energy(x):

        # interpolate.interpn(
        #     points=None,
        #     values=None,
        # )
        # print(x)
        # print(x.shape)
        _x = np.array([x])
        # print(_x)
        val = interp(_x)
        # print(val)
        return -val[0]

        # angles = angles_from_qiskit_format(x)
        # angles = angles_to_qaoa_format(angles)

        # x = np.concatenate([angles['gamma'], angles['beta']])
        # circuit = get_maxcut_qaoa_circuit(
        #     G, beta=[x[1]], gamma=[x[0]],
        #     # transpile_to_basis=True, save_state=False)
        #     transpile_to_basis=False, save_state=False)

        # return _executor_of_qaoa_maxcut_energy(
        #     qc=circuit, G=G, noise_model=noise_model, shots=shots
        # )
        # print(x)
        # return -noisy_qaoa_maxcut_energy(
        #     G=G, beta=x[:p], gamma=x[p:], precomputed_energies=None, noise_model=noise_model
        # )

    optimizer_name = raw_optimizer_clazz.__name__

    # opt path for ADAM
    opt_path1 = np.load(
        # "figs/opt_on_p2_landscape/2022-10-11_15:51:03/ADAM.npz",
        "figs/opt_on_p2_landscape/2022-10-11_16:27:29/ADAM.npz", # 0.0.0.0
        allow_pickle=True)['params_path'].tolist()

    # opt_path_SPSA = np.load(
    opt_path2 = np.load(
        # "figs/opt_on_p2_landscape/2022-10-11_15:51:47/SPSA.npz",
        # "figs/opt_on_p2_landscape/2022-10-11_16:01:42/ADAM.npz",
        # "figs/opt_on_p2_landscape/2022-10-11_16:10:48/ADAM.npz", 
        # "figs/opt_on_p2_landscape/2022-10-11_16:25:32/ADAM.npz", # initial points:  {'gamma': array([0.31962453, 0.47234702]), 'beta': array([0.20990108, 0.68302982])}
        # "figs/opt_on_p2_landscape/2022-10-11_18:54:00/ADAM.npz",
        # "figs/opt_on_p2_landscape/2022-10-11_18:58:32/ADAM.npz",
        # "figs/opt_on_p2_landscape/2022-10-11_19:03:04/ADAM.npz",
        "figs/opt_on_p2_landscape/2022-10-11_19:06:12/ADAM.npz",
        allow_pickle=True)['params_path'].tolist()

    opt_paths = [opt_path1, opt_path2]
    # print("opt path len: ", len(opt_path))

    vis_p2_landscape(_partial_qaoa_energy, opt_paths)


def _get_min_given_init_pt(
    G, 
    p,
    C,
    initial_point: dict  # ! QAOA format
) -> float:

    noise_model = None
    qinst = AerSimulator(method="statevector")
    shots = 2048
    _, C, offset = get_maxcut_qaoa_qiskit_circuit_unbinded_parameters(
        G, p
    )

    def _partial_qaoa_energy(x):
        # TODO use unbinded circuit to speed up if necessary
        # by shots
        circuit = get_maxcut_qaoa_circuit(
            G, beta=x[:p], gamma=x[p:],
            transpile_to_basis=False, save_state=False)

        return -_executor_of_qaoa_maxcut_energy(
            qc=circuit, G=G, noise_model=noise_model, shots=shots
        )

        # by statevector
        # return -noisy_qaoa_maxcut_energy(
        #     G=G, beta=x[:p], gamma=x[p:], precomputed_energies=None, noise_model=noise_model
        # )

    # print(p)
    raw_optimizer_clazz = SPSA
    # raw_optimizer_clazz = ADAM
    
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer_clazz
    )(
        bounds=None,
        landscape=None,
        fun_type='FUN',
        fun=_partial_qaoa_energy,
        
        # parameter of raw optimizer
        # lr=1e-3,
        # lr=1e-2,
        # maxiter=1000,
    )

    init_pt_qiskit = angles_to_qiskit_format(angles_from_qaoa_format(**initial_point))
    print(init_pt_qiskit)
    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        initial_point=init_pt_qiskit,
        quantum_instance=qinst,
    )

    # I = Operator(np.eye(2**n_qubits))
    # I = Pauli("I" * n_qubits)
    # print(C)
    # optimizer.minimize(fun=None, x0=)
    # I = PrimitiveOp(I)
    # I = PauliSumOp.from_list([("I" * (4 * p), 1.0)])
    # result = qaoa.compute_minimum_eigenvalue(I) # useless
    result = qaoa.compute_minimum_eigenvalue(C)

    # eigenstate, eigenvalue, params_path = optimize_under_noise(
    #     G=G, init_beta_gamma=initial_point, noise_model=None, num_shots=shots, opt_method='L-BFGS-B'
    # )

    eigenvalue = result.eigenvalue
    # params = optimizer.params_path
    # print(params[0], angles_from_qiskit_format(params[0]))
    # print(counts)
    # print(qaoa.optimal_params)
    # print("opt_cut                     :", opt_cut)
    print("minimum:", eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))
    # print("len of params:", len(params))
    return eigenvalue


def find_good_initial_points_on_recon_LS_and_verify_top():
    """Find good initial points on recon landscapes.

    This is to verify that recon. will give us correct information / intuition.
    """
    sf = 0.05
    data_dir = "figs/gen_p2_landscape/2022-10-01_16:15:33/G41_nQ8_p2_depolar0.001_0.005"
    data = np.load(f"{data_dir}/data.npz", allow_pickle=True)
    origin = data['origin'].tolist()
    full_ranges: list = data['full_ranges'].tolist()
    bounds = data['bounds'].tolist()
    n_pts_per_unit = data['n_pts_per_unit']
    C_opt = data['C_opt']

    recon_dir = "figs/recon_p2_landscape/2022-10-01_19:50:01" # 0.05
    recon = np.load(f"{recon_dir}/recon_p2_landscape_sf{sf:.3f}.npz")['arr_0']

    # get problem instance info from QAOAKit
    reg3_dataset_table = get_3_reg_dataset_table()
    n_qubits = 8
    p = 2
    sf = 0.05
    df = reg3_dataset_table.reset_index()
    # print(df['p_max'].value_counts())
    df = df[(df["n"] == n_qubits) & (df["p_max"] == p)]
    for row_id, row in df.iloc[1:2].iterrows():
        pass

    G = row['G']

    assert row_id == 41

    print("============= origin and recon 4-D LS are loaded ===============")
    print(f"shape of recon LS: {recon.shape}")
    
    # print(df['p_max'])
    # return
    
    # find, say 100 points that are \epsilon-close to minimum value of recon landscape
    
    recon = -recon
    C_opt = -C_opt
    eps = 0.2 # 4
    eps = 0.3 # 22
    eps = 0.4 # 63
    eps = 0.5 #
    
    min_recon = np.min(recon)
    print(f"minimum recon: {min_recon:.5f}")
    print(f"C_opt: {C_opt:.5f}")

    mask = np.abs(recon - min_recon) < eps
    # print(mask)
    print(np.sum(mask == True))
    # return
    ids = np.argwhere(mask == True) # return indices of points where mask == True
    # print(ids)

    # landscape.shape = (beta, beta, gamma, gamma)
    # randomly sample some points within eps as initial points
    # print(full_ranges)
    # full_ranges = np.array(full_ranges)
    print(ids)
    min_energies = []
    for idx in ids: # idx: tuples
        # print(idx, recon[idx[0], idx[1], idx[2], idx[3]])
        # continue
        init_beta_gamma = [0 for _ in range(2*p)]
        for i in range(2*p):
            init_beta_gamma[i] = full_ranges[i][idx[i]]
        initial_point = {
            'beta': np.array(init_beta_gamma[:p]),
            'gamma': np.array(init_beta_gamma[p:]),
        }

        # print(initial_point)
        min_energy = _get_min_given_init_pt(G=G, p=p, C=None, initial_point=initial_point)

        min_energies.append(min_energy)

    timestamp = get_curr_formatted_timestamp()
    save_dir = f"figs/find_init_pts_by_recon/{timestamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(
        f"{save_dir}/G{row_id}_nQ{n_qubits}_p{p}",
        ids=ids,
        min_energies=min_energies, # one to one correspondence to ids
        min_recon=min_recon,
        eps=eps,
        C_opt=C_opt,
    )


def recon_large_qubits_p2_landscape_top():
    is_recon = True
    data_dir = "figs/gen_p2_landscape/2022-10-18_15:50:00_n6_p2_ideal"
    data = np.load(f"{data_dir}/sv-aer-gpu-n=16-p=2-seed=0-12-15.npz", allow_pickle=True)

    timestamp = get_curr_formatted_timestamp()
    bounds = {
        'beta': data['beta_bound'],
        'gamma': data['gamma_bound']
    }

    # n_pts_per_unit = {
    #     'beta': data['beta_bound'],
    #     'gamma': data['gamma_bound']
    # }

    # n_pts = {}
    # for label, bound in bounds.items():
    #     bound_len = bound[1] - bound[0]
    #     n_pts[label] = np.floor(n_pts_per_unit[label] * bound_len).astype(int)
    
    # print('bounds: ', bounds)
    # print('n_pts: ', n_pts)
    # print('n_pts_per_unit: ', n_pts_per_unit)

    # full_range = {
    #     'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
    #     'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    # }

    landscape = data['data']
    print(landscape.shape)

    # landscape = landscape.reshape(25, 50)

    full_range = {
        'beta': np.linspace(- bounds['beta'], bounds['beta'], landscape.shape[0]),
        'gamma': np.linspace(- bounds['gamma'], bounds['gamma'], landscape.shape[2])
    }
    print(full_range)

    # origin = {'ideals': landscape}
    # recon_dir = f"figs/recon/{timestamp}"
    figdir = f"{data_dir}/2D_CS_recon"
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    # for sf in np.arange(0.05, 0.5, 0.03):
    for sf in [0.05]:
        # recon = two_D_CS_p1_recon_with_given_landscapes(
        #     figdir=None,
        #     origin=origin,
        #     full_range=None,
        #     sampling_frac=sf
        # )

        if not is_recon:
            recon = recon_4D_landscape_by_2D(
                origin=landscape,
                sampling_frac=sf
            )
            np.savez_compressed(f"{figdir}_sf{sf:.3f}",
                recon=recon
            )
        else:
            recon = np.load(f"{figdir}_sf{sf:.3f}.npz",
                allow_pickle=True
            )['recon']
            
        mse = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "MSE")
        # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
        cos = cosine(landscape.reshape(-1), recon.reshape(-1))
        # ncc = cal_recon_error()
        print("RMSE: ", mse)
        print("Cosine: ", cos)
        origin_2d = landscape.reshape(landscape.shape[0] * landscape.shape[1],
            landscape.shape[2] * landscape.shape[3])
        recon_2d = recon.reshape(recon.shape[0] * recon.shape[1],
            recon.shape[2] * recon.shape[3])

        vis_landscapes(
            landscapes=[origin_2d, recon_2d],
            labels=["origin", "recon"],
            full_range={
                "beta": range(origin_2d.shape[0]),
                "gamma": range(origin_2d.shape[1])
            },
            true_optima=None,
            title="Compare different ZNE configs and reconstruction",
            save_path=f'{figdir}/origin_and_2D_recon_sf{sf:.3f}.png',
            params_paths=[None, None]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim', type=str, help="Your aims, vis, opt", required=True)
    args = parser.parse_args()
    # gen_p1_landscape_top()
    # reconstruct_by_distributed_landscapes_top()
    # reconstruct_by_distributed_landscapes_two_noisy_simulations_top()
    # test_4D_CS()
    # gen_p2_landscape_top()
    # reconstruct_p2_landscape_top()
    if args.aim == 'opt':
        opt_on_p2_landscape_top()
    elif args.aim == 'vis':
        vis_p2_landscape_top()
    elif args.aim == 'vis_recon':
        vis_p2_recon_landscape_top()
    elif args.aim == 'find':
        find_good_initial_points_on_recon_LS_and_verify_top()
    elif args.aim == 'check':
        check_QAOAAnsatz_bound()
    elif args.aim == 'check1':
        check_QAOAAnsatz_bound_p1()
    elif args.aim == 'check2':
        check_QAOAAnsatz_bound_p2()
    elif args.aim == 'large':
        recon_large_qubits_p2_landscape_top()
    else:
        assert False, f"Invalid aim: {args.aim}"