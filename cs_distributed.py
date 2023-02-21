import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import concurrent.futures
import timeit
from typing import List
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
    cal_recon_error,
    gen_p1_landscape,
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
from data_loader import load_grid_search_data, load_ibm_data
from sklearn.linear_model import LinearRegression


test_utils_folder = Path(__file__).parent


def gen_p1_landscape_top():
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

            # p1Q = 0.001
            # p2Q = 0.005

            p1Q = 0.003
            p2Q = 0.007

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

                figdir = f'figs/gen_p1_landscape/{signature}/G{row_id}_nQ{n_qubits}_p{p}_depolar{p1Q}_{p2Q}'
                
                if not os.path.exists(figdir):
                    os.makedirs(figdir)

                nx.draw_networkx(G)
                plt.title(f"")
                plt.savefig(f"{figdir}/G{row_id}.png")
                plt.cla()

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
                    C_opt=C_opt
                )

                print(" ================ ")

            end_time = time.time()
            print(f"for p={p}, nQ={n_qubits}, it takes {end_time-start_time} s")

    return



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


def reconstruct_by_distributed_landscapes_top():
    # 6944.407839, 5891.432676, 5011.566536, 4321.232713, 3659.859955, 2980.218758, 2523.759679, 2140.338593, 1727.501286,
    # 1371.053043, 1069.624135, 812.519126, 588.600523, 

    noisy_data1 = np.load("figs/cnt_opt_miti/2022-08-08_19:48:31/data.npz", allow_pickle=True)
    ideal = noisy_data1['origin'].tolist()['ideals']
    full_range = noisy_data1['full_range'].tolist()

    # depolarizing 0.001 and 0.005, one-qubit gate error and two-qubit gate error
    noisy1 = noisy_data1['origin'].tolist()['unmitis']

    noisy_data_dir2 = "figs/gen_p1_landscape/2022-09-29_00:31:54/G40_nQ8_p1_depolar0.003_0.007"
    noisy_data2 = np.load(
        "figs/gen_p1_landscape/2022-09-29_00:31:54/G40_nQ8_p1_depolar0.003_0.007/data.npz",
        allow_pickle=True)

    # depolarizing 0.003 and 0.007, one-qubit gate error and two-qubit gate error
    noisy2 = noisy_data2['origin'].tolist()['unmitis']

    datas = [
        ideal,
        noisy1,
        noisy2
    ]

    for data in datas:
        print(data.shape)

    # --------- data prepared OK -----------

    sfs = np.arange(0.05, 0.5, 0.03)

    errors = []
    for sf in sfs:

        if True:
            recon = two_D_CS_p1_recon_with_distributed_landscapes(
                ideal=ideal.copy(),
                origins=datas,
                sampling_frac=sf
            )
            
            np.savez_compressed(f"{noisy_data_dir2}/recon_by_{len(datas)}_landscapes_sf{sf:.3f}", recon)
        else:
            recon = np.load(f"{noisy_data_dir2}/recon_by_{len(datas)}_landscapes_sf{sf:.3f}.npz")['arr_0']

        _vis_recon_distributed_landscape(
            landscapes=datas + [recon],
            labels=['ideal', 'depolarizing, 0.001, 0.005', 'depolarizing, 0.003, 0.007', 'reconstructed by 1/3 of each'],
            full_range=full_range,
            bounds=None,
            true_optima=None,
            title=f'reconstruct distributed landscapes, sampling fraction: {sf:.3f}',
            save_path=f'{noisy_data_dir2}/recon_by_{len(datas)}_landscapes.png'
        )

        # error = np.linalg.norm(ideal - recon)
        error = cal_recon_error(ideal.reshape(-1), recon.reshape(-1))
        errors.append(error)
        print("reconstruct error: ", error)
    
    print(errors)

    return


def get_geometric_mean(a: np.ndarray):
    assert len(a.shape) == 1
    mean = a.prod() ** (1.0 / len(a))
    return mean


def ndarray2D_to_set_of_tuples(a):
    ids = []

    for idx in a:
        ids.append(tuple(idx))

    ids = set(ids)
    print("ndarray to set:", len(ids))
    return ids

# def choose_index_with_pos_value(ls, )


def normalize_by_geo_mean(ls1, ls2, n_pts: int):
    rng = np.random.default_rng(0)
    ls1 = ls1.copy()
    ls2 = ls2.copy()

    # minimum = min(np.min(ls1), np.min(ls2)) - 0.1

    # print("minimum =", minimum)

    # ls1 -= minimum
    # ls2 -= minimum

    # assert ls1.min() > 0
    # assert ls2.min() > 0

    shape = ls1.shape
    print("shape =", shape)

    ids = []
    crs = []

    cnt_pts = n_pts
    # only want positive terms
    # 从所有positive elements中随机选index

    print("total:", np.prod(shape))

    # ----------- start intersection ------------
    ids_pos1 = np.argwhere(ls1 > 0)
    ids_pos2 = np.argwhere(ls2 > 0)

    print(ids_pos1.shape)
    print(ids_pos2.shape)
    # ids_pos1 = tuple(ids_pos1)
    # ids_pos2 = tuple(ids_pos2)

    assert len(shape) == 2
    ids_pos1 = ndarray2D_to_set_of_tuples(ids_pos1)
    ids_pos2 = ndarray2D_to_set_of_tuples(ids_pos2)

    intersect = ids_pos1.intersection(ids_pos2)
    intersect: List[tuple] = list(intersect)
    print("intersect:", len(intersect), intersect[:5])

    # ----------- start sampling ------------

    ids = rng.choice(intersect, size=n_pts, replace=False)
    print("sampled:", len(ids), ids[:5])

    for idx in ids:
        idx = tuple(idx)
        e1 = ls1[idx]
        e2 = ls2[idx]
        assert e1 > 0 and e2 > 0
        cr = e1 / e2
        assert cr > 0
        crs.append(cr)

    crs = np.array(crs)
    cr = get_geometric_mean(crs)

    ls2 *= cr
    print("CR =", cr)
    
    return ls1, ls2, cr


def T_flatten(a: np.ndarray):
    assert len(a.shape) == 2
    return a.copy().T.flat[:]


def inv_T_flatten(a: np.ndarray, shape: tuple):
    assert len(a.shape) == 1 and len(shape) == 2
    return a.reshape(shape[::-1]).T


def normalize_by_linear_regression(ls1, ls2, n_pts, ri):
    shape = ls1.shape
    print("shape =", shape)
    assert (ls1 == inv_T_flatten(T_flatten(ls1), ls1.shape)).all()
    fls1 = T_flatten(ls1)
    fls2 = T_flatten(ls2)

    rng = np.random.default_rng(7)

    # pick n_pts random points from ri to train NCM
    # note that ri is fixed for normalization and parallel reconstruction
    ids = rng.choice(ri, n_pts, replace=False)
    print(ids.shape)

    y = fls1[ids]
    x = fls2[ids]
    print(x.shape)
    print(y.shape)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model = model.fit(x, y)

    r_sq = model.score(x, y)
    print('coefficient of determination(𝑅²) :', r_sq)
    # coefficient of determination(𝑅²) : 0.715875613747954
    print('intercept:', model.intercept_)
    # （标量） 系数b0 intercept: 5.633333333333329 -------this will be an array when y is also 2-dimensional
    print('slope:', model.coef_)

    # ls2_flat = ls2.reshape(-1)
    ls2_flat = ls2.flatten().reshape(-1, 1)
    y_pred = model.predict(ls2_flat)
    print(y_pred.shape)
    ls2_normalized = y_pred.reshape(shape)

    return ls1, ls2_normalized


def reconstruct_by_distributed_landscapes_two_noisy_simulations_top(
    n_qubits_list: List[int], p: int, sf: float, noise1: str, noise2: str, seed: int,
    normalize: str, norm_frac: float, error_type: str, recon_dir: str
):
    is_existing_recon = True if isinstance(recon_dir, str) else False
    if not is_existing_recon:
        signature = get_curr_formatted_timestamp()
        recon_dir = f"figs/recon_distributed_landscape/{signature}"
        if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)
    else:
        print("recon_dir specified:", recon_dir)

    errors1 = []
    errors2 = []
    def get_data(noise: str):
        r"""
        Deal with the data from IBM and grid search together.
        """
        if noise == 'ibm-1':
            assert seed == 1, "seed should be 1 for ibm-1"
            noisy_data = load_ibm_data(mid=1, seed=seed)
        elif noise == 'ibm-2':
            assert seed == 1, "seed should be 1 for ibm-2"
            noisy_data = load_ibm_data(mid=2, seed=seed)
        elif os.path.exists(noise): # ! special case
            assert seed == 1 and p == 1, "seed should be 1 and p should be 1 for local data"
            path = noise
            # path = "figs/grid_search/ibm/maxcut/sv-ideal-p=1/maxcut-sv-ideal-n=6-p=1-seed=1-50-100-IBM1-transpiled-H.npz"
            noisy_data = np.load(path, allow_pickle=True)
            noisy_data = dict(noisy_data)
            noisy_data['full_range'] = None # ! special case, only for visualization
        else:
            ansatz = 'qaoa'
            noisy_data, _, _ = load_grid_search_data(
                n_qubits=n_qubits, p=p, ansatz=ansatz,
                problem=problem, method=method,
                noise=noise, beta_step=50, gamma_step=100, seed=0,
            )

        return noisy_data


    for n_qubits in n_qubits_list:
        
        method = 'sv'
        problem = 'maxcut'
        # noise1 = 'depolar-0.001-0.005'
        # noise2 = 'depolar-0.003-0.007'
        # noise2 = 'depolar-0.001-0.02'
        # if noise1 == 'ibm':
        #     noisy_data1 = load_ibm_data(mid=1, seed=seed)
        #     noisy_data2 = load_ibm_data(mid=2, seed=seed)
        # else:
        #     noisy_data1, _, _ = load_grid_search_data(
        #         n_qubits=n_qubits, p=p, problem=problem, method=method,
        #         noise=noise1, beta_step=50, gamma_step=100, seed=0,
        #     )

        #     noisy_data2, _, _ = load_grid_search_data(
        #         n_qubits=n_qubits, p=p, problem=problem, method=method,
        #         noise=noise2, beta_step=50, gamma_step=100, seed=0,
        #     )
        noisy_data1 = get_data(noise1)
        noisy_data2 = get_data(noise2)

        if noisy_data1['full_range'] is not None:
            full_range = noisy_data1['full_range']
        else:
            full_range = noisy_data2['full_range']

        # depolarizing 0.001 and 0.005, one-qubit gate error and two-qubit gate error
        noisy1 = noisy_data1['data']

        # depolarizing 0.003 and 0.007, one-qubit gate error and two-qubit gate error
        noisy2 = noisy_data2['data'] 

        datas = [noisy1, noisy2]
        for data in datas:
            print(data.shape)
        
        rng = np.random.default_rng(0)
        landscape_shape = datas[0].shape
        n_pts = np.prod(landscape_shape)
        k = round(sf * n_pts)
        random_indices = rng.choice(np.prod(n_pts), k, replace=False) # random sample of indices

        # --------- data prepared OK, check if want to normalize -----------
        
        
        if normalize == 'geo':
            raise NotImplementedError("Current implement has some mistakes, refer to `linear`")
            n_normalize = round(norm_frac * n_pts)
            print("# points used to derive normalize ratio =", n_normalize)
            noisy1, noisy2, _ = normalize_by_geo_mean(noisy1, noisy2, n_normalize)
            datas = [noisy1, noisy2]
        elif normalize == 'linear':
            print("normalized by linear regression")
            n_normalize = round(norm_frac * sf * n_pts)
            # rng.choice(random_indices, n_normalize, replace=False)
            noisy1, noisy2 = normalize_by_linear_regression(noisy1, noisy2, n_normalize, random_indices)
            datas = [noisy1, noisy2]

        elif normalize == None:
            print("do not normalize")
        else:
            raise NotImplementedError()

        # ratios_cfg1 = [0.0, 0.25, 0.5, 0.75, 1.0]
        # ratios_cfg1 = np.linspace(norm_frac, 1.0, 5)
        ratios_cfg1 = [0.2, 0.5, 0.8, 1.0]
        print(ratios_cfg1)
        
        for ratio in ratios_cfg1:
            ratios = [ratio, 1-ratio]

            recon_fname = f"recon-n={n_qubits}-ratios={ratios}-sf={sf:.3f}-norm={normalize}-nf={norm_frac:.3f}"
            if not is_existing_recon:
                if p == 1:
                    # recon = np.zeros_like(datas[0])
                    recon = two_D_CS_p1_recon_with_distributed_landscapes(
                        origins=datas,
                        sampling_frac=sf,
                        ratios=[ratio, 1-ratio],
                        ri=random_indices
                    )
                elif p == 2:
                    raise NotImplementedError()
                
                np.savez_compressed(f"{recon_dir}/{recon_fname}", recon=recon)

            else:
                recon = np.load(f"{recon_dir}/{recon_fname}.npz", allow_pickle=True)['recon']


            if not is_existing_recon:
                _vis_recon_distributed_landscape(
                    landscapes=datas + [recon],
                    labels=[noise1, noise2, f'recon. by {ratios} of each'],
                    full_range=full_range,
                    bounds=None,
                    true_optima=None,
                    title=f'reconstruct distributed landscapes, sampling fraction: {sf:.3f}',
                    save_path=f"{recon_dir}/{recon_fname}.png"
                )

            # error1 = np.linalg.norm(noisy1 - recon)
            error1 = cal_recon_error(noisy1.reshape(-1), recon.reshape(-1), error_type)
            errors1.append(error1)

            # error2 = np.linalg.norm(noisy2 - recon)
            error2 = cal_recon_error(noisy2.reshape(-1), recon.reshape(-1), error_type)
            errors2.append(error2)

            # print(f"reconstruct error 1: {error1}; error 2: {error2}")

    print('ratio cfg:', ratios_cfg1)
    print("Sqrt MSE between recon and noise1's original:", errors1)
    print("Sqrt MSE between recon and noise2's original:", errors2)

    errors1 = np.array(errors1).reshape(len(n_qubits_list), len(ratios_cfg1))
    errors2 = np.array(errors2).reshape(len(n_qubits_list), len(ratios_cfg1))
    
    print('')
    if os.path.exists(noise2):
        noise2 = noise2.replace('/', '+')
    if os.path.exists(noise1):
        noise1 = noise1.replace('/', '+')
    save_path = f"{recon_dir}/dist_LS_errors-ns={n_qubits_list}-p={p}-sf={sf:.3f}-r={ratios_cfg1}-n1={noise1}-n2={noise2}-norm={normalize}-nf={norm_frac:.3f}-error={error_type}"
    print(f"recon errors save to {save_path}")
    np.savez_compressed(
        save_path,
        n_qubits_list=n_qubits_list,
        errors1=errors1,
        errors2=errors2,
        ratios=ratios_cfg1
    )

    return


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


def test_4D_CS():
    t = np.linspace(0, 1, 10000)
    origin = np.cos(2 * 97 * t * np.pi).reshape(10, 10, 10, 10)
    recon = recon_4D_landscape_by_2D(
        origin=origin,
        sampling_frac=0.05
    )

    error = np.linalg.norm(origin - recon)

    print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', type=int, help="Number of qubits", required=True)
    parser.add_argument('-p', type=int, help="QAOA circuit depth", required=True)
    parser.add_argument('--noise1', type=str)
    parser.add_argument('--noise2', type=str)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--sf', type=float, required=True)
    parser.add_argument('--normalize', type=str, default=None)
    parser.add_argument('--norm_frac', type=float, default=0)
    parser.add_argument('--ns', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('--error', type=str, required=True)
    parser.add_argument('--recon_dir', type=str, default=None)
    # parser.add_argument('-n', type=int, help="Number of qubits", required=True)
    # parser.add_argument('-p', type=str, help="QAOA layers")

    args = parser.parse_args()
    
    # gen_p1_landscape_top()
    # reconstruct_by_distributed_landscapes_top()
    # for nq in [12, 16, 20]:
    # for n in args.ns:
    reconstruct_by_distributed_landscapes_two_noisy_simulations_top(
        n_qubits_list=args.ns, p=args.p, sf=args.sf, noise1=args.noise1, noise2=args.noise2,
        seed=args.seed,
        normalize=args.normalize, norm_frac=args.norm_frac,
        error_type=args.error, recon_dir=args.recon_dir
    )
    # test_4D_CS()