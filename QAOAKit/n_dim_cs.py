import time
from typing import Callable, List, Optional, Tuple
import networkx as nx
import numpy as np
import cvxpy as cvx
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit import Aer
import qiskit
from qiskit.providers.aer import AerSimulator
from functools import partial
from pathlib import Path
import copy
import timeit
import sys, os
from scipy.fftpack import dct, diff, idct
from scipy.optimize import minimize
from sklearn import linear_model

from qiskit.providers.aer.noise import NoiseModel
import concurrent.futures
# from mitiq import zne, Observable, PauliString
# from mitiq.zne.zne import execute_with_zne
# from mitiq.interface.mitiq_qiskit.qiskit_utils import (
#     execute,
#     execute_with_noise,
#     execute_with_shots_and_noise
# )

# from mitiq.interface import convert_to_mitiq

from qiskit.quantum_info import Statevector
from sympy import beta, per

from QAOAKit.compressed_sensing import recon_2D_by_LASSO

from .qaoa import get_maxcut_qaoa_circuit
from .utils import (
    angles_to_qaoa_format,
    get_curr_formatted_timestamp,
    noisy_qaoa_maxcut_energy,
    angles_from_qiskit_format,
    maxcut_obj,
    get_adjacency_matrix,
    obj_from_statevector,
    qaoa_maxcut_energy
)
from .noisy_params_optim import (
    compute_expectation,
    get_depolarizing_error_noise_model
)

# vis
import numpy as np
import matplotlib.pyplot as plt
from random import sample

# qiskit Landscape optimizer
from qiskit.algorithms.optimizers import (
    Optimizer, OptimizerResult
)

from qiskit.algorithms.optimizers.optimizer import POINT

from QAOAKit import qaoa

def L1_norm(x):
    return np.linalg.norm(x,ord=1)

def solve_by_l1_norm(Theta, y):
    constr = ({'type': 'eq', 'fun': lambda x:  Theta @ x - y})
    x0 = np.linalg.pinv(Theta) @ y 
    res = minimize(L1_norm, x0, method='SLSQP',constraints=constr)
    s = res.x
    return s

def recon_by_Lasso(Theta, y, alpha):
    n = Theta.shape[1]
    lasso = linear_model.Lasso(alpha=alpha)# here, we use lasso to minimize the L1 norm
    # lasso.fit(Theta, y.reshape((M,)))
    lasso.fit(Theta, y)
    # Plotting the reconstructed coefficients and the signal
    # Creates the fourier transform that will most minimize l1 norm 
    recons = idct(lasso.coef_.reshape((n, 1)), axis=0)
    return recons + lasso.intercept_


def recon_1D_by_cvxpy(A, b):
    vx = cvx.Variable(A.shape[1])
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A*vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    Xat = np.array(vx.value).squeeze()
    Xa = idct(Xat)
    return Xa

# ================== CS p==1 ==================

def _vis_one_D_p1_recon(
        origin_dict,
        recon_dict,
        # gamma_range,
        # beta_range,
        # C_opt, bound, var_opt,
        bounds,
        full_range,
        true_optima,
        # mitis_recon,
        # unmitis_recon,
        # ideal_recon,
        # xlabel, 
        title,
        save_path,
        recon_params_path_dict=None,
        origin_params_path_dict=None
    ):

    # plt.figure
    plt.rc('font', size=28)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 30))
    fig.suptitle(title, y=0.92)
    axs = axs.reshape(-1)

    # TODO Check ij and xy
    X, Y = np.meshgrid(full_range['beta'], full_range['gamma'])

    # c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    idx = 0
    for label, origin in origin_dict.items():
        recon = recon_dict[label]
        # axs[idx]
        # Z = np.array(Z).T
        # c = axs[idx].pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
        
        # im = axs[idx].imshow(origin)
        im = axs[idx].pcolormesh(X, Y, origin) #, cmap='viridis', vmin=origin.min(), vmax=origin.max())
        axs[idx].set_title(f"origin, {label}")
        if true_optima:
            axs[idx].plot(true_optima[1], true_optima[0], marker="o", color='red', markersize=7, label="true optima")
        axs[idx].set_xlabel('beta')
        axs[idx].set_ylabel('gamma')
        # axs[idx].set_xlim(bottom=full_range['beta'][0], top=full_range['beta'][-1])
        # axs[idx].set_xlim(left=bounds['beta'][0], right=bounds['beta'][1])
        # axs[idx].set_ylim(bottom=bounds['gamma'][0], top=bounds['gamma'][1])

        # im = axs[idx + 3].imshow(recon)
        im = axs[idx + 3].pcolormesh(X, Y, recon)
        axs[idx + 3].set_title(f"recon, {label}")
        if true_optima:
            axs[idx + 3].plot(true_optima[1], true_optima[0], marker="o", color='red', markersize=7, label="true optima")
        axs[idx + 3].set_xlabel('beta')
        axs[idx + 3].set_ylabel('gamma')
        # axs[idx + 3].set_xlim(left=bounds['beta'][0], right=bounds['beta'][1])
        # axs[idx + 3].set_ylim(bottom=bounds['gamma'][0], top=bounds['gamma'][1])

        # origin
        if origin_params_path_dict and label in origin_params_path_dict:
            xs = [] # beta
            ys = [] # gamma
            for param in origin_params_path_dict[label]:
                xs.append(param[1])
                ys.append(param[0])

            axs[idx].plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
            axs[idx].plot(xs[0], ys[0], marker="o", color='white', markersize=9, label="initial point")
            axs[idx].plot(xs[-1], ys[-1], marker="s", color='white', markersize=12, label="last point")

        # recon
        if recon_params_path_dict and label in recon_params_path_dict:
            xs = [] # beta
            ys = [] # gamma
            for param in recon_params_path_dict[label]:
                xs.append(param[1])
                ys.append(param[0])

            axs[idx + 3].plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
            axs[idx + 3].plot(xs[0], ys[0], marker="o", color='white', markersize=9, label="initial point")
            axs[idx + 3].plot(xs[-1], ys[-1], marker="s", color='white', markersize=12, label="last point")
        
        
        idx += 1
    plt.legend()
    fig.colorbar(im, ax=[axs[i] for i in range(6)])
    # plt.title(title)
    # plt.subtitle(title)
    fig.savefig(save_path)
    plt.close('all')


def _get_ideal_unmiti_miti_value_for_one_point_p1(
    G,
    beta,
    gamma,
    shots,
    noise_model
):
    """Generate ideal, unmitigated and mitigated energy for given one point.
    """
    circuit = get_maxcut_qaoa_circuit(
        G, beta=[beta], gamma=[gamma],
        # transpile_to_basis=True, save_state=False)
        transpile_to_basis=False, save_state=False)

    ideal = _executor_of_qaoa_maxcut_energy(
        circuit.copy(), G, noise_model=None, shots=shots)
    unmiti = _executor_of_qaoa_maxcut_energy(
        circuit.copy(), G, noise_model=copy.deepcopy(noise_model), shots=shots)
    # miti = 0
    miti = execute_with_zne(
        circuit.copy(),
        executor=partial(
            _executor_of_qaoa_maxcut_energy, G=G, noise_model=None, shots=shots),
    )
    print(miti)

    return ideal, unmiti, miti


def _executor_of_qaoa_maxcut_energy(qc, G, noise_model, shots) -> float:
    """Generate mitigated QAOA MaxCut energy.
    """
    backend = AerSimulator()
    qc.measure_all()
    # backend = Aer.get_backend('aer_simulator')
    # noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model
    # noise_model = None
    counts = backend.run(
        qc,
        shots=shots,
        # noise_model=noise_model if is_noisy else None
        noise_model=noise_model
    ).result()

    counts = counts.get_counts()
    # print(counts)
    
    expval = compute_expectation(counts, G)
    # print(expval)
    return -expval
    # return expval


def _get_ideal_unmiti_miti_value_for_one_point_p1_wrapper_for_concurrency(param):
    return _get_ideal_unmiti_miti_value_for_one_point_p1(*param)


def one_D_CS_p1_recon_with_given_landscapes_and_varing_sampling_frac(
        figdir: str,
        origin: dict,
        # bounds: dict,
        full_range: dict,
        n_pts: dict,
        # n_samples: dict,
        # beta_opt: np.array, # converted
        # gamma_opt: np.array, # converted
        # noise_model: NoiseModel,
        # params_path: list,
        # C_opt: float,
        sampling_frac: float,
        alpha: float
    ):
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # alpha = 0.1
    # n_pts_per_unit = 36     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    # bounds = {'beta': [-np.pi/4, np.pi/4],
    #           'gamma': [-np.pi, np.pi]}

    n_samples = {}
    for label, _ in n_pts.items():
        n_samples[label] = np.ceil(n_pts[label] * sampling_frac).astype(int)
    
    # print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    print('n_samples: ', n_samples)
    print('alpha: ', alpha)
    # print('n_pts_per_unit: ', n_pts_per_unit)
    # sample P points from N randomly

    # mitis = []
    # unmitis = []
    # ideals = []

    _LABELS = ['mitis', 'unmitis', 'ideals']

    # x = np.cos(2 * 97 * np.pi * full_range) + np.cos(2 * 777 * np.pi * full_range)
    # x = np.cos(2 * np.pi * full_range) # + np.cos(2 * np.pi * full_range)
    
    print('start: solve l1 norm')
    recon = {label: [] for label in _LABELS}
    for idx_gamma, _ in enumerate(full_range['gamma']):
        Psi = dct(np.identity(n_pts['beta']))
        perm = np.floor(np.random.rand(n_samples['beta']) * n_pts['beta']).astype(int)

        # ideals_recon = recon_by_Lasso(Psi[perm, :], origin['ideals'][idx_gamma, perm], alpha)
        # unmitis_recon = recon_by_Lasso(Psi[perm, :], origin['unmitis'][idx_gamma, perm], alpha)
        # mitis_recon = recon_by_Lasso(Psi[perm, :], origin['mitis'][idx_gamma, perm], alpha)
        
        ideals_recon = recon_1D_by_cvxpy(Psi[perm, :], origin['ideals'][idx_gamma, perm])
        unmitis_recon = recon_1D_by_cvxpy(Psi[perm, :], origin['unmitis'][idx_gamma, perm])
        mitis_recon = recon_1D_by_cvxpy(Psi[perm, :], origin['mitis'][idx_gamma, perm])

        recon['ideals'].append(ideals_recon.copy())
        recon['unmitis'].append(unmitis_recon.copy())
        recon['mitis'].append(mitis_recon.copy())


    for label, arr in recon.items():
        recon[label] = np.array(arr)

    if figdir:
        _vis_one_D_p1_recon(
            origin_dict=origin,
            recon_dict=recon,
            title='test',
            save_path=f'{figdir}/origin_and_recon_sf{sampling_frac:.3f}_alpha{alpha:.3f}.png'
        )

    print('end: solve l1 norm')
    return recon


def gen_p1_landscape(
        G: nx.Graph,
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        bounds = {'beta': [-np.pi/4, np.pi/4],
                 'gamma': [-np.pi, np.pi]},
        n_shots: int=2048,
        n_pts_per_unit: int=36
    ):
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # n_shots = 2
    # n_pts_per_unit = 2     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    # bounds = {'beta': [-np.pi/4, np.pi/4],
    #           'gamma': [-np.pi, np.pi]}

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

    _LABELS = ['mitis', 'unmitis', 'ideals']
    origin = {label: [] for label in _LABELS}

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    params = []
    for gamma in full_range['gamma']:
        for beta in full_range['beta']:
            param = (
                G.copy(),
                beta,
                gamma,
                n_shots,
                copy.deepcopy(noise_model)
            )
            params.append(param)
    
    print('totally num. need to calculate energies:', len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # ! submit will shows exception, while map does not
        # future = executor.submit(
        #     _get_ideal_unmiti_miti_value_for_one_point_p1_wrapper_for_concurrency,
        #     *params[0]
        # )
        futures = executor.map(
            _get_ideal_unmiti_miti_value_for_one_point_p1_wrapper_for_concurrency, params
        )
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {(end_time - start_time) / 3600} h")

    for f in futures:
        print(f)
        origin['ideals'].append(f[0])
        origin['unmitis'].append(f[1])
        origin['mitis'].append(f[2])

    for label, arr in origin.items():
        origin[label] = np.array(arr).reshape(n_pts['gamma'], n_pts['beta'])
        print(origin[label].shape)

    np.savez_compressed(f"{figdir}/data",
        # landscapes
        origin=origin,

        # parameters
        n_pts=n_pts,
        full_range=full_range,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        C_opt=C_opt)

    print('generated landscapes data is saved')
    return


# @DeprecationWarning
def one_D_CS_p1_generate_landscape(
        G: nx.Graph, 
        p: int,
        figdir: str,
        beta_opt: np.ndarray, # converted
        gamma_opt: np.ndarray, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    params = []
    params.append((
        G.copy(),
        p,
        figdir,
        beta_opt.copy(),
        gamma_opt.copy(),
        noise_model.copy() if noise_model else None,
        params_path.copy(),
        C_opt
    ))
    
    # print('choose 10 randomly:', len(params))

    # print('start MP')
    # with concurrent.futures.ProcessPoolExecutor() as executor:
        # n_optima_list = executor.map(
        #     lambda x: vis_landscape_heatmap_multi_p_and_count_optima(*x), params, chunksize=16)
    # for param in params:
    #     future = executor.submit(
    #         # CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable,
    #         # CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable_and_sampling_frac,
    #         one_D_CS_p1_recon_task,
    #         *param
    #     )
    #     print(future.result())

    gen_p1_landscape(*params[0])
        
    return [], []

# ------------------------- generate gradient of landscape -------------------

def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=1):
    """
    We compute the gradient with the numeric differentiation in the parallel way,
    around the point x_center.

    Args:
        x_center (ndarray): point around which we compute the gradient
        f (func): the function of which the gradient is to be computed.
        epsilon (float): the epsilon used in the numeric differentiation.
        max_evals_grouped (int): max evals grouped
    Returns:
        grad: the gradient computed

    """
    forig = f(*((x_center,)))
    grad = []
    ei = np.zeros((len(x_center),), float)
    todos = []
    for k in range(len(x_center)):
        ei[k] = 1.0
        d = epsilon * ei
        todos.append(x_center + d)
        ei[k] = 0.0

    counter = 0
    chunk = []
    chunks = []
    length = len(todos)
    # split all points to chunks, where each chunk has batch_size points
    for i in range(length):
        x = todos[i]
        chunk.append(x)
        counter += 1
        # the last one does not have to reach batch_size
        if counter == max_evals_grouped or i == length - 1:
            chunks.append(chunk)
            chunk = []
            counter = 0

    for chunk in chunks:  # eval the chunks in order
        parallel_parameters = np.concatenate(chunk)
        todos_results = f(parallel_parameters)  # eval the points in a chunk (order preserved)
        if isinstance(todos_results, float):
            grad.append((todos_results - forig) / epsilon)
        else:
            for todor in todos_results:
                grad.append((todor - forig) / epsilon)

    return np.array(grad)


def qaoa_maxcut_energy_2b_wrapped(x, qc, G, is_noisy, shots):
    # x = [beta, gamma]
    n = x.shape[0]
    beta = x[:n]
    gamma = x[n:]
    energy = _executor_of_qaoa_maxcut_energy(qc.copy(), G, is_noisy=is_noisy, shots=shots)
    return energy


def _p1_grad_for_one_point(
    G,
    beta: float,
    gamma: float,
    shots: int
):
    qc = get_maxcut_qaoa_circuit(
        G, beta=[beta], gamma=[gamma],
        transpile_to_basis=True, save_state=False)

    x0 = np.array([beta, gamma])
    fun = partial(qaoa_maxcut_energy_2b_wrapped,
        qc=qc,
        G=G,
        is_noisy=False,
        shots=shots
    )
    grad = gradient_num_diff(x_center=x0, f=fun, epsilon=1e-10)
    return grad


def _p1_grad_for_one_point_mapper(param):
    return _p1_grad_for_one_point(*param)


def p1_generate_grad(
    G: nx.Graph,
    p: int,
    figdir: str,
    beta_opt: np.array, # converted
    gamma_opt: np.array, # converted
    noise_model: NoiseModel,
    params_path: list,
    C_opt: float
):
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # alpha = 0.1
    n_shots = 2048
    n_pts_per_unit = 36     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    bounds = {'beta': [-np.pi/4, np.pi/4],
              'gamma': [-np.pi, np.pi]}

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

    _LABELS = ['mitis', 'unmitis', 'ideals']
    origin = {label: [] for label in _LABELS}

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    params = []
    for gamma in full_range['gamma']:
        for beta in full_range['beta']:
            param = (
                G.copy(),
                beta,
                gamma,
                n_shots
            )
            params.append(param)
    
    print(len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # future = executor.submit(
        #     _p1_grad_for_one_point_mapper,
        #     params[0]
        # )
        futures = executor.map(
            _p1_grad_for_one_point_mapper, params
        )
    # print(future.result())
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {end_time - start_time} s")

    grads = []
    for f in futures:
        grads.append(f)

    grads = np.array(grads).reshape(n_pts['gamma'], n_pts['beta'], 2*p)

    # for f in futures:
        # origin['ideals'].append(f[0])
        # origin['unmitis'].append(f[1])
        # origin['mitis'].append(f[2])

    # for label, arr in origin.items():
        # origin[label] = np.array(arr).reshape(n_pts['gamma'], n_pts['beta'])
        # print(origin[label].shape)
        
    np.savez_compressed(f"{figdir}/grad_data",
        # ! gradient
        grads=grads,

        # ! reconstruct
        # origin=origin,
        # recon=recon,
        # mitis=mitis, unmitis=unmitis, ideals=ideals,
        # unmitis_recon=unmitis_recon, mitis_recon=mitis_recon, ideals_recon=ideals_recon,

        # ! parameters
        n_pts=n_pts,
        # n_samples=n_samples, sampling_frac=sampling_frac,
        # perm=perm,
        full_range=full_range,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        # ! n_optima
        # improved_n_optima=improved_n_optima,
        # improved_n_optima_recon=improved_n_optima_recon,
        C_opt=C_opt)

    return


# ============================ two D CS ====================
# tutorial: http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

def dct2(x):
    return dct(dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct4(x):
    def f(a):
        idct(np.transpose(idct(np.transpose(a), norm='ortho', axis=0)), norm='ortho', axis=0)

    return f(f(x))


def recon_2D_by_cvxpy(nx, ny, A, b):
    # do L1 optimization
    vx = cvx.Variable(nx * ny)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    # b = b.reshape(-1)
    constraints = [A*vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    Xat2 = np.array(vx.value).squeeze()

    # reconstruct signal
    Xat = Xat2.reshape(nx, ny).T # stack columns
    Xa = idct2(Xat)

    # # confirm solution
    # if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    #     print('Warning: values at sample indices don\'t match original.')

    # # create images of mask (for visualization)

    # X_max = np.max(X)
    # mask = np.zeros(X.shape)
    # mask.T.flat[ri] = X_max
    # Xm = X_max * np.ones(X.shape)
    # Xm.T.flat[ri] = X.T.flat[ri]

    # if fig_dir:
    #     # plt.rc('font', size=20)
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))
    #     # fig, axs = plt.subplots(1, 4)
    #     axs = axs.reshape(-1)
    #     axs[0].imshow(X.T)
    #     # axs[1].imshow(mask.T)
    #     axs[1].imshow(Xa.T)
    #     im = axs[2].imshow(Xm.T)

    #     fig.colorbar(im, ax=[axs[i] for i in range(3)])
    #     plt.show()

    return Xa


def recon_4D_by_cvxpy(shape, A, b):
    # do L1 optimization
    vx = cvx.Variable(np.prod(shape))
    objective = cvx.Minimize(cvx.norm(vx, 1))
    # b = b.reshape(-1)
    constraints = [A*vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    Xat4 = np.array(vx.value).squeeze()

    # return Xat4

    # reconstruct signal
    # Xat = Xat2.reshape(nx, ny).T # stack columns
    Xat = np.transpose(Xat4.reshape(*shape))
    Xa = idct4(Xat)

    return Xa


def recon_4D_landscape_by_2D(
    origin: np.ndarray,
    sampling_frac: float,
    random_indices: np.ndarray=None,
    method: str="BP"
) -> np.ndarray:
    """Reconstruct landscapes by sampling on given landscapes.

    """
    # ! Convention: First beta, Last gamma
    shape_4d = origin.shape
    origin_2d = origin.reshape(shape_4d[0] * shape_4d[1], 
        shape_4d[2] * shape_4d[3])
    
    ny, nx = origin_2d.shape

    n_pts = np.prod(shape_4d)

    print(f"total samples: {n_pts}")

    # extract small sample of signal
    k = round(n_pts * sampling_frac)
    if not isinstance(random_indices, np.ndarray):
        ri = np.random.choice(n_pts, k, replace=False) # random sample of indices
    else:
        print("use inputted random indices")
        assert len(random_indices.shape) == 1 and random_indices.shape[0] == k
        ri = random_indices # for short 

    # create dct matrix operator using kron (memory errors for large ny*nx)
    # idct_list = [idct(np.identity(dim), norm='ortho', axis=0) for dim in shape]
    # A = idct_list[0]
    # for i in range(1, 4):
    #     A = np.kron(A, idct_list[i])

    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0),
    )
    A = A[ri,:] # same as phi times kron

    # b = X.T.flat[ri]
    # recon = recon_4D_by_cvxpy(shape, A, origin.T.flat[ri])
    b = origin_2d.T.flat[ri]
    if method == 'BP':
        recon = recon_2D_by_cvxpy(nx, ny, A, b)
    elif method == 'BPDN':
        recon = recon_2D_by_LASSO(nx, ny, A, b, 0.001)
    else:
        assert False, "Invalid CS method"

    recon = recon.reshape(*shape_4d)

    print('end: solve l1 norm')
    return recon



# ! Error
def recon_4D_landscape(
    figdir: str,
    origin: np.ndarray,
    full_range: dict,
    # n_pts: dict,
    sampling_frac: float
):
    """Reconstruct landscapes by sampling on given landscapes.

    """
    # ! Convention: First beta, Last gamma
    
    origin = origin.reshape()
    
    # ny, nx = origin.shape

    shape = origin.shape

    n_pts = np.prod(shape)

    # extract small sample of signal
    k = round(n_pts * sampling_frac)
    ri = np.random.choice(n_pts, k, replace=False) # random sample of indices

    # create dct matrix operator using kron (memory errors for large ny*nx)
    idct_list = [idct(np.identity(dim), norm='ortho', axis=0) for dim in shape]
    A = idct_list[0]
    for i in range(1, 4):
        A = np.kron(A, idct_list[i])

    # A = np.kron(
    #     idct(np.identity(nx), norm='ortho', axis=0),
    #     idct(np.identity(ny), norm='ortho', axis=0),
    #     )
    A = A[ri,:] # same as phi times kron

    # b = X.T.flat[ri]
    recon = recon_4D_by_cvxpy(shape, A, origin.T.flat[ri])

    # if figdir:
    #     _vis_one_D_p1_recon(
    #         origin_dict=origin,
    #         recon_dict=recon,
    #         full_range=full_range,
    #         bounds=None,
    #         true_optima=None,
    #         title='test',
    #         save_path=f'{figdir}/origin_and_2D_recon_sf{sampling_frac:.3f}.png'
    #     )

    print('end: solve l1 norm')
    return recon

