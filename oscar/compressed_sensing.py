import imp
import itertools
from math import fabs
from re import L
import re
import time
from turtle import left, right
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
from mitiq import zne, Observable, PauliString
from mitiq.zne.zne import execute_with_zne
from mitiq.zne.inference import (
    LinearFactory
)
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    execute,
    execute_with_noise,
    execute_with_shots_and_noise
)

from mitiq.interface import convert_to_mitiq

from qiskit.quantum_info import Statevector
from sympy import beta, per

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
from .vis import (
    _vis_recon_distributed_landscape
)

# vis
import numpy as np
import matplotlib.pyplot as plt
from random import random, sample

# qiskit Landscape optimizer
from qiskit.algorithms.optimizers import (
    Optimizer, OptimizerResult
)

from qiskit.algorithms.optimizers.optimizer import POINT

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
    result = prob.solve(verbose=False)
    Xat = np.array(vx.value).squeeze()
    Xa = idct(Xat)
    return Xa


# ================== CS p==1 ==================

def _get_ideal_unmiti_miti_value_for_one_point(
    G,
    beta: list,
    gamma: list,
    shots: int,
    noise_model,
    ignore_miti: bool=False,
    ignore_unmiti: bool=False,
    ignore_ideal: bool=False,
    mitigation_params: dict=None
):
    """Generate ideal, unmitigated and mitigated energy for given one point.
    """
    circuit = get_maxcut_qaoa_circuit(
        G, beta=beta, gamma=gamma,
        # transpile_to_basis=True, save_state=False)
        transpile_to_basis=False, save_state=False)

    print("!")
    if ignore_ideal:
        ideal = 0
    else:
        ideal = _executor_of_qaoa_maxcut_energy(
            circuit.copy(), G, noise_model=None, shots=shots)
    
    if ignore_unmiti:
        unmiti = 0
    else:
        unmiti = _executor_of_qaoa_maxcut_energy(
            circuit.copy(), G, noise_model=copy.deepcopy(noise_model), shots=shots)
    
    if ignore_miti:
        miti = 0
    else:
        if isinstance(mitigation_params, dict):
            factory = mitigation_params['factory']
        else:
            factory = None

        miti = execute_with_zne(
            circuit.copy(),
            executor=partial(
                _executor_of_qaoa_maxcut_energy, G=G, noise_model=noise_model, shots=shots),
            factory=factory
        )
    # print(miti)

    return ideal, unmiti, miti


def _executor_of_qaoa_maxcut_energy(qc, G, noise_model, shots) -> float:
    """Generate mitigated QAOA MaxCut energy. For minimize.
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
    # ! for minimize
    return -expval
    # return expval


def _get_ideal_unmiti_miti_value_for_one_point_wrapper_for_concurrency(param):
    return _get_ideal_unmiti_miti_value_for_one_point(*param)


def gen_p1_landscape(
        G: nx.Graph,
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        mitigation_params: dict=None,
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
                [beta],
                [gamma],
                n_shots,
                copy.deepcopy(noise_model),
                False,
                True,
                True,
                copy.deepcopy(mitigation_params)
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
            _get_ideal_unmiti_miti_value_for_one_point_wrapper_for_concurrency, params
        )
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {(end_time - start_time) / 3600} h")

    for f in futures:
        # print(f)
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


def recon_2D_by_LASSO(nx, ny, A, b, alpha: float):
    vx = cvx.Variable(nx * ny)
    objective = cvx.Minimize(alpha * cvx.norm(vx, 1) + cvx.norm(A*vx - b, 2)**2)
    prob = cvx.Problem(objective)
    result = prob.solve(verbose=False)
    Xat2 = np.array(vx.value).squeeze()

    # reconstruct signal
    Xat = Xat2.reshape(nx, ny).T # stack columns
    Xa = idct2(Xat)
    return Xa


# def recon_2D_by_BPDN(nx, ny, A, b):
#     """Basic Pursuit Denoising.
#     """
#     vx = cvx.Variable(nx * ny)
#     objective = cvx.Minimize(cvx.norm(vx, 1))
#     prob = cvx.Problem(objective)
#     result = prob.solve(verbose=True)
#     Xat2 = np.array(vx.value).squeeze()

#     # reconstruct signal
#     Xat = Xat2.reshape(nx, ny).T # stack columns
#     Xa = idct2(Xat)

#     return Xa


def recon_2D_by_cvxpy(nx, ny, A, b):
    # do L1 optimization
    vx = cvx.Variable(nx * ny)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    # b = b.reshape(-1)
    constraints = [A*vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
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


def recon_2D_by_cvxpy_bak(X, sampling_frac: float, fig_dir: str):
    assert len(X.shape) == 2
    ny, nx = X.shape

    # extract small sample of signal
    k = round(nx * ny * sampling_frac)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
    b = X.T.flat[ri]
    # b = np.expand_dims(b, axis=1)

    # create dct matrix operator using kron (memory errors for large ny*nx)
    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
        )
    A = A[ri,:] # same as phi times kron

    # vx = recon_by_Lasso(A, b, 0.1)
    # Xat2 = vx

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

    # confirm solution
    if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
        print('Warning: values at sample indices don\'t match original.')

    # create images of mask (for visualization)

    X_max = np.max(X)
    mask = np.zeros(X.shape)
    mask.T.flat[ri] = X_max
    Xm = X_max * np.ones(X.shape)
    Xm.T.flat[ri] = X.T.flat[ri]

    if fig_dir:
        # plt.rc('font', size=20)
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))
        # fig, axs = plt.subplots(1, 4)
        axs = axs.reshape(-1)
        axs[0].imshow(X.T)
        # axs[1].imshow(mask.T)
        axs[1].imshow(Xa.T)
        im = axs[2].imshow(Xm.T)

        fig.colorbar(im, ax=[axs[i] for i in range(3)])
        plt.show()

    return Xa


def two_D_CS_p1_recon_with_distributed_landscapes(
    origins: List[np.ndarray],
    sampling_frac: float,
    ratios: list=None,
    ri: np.ndarray=None,
) -> None:
    rng = np.random.default_rng(0)
    """Reconstruct landscapes by sampling on distributed landscapes.

    Args:
        origins (List[np.ndarray]): List of full landscapes.
        sampling_frac (float): sampling fraction
        ratios (list, optional): Ratios of samples coming from each original landscapes. Defaults to None.
        ri (np.ndarray, optional): Random indices of original landscapes to do compressed sensing. Defaults to None.

    """
    # ! Convention: First beta, Last gamma
    if not isinstance(ratios, list):
        ratios = [1.0 / len(origins) for _ in range(len(origins))]
    else:
        assert len(ratios) == len(origins)
    assert np.isclose(sum(ratios), 1.0)
    
    print('start: solve l1 norm')
    ny, nx = origins[0].shape

    # extract small sample of signal
    
    k = round(nx * ny * sampling_frac)
    if not isinstance(ri, np.ndarray):
        ri = rng.choice(nx * ny, k, replace=False) # random sample of indices
    else:
        assert len(ri.shape) == 1 and ri.shape[0] == k
    
    print(f"ratios: {ratios}, k: {k}")
    # b = np.expand_dims(b, axis=1)

    # create dct matrix operator using kron (memory errors for large ny*nx)
    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
        )
    A = A[ri,:] # same as phi times kron

    # b = X.T.flat[ri]

    b = np.zeros(k)
    origins_T = np.array([
        o.T for o in origins
    ])
    # n_origins = len(origins)
    # for ik in range(k):
    #     which_origin = ik % n_origins
    #     b[ik] = origins_T[which_origin].flat[ri[ik]]
    
    for ik in range(k):
        # which_origin = ik % n_origins
        which_origin = rng.choice(len(origins), 1, p=ratios)[0]
        b[ik] = origins_T[which_origin].flat[ri[ik]]

    recon = recon_2D_by_cvxpy(nx, ny, A, b)

    print('end: solve l1 norm')
    return recon


# rename from recon_p1_landscape to recon_2D landscape
def recon_2D_landscape(
    origin: np.ndarray,
    sampling_frac: float,
    random_indices: np.ndarray=None,
    method='BP'
) -> np.ndarray:
    """Reconstruct a p==1 landscape.

    Args:
        origin (np.ndarray): original landscape.
        sampling_frac (float): sampling fraction of points that used to conduct compressed sensing.
        random_indices (np.ndarray, optional): indices that used to conduct compressed sensing.
        Can be specified by caller, otherwise generated by this function itself. Defaults to None.

    Returns:
        np.ndarray: reconstructed landscape
    """

    assert len(origin.shape) == 2

    ny, nx = origin.shape

    # extract small sample of signal
    k = round(nx * ny * sampling_frac)
    assert k > 0, "k should be positive, check sampling_frac"
    if not isinstance(random_indices, np.ndarray):
        ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
    else:
        print("use inputted random indices")
        assert len(random_indices.shape) == 1 and random_indices.shape[0] == k
        ri = random_indices # for short 

    # print(f"nx: {nx}, ny: {ny}, k: {k}")

    # create dct matrix operator using kron (memory errors for large ny*nx)
    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
        )
    A = A[ri,:] # same as phi times kron

    # b = X.T.flat[ri]
    if method == 'BP':
        recon = recon_2D_by_cvxpy(nx, ny, A, origin.T.flat[ri])
    elif method == 'BPDN':
        recon = recon_2D_by_LASSO(nx, ny, A, origin.T.flat[ri], 0.001)
    else:
        assert False, "Invalid CS method"

    return recon

# =============================== 4D cs =================================

def gen_p2_landscape(
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

    # in order of beta1 beta2 gamma1 gamma2
    full_ranges = []
    for ip in range(p):
        full_ranges.append(full_range['beta'].copy())
    
    for ip in range(p):
        full_ranges.append(full_range['gamma'].copy())

    # full_ranges = [
    #     full_range['beta'].copy(),
    #     full_range['beta'].copy(),
    #     full_range['gamma'].copy(),
    #     full_range['gamma'].copy()
    # ]

    params = []
    # former 2 are betas, latter 2 are gammas
    for beta2_gamma2 in itertools.product(*full_ranges):
        param = (
            G.copy(),
            beta2_gamma2[:p], # beta
            beta2_gamma2[p:], # gamma
            n_shots,
            copy.deepcopy(noise_model),
            True
        )
        params.append(param)
    
    print('totally number of points need to calculate energies:', len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # ! submit will shows exception, while map does not
        # future = executor.submit(
        #     _get_ideal_unmiti_miti_value_for_one_point_p1_wrapper_for_concurrency,
        #     *params[0]
        # )
        futures = executor.map(
            _get_ideal_unmiti_miti_value_for_one_point_wrapper_for_concurrency, params
        )
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {(end_time - start_time) / 3600} h")

    for f in futures:
        # print(f)
        origin['ideals'].append(f[0])
        origin['unmitis'].append(f[1])
        origin['mitis'].append(f[2])

    shape = []
    for ip in range(p):
        shape.append(n_pts['gamma'])
    for ip in range(p):
        shape.append(n_pts['beta'])

    for label, arr in origin.items():
        origin[label] = np.array(arr).reshape(*shape)
        print(origin[label].shape)

    np.savez_compressed(f"{figdir}/data",
        # landscapes
        origin=origin,

        # parameters
        shape=shape,
        n_pts=n_pts,
        full_range=full_range,
        full_ranges=full_ranges,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        C_opt=C_opt)

    print('generated landscapes data is saved')
    return


# ------------------------- helper functions

def cal_recon_error(x, x_recon, residual_type):
    # print(x.shape, x_recon.shape)

    assert len(x.shape) == 1 or len(x.shape) == 2 and x.shape[1] == 1
    assert len(x_recon.shape) == 1 or len(x_recon.shape) == 2 and x_recon.shape[1] == 1

    x = x.reshape(-1)
    x_recon = x_recon.reshape(-1)

    assert x.shape[0] == x_recon.shape[0]

    diff = x - x_recon

    if residual_type == 'MIN_MAX':
        res = np.sqrt((diff ** 2).mean()) / (x.max() - x.min())
    elif residual_type == 'MEAN':
        res = np.sqrt((diff ** 2).mean()) / x.mean()
    elif residual_type == 'MSE': # ! RMSE, Sqrt MSE, just let it be
        res = np.sqrt((diff ** 2).mean())
        # res = (diff ** 2).mean()
    elif residual_type == 'CROSS_CORRELATION':
        res = np.correlate(x_recon, x, mode='valid')
        # print(res)
        # assert np.isclose(res[0], np.sum(x_recon * x))
        res = res[0] / np.sqrt(np.sum(x_recon ** 2) * np.sum(x ** 2))
    elif residual_type == 'CONV':
        # print(x_recon.shape)
        # print(x_recon.shape, x.shape)
        res = np.convolve(x_recon, x) # take x as filter
        # print(res)
        res = res[0]
        # assert np.isclose(res[0])
        # return res[0]
    elif residual_type == 'NRMSE':
        res = np.sqrt((diff ** 2).mean())
        quantiles = np.nanquantile(x, q=(0.25, 0.5, 0.75))
        res /= (quantiles[2] - quantiles[0])
    elif residual_type == 'ZNCC':
        res = 0
        # res = np.sum((x_recon - x_recon.mean()) * (x - x.mean())) / np.sqrt(x_recon.var() * x.var())
    else:
        raise NotImplementedError(f"Invalid residual_type {residual_type}")

    return res