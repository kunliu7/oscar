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
from qiskit_aer import AerSimulator
from functools import partial
from pathlib import Path
import copy
import timeit
import sys, os
from scipy.fftpack import dct, diff, idct
from scipy.optimize import minimize
from sklearn import linear_model

from qiskit_aer.noise import NoiseModel
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


@DeprecationWarning
def solve_by_l1_norm(Theta, y):
    constr = ({'type': 'eq', 'fun': lambda x:  Theta @ x - y})
    x0 = np.linalg.pinv(Theta) @ y 
    res = minimize(L1_norm, x0, method='SLSQP',constraints=constr)
    s = res.x
    return s


@DeprecationWarning
def recon_by_Lasso(Theta, y, alpha):
    n = Theta.shape[1]
    lasso = linear_model.Lasso(alpha=alpha)# here, we use lasso to minimize the L1 norm
    # lasso.fit(Theta, y.reshape((M,)))
    lasso.fit(Theta, y)
    # Plotting the reconstructed coefficients and the signal
    # Creates the fourier transform that will most minimize l1 norm 
    recons = idct(lasso.coef_.reshape((n, 1)), axis=0)
    return recons + lasso.intercept_


@DeprecationWarning
def recon_1D_by_cvxpy(A, b):
    vx = cvx.Variable(A.shape[1])
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A*vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    Xat = np.array(vx.value).squeeze()
    Xa = idct(Xat)
    return Xa


# ============================ two D CS ====================
# reference: http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

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