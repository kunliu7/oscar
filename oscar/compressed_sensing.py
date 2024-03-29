#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reconstruct 2D landscapes by compressed sensing.

Reference: http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/
"""

from typing import List

import cvxpy as cvx
import numpy as np
from scipy.fftpack import dct, idct


def L1_norm(x):
    return np.linalg.norm(x, ord=1)


def dct2(x):
    return dct(dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def recon_2D_by_LASSO(nx, ny, A, b, alpha: float):
    vx = cvx.Variable(nx * ny)
    objective = cvx.Minimize(
        alpha * cvx.norm(vx, 1) + cvx.norm(A*vx - b, 2)**2)
    prob = cvx.Problem(objective)
    result = prob.solve(verbose=False)
    Xat2 = np.array(vx.value).squeeze()

    Xat = Xat2.reshape(nx, ny).T  # stack columns
    Xa = idct2(Xat)
    return Xa


def recon_2D_by_cvxpy(nx, ny, A, b):
    vx = cvx.Variable(nx * ny)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A*vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    Xat2 = np.array(vx.value).squeeze()

    Xat = Xat2.reshape(nx, ny).T  # stack columns
    Xa = idct2(Xat)

    return Xa


def two_D_CS_p1_recon_with_distributed_landscapes(
    origins: List[np.ndarray],
    sampling_frac: float,
    ratios: list = None,
    ri: np.ndarray = None,
) -> np.ndarray:
    """Reconstruct landscapes by sampling on distributed landscapes.

    Args:
        origins (List[np.ndarray]): List of full landscapes.
        sampling_frac (float): sampling fraction
        ratios (list, optional): Ratios of samples coming from each original landscapes. Defaults to None.
        ri (np.ndarray, optional): Random indices of original landscapes to do compressed sensing. Defaults to None.

    Returns:
        nd.ndarray: Reconstructed landscape.
    """
    if not isinstance(ratios, list):
        ratios = [1.0 / len(origins) for _ in range(len(origins))]
    else:
        assert len(ratios) == len(origins)
    assert np.isclose(sum(ratios), 1.0)

    rng = np.random.default_rng(0)
    ny, nx = origins[0].shape

    # extract small sample of signal
    k = round(nx * ny * sampling_frac)
    if not isinstance(ri, np.ndarray):
        ri = rng.choice(nx * ny, k, replace=False)  # random sample of indices
    else:
        assert len(ri.shape) == 1 and ri.shape[0] == k

    print(f"sampling frac: {ratios}, k: {k}")

    # create dictionary using kron (memory errors for large ny*nx)
    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
    )
    A = A[ri, :]  # same as phi times kron

    b = np.zeros(k)
    origins_T = np.array([
        o.T for o in origins
    ])

    # randomly choose on which original landscape to sample based on ratios
    # if there is only one landscape, b = X.T.flat[ri]
    for ik in range(k):
        which_origin = rng.choice(len(origins), 1, p=ratios)[0]
        b[ik] = origins_T[which_origin].flat[ri[ik]]

    recon = recon_2D_by_cvxpy(nx, ny, A, b)
    return recon


def recon_2D_landscape(
    origin: np.ndarray,
    sampling_frac: float,
    random_indices: np.ndarray = None,
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
        # random sample of indices
        ri = np.random.choice(nx * ny, k, replace=False)
    else:
        print("use predefined random indices")
        assert len(random_indices.shape) == 1 and random_indices.shape[0] == k
        ri = random_indices  # for short

    # print(f"nx: {nx}, ny: {ny}, k: {k}")

    # create dictionary using kron (memory errors for large ny*nx)
    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
    )
    A = A[ri, :]  # same as phi times kron

    # b = X.T.flat[ri]
    if method == 'BP':
        recon = recon_2D_by_cvxpy(nx, ny, A, origin.T.flat[ri])
    elif method == 'BPDN':
        recon = recon_2D_by_LASSO(nx, ny, A, origin.T.flat[ri], 0.001)
    else:
        raise NotImplementedError(f"method {method} not supported")

    return recon


def cal_recon_error(x, x_recon, residual_type):
    # print(x.shape, x_recon.shape)
    assert len(x.shape) == 1 or len(x.shape) == 2 and x.shape[1] == 1
    assert len(x_recon.shape) == 1 or len(
        x_recon.shape) == 2 and x_recon.shape[1] == 1

    x = x.reshape(-1)
    x_recon = x_recon.reshape(-1)

    assert x.shape[0] == x_recon.shape[0]

    diff = x - x_recon

    if residual_type == 'MIN_MAX':
        res = np.sqrt((diff ** 2).mean()) / (x.max() - x.min())
    elif residual_type == 'MEAN':
        res = np.sqrt((diff ** 2).mean()) / x.mean()
    elif residual_type == 'RMSE':  # ! RMSE, Sqrt MSE, just let it be
        res = np.sqrt((diff ** 2).mean())
        # res = (diff ** 2).mean()
    elif residual_type == 'CROSS_CORRELATION':
        res = np.correlate(x_recon, x, mode='valid')
        res = res[0] / np.sqrt(np.sum(x_recon ** 2) * np.sum(x ** 2))
    elif residual_type == 'CONV':
        res = np.convolve(x_recon, x)  # take x as filter
        res = res[0]
    elif residual_type == 'NRMSE':
        res = np.sqrt((diff ** 2).mean())
        quantiles = np.nanquantile(x, q=(0.25, 0.5, 0.75))
        res /= (quantiles[2] - quantiles[0])
    elif residual_type == 'ZNCC':
        res = np.sum((x_recon - x_recon.mean()) * (x - x.mean())
                     ) / np.sqrt(x_recon.var() * x.var())
    else:
        raise NotImplementedError(f"Invalid residual_type {residual_type}")

    return res
