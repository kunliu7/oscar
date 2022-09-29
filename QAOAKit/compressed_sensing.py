import imp
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

# @DeprecationWarning()
def cosamp(phi, u, s, epsilon=1e-10, max_iter=1000):
    """
    Return an `s`-sparse approximation of the target signal
    Input:
        - phi, sampling matrix
        - u, noisy sample vector
        - s, sparsity
    
    cosamp function is available at https://github.com/avirmaux/CoSaMP
    """
    a = np.zeros(phi.shape[1])
    v = u
    it = 0 # count
    halt = False
    while not halt:
        it += 1
        
        y = np.dot(np.transpose(phi), v)
        omega = np.argsort(y)[-(2*s):] # large components
        omega = np.union1d(omega, a.nonzero()[0]) # use set instead?
        phiT = phi[:, omega]
        b = np.zeros(phi.shape[1])
        # Solve Least Square
        b[omega], _, _, _ = np.linalg.lstsq(phiT, u)
        
        # Get new estimate
        b[np.argsort(b)[:-s]] = 0
        a = b
        
        # Halt criterion
        v_old = v
        v = u - np.dot(phi, a)

        diff_old = np.linalg.norm(v - v_old)
        halt = (diff_old  < epsilon) or \
            np.linalg.norm(v) < epsilon or \
            it > max_iter
        
        print(f'iter = {it}, {diff_old:.3f}\r', end='')
    return a

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


# @DeprecationWarning()
def recon_by_CoSaMP(y, Psi, perm, sparsity):
    # n = x.shape[0]
    # xt = np.fft.fft(x) # Fourier transformed signal
    # PSD = xt * np.conj(xt) / n # Power spectral density
    # p = n // 2 # num. random samples, p = n/32
    # perm = np.floor(np.random.rand(p) * n).astype(int)
    # y = x[perm]

    # reconstruct
    # Psi = dct(np.identity(n)) # Build Psi
    Theta = Psi[perm,:]       # Measure rows of Psi
    s = cosamp(Theta, y, sparsity, epsilon=1.e-10, max_iter=200) # CS via matching pursuit
    # xrecon = idct(s) # reconstruct full signal
    xrecon = np.fft.ifft(s)
    return xrecon


# @DeprecationWarning()
def _vis_y_and_x_recon(
        C_opt, bound, var_opt,
        recon_range, x_recon,
        sample_range, x_sample,
        full_range, x_full,
        xlabel, 
        title,
        save_path
    ):
    fig, ax = plt.subplots()

    ax.hlines(y=C_opt, xmin=bound[0], xmax=bound[1], colors=['gray'], label='C optimal', linestyles=['dashed'], linewidth=1)
    ax.plot(var_opt, C_opt, marker="o", color='red', markersize=5, label='optimal point')
    # print(unmitis)
    # ax.plot(full_range, unmitis_recon, marker="o", markersize=2, label="unmitigated, recon")
    ax.scatter(recon_range, x_recon, marker="o", s=2, label="reconstruct", c='blue')
    ax.scatter(sample_range, x_sample, marker="x", label="sample", c='red')
    ax.plot(full_range, x_full, marker="*", markersize=2, label='full', color='green')
    # ax.scatter(full_range, unmitis_recon, marker="o", s=2, label="unmitigated, recon")
    
    ax.set_ylabel('energy or cost value')
    ax.set_xlabel(xlabel)
    # ax.set_xlim(full_range[0], full_range[4096])
    ax.set_ylim(-1, C_opt + 2)
    ax.set_title(title)
    ax.legend()

    fig.savefig(save_path)
    plt.close('all')


def _vis_miti_and_unmiti_recon(
        C_opt, bound, var_opt,
        full_range,
        mitis_recon,
        unmitis_recon,
        ideal_recon,
        xlabel, 
        title,
        save_path
    ):
    fig, ax = plt.subplots()

    ax.hlines(y=C_opt, xmin=bound[0], xmax=bound[1], colors=['gray'], label='C optimal', linestyles=['dashed'], linewidth=1)
    ax.plot(var_opt, C_opt, marker="o", color='r', markersize=5, label='optimal point')
    # print(unmitis)
    ax.scatter(full_range, mitis_recon, marker="o", s=2, c='orange', label="mitigated, recon")
    ax.scatter(full_range, unmitis_recon, marker="x", s=2, c='blue', label="unmitigated, recon")
    ax.plot(full_range, ideal_recon, marker='*', markersize=2, color='green', label="ideal recon")
    
    ax.set_ylabel('energy or cost value')
    ax.set_xlabel(xlabel)
    # ax.set_xlim(full_range[0], full_range[4096])
    ax.set_ylim(-1, C_opt + 2)
    ax.set_title(title)
    ax.legend()

    fig.savefig(save_path)
    plt.close('all')




# @DeprecationWarning()
def CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable(
        G: nx.Graph,
        p: int,
        figdir: str,
        var_idx: int, # p==2, 0-3 -> beta0, beta1, gamma0, gamma1
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        sampling_frac: float
    ):
    # ! Convention: First beta, Last gamma


    # ! executor that mitiq wants. only QuantumCircuit is needed
    # ! get_statevector() needs qc.save_state(),
    # and thus qc.qasm() could not be called
    # and thus could not be transformed to mitiq.circuit

    # obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    # def mitiq_executor_of_qaoa_maxcut_energy(qc) -> float:
    #     backend = AerSimulator(method="statevector")
    #     noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model
    #     sv = backend.run(
    #         qc,
    #         noise_model=noise_model
    #     ).result().get_statevector()
    #     return obj_from_statevector(sv, obj, precomputed_energies=None)

    def mitiq_executor_of_qaoa_maxcut_energy(qc) -> float:
        backend = AerSimulator()
        qc.measure_all()
        # backend = 
        # backend = Aer.get_backend('aer_simulator')
        noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model
        # noise_model = None
        counts = backend.run(
            qc,
            shots=2048,
            noise_model=noise_model
        ).result().get_counts()
        
        expval = compute_expectation(counts, G)
        return -expval

    # for edge in G.edges:
    #     obs = Observable(PauliString("I" * n)) #, PauliString("X", coeff=-1.75))
    # obs = Observable(PauliString("I"), PauliString("X", coeff=-1.75))

    # n = G.number_of_nodes()
    # def mitiq_executor_of_qaoa_maxcut_energy(qc) -> float:
    #     noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model 
    #     return execute_with_shots_and_noise(qc, obs.matrix(), noise_model, shots=1024)
        
    assert var_idx >= 0 and var_idx < 2*p
    
    # beta first, gamma later
    b_or_g = 'beta' if var_idx < p else 'gamma'
    bounds = {'beta': [-np.pi/4, np.pi/4], 'gamma': [-np.pi, np.pi]}
    bound = bounds[b_or_g]
    bound_len = bound[1] - bound[0]

    # hyper parameters
    n_pts = 36     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    sparsity = 4 # n_samples // 4       # K-sparse of s per unit
    n_samples = int(n_pts * sampling_frac) # 12  # num. of random samples per unit
    window = np.array([1024, 1280]) / n_pts

    # extend by bound_len
    n_pts = np.floor(n_pts * bound_len).astype(int)
    # n_samples = np.floor(sparsity * np.log(n_pts / sparsity)).astype(int)
    n_samples = np.floor(n_samples * bound_len).astype(int)
    # sparsity = np.floor(sparsity * bound_len).astype(int)
    # window = np.floor(window * bound_len).astype(int)

    print(f'n_pts={n_pts}, n_samples={n_samples}')

    # return

    # sample P points from N randomly
    perm = np.floor(np.random.rand(n_samples) * n_pts).astype(int)

    full_range = np.linspace(bound[0], bound[1], n_pts)
    # full_range = np.linspace(0, 1, n_pts)
    var_range = full_range[perm]

    # labels
    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])
    _X_Y_LABELS = [f'beta{i}' for i in range(p)]
    _X_Y_LABELS.extend([f'gamma{i}' for i in range(p)])

    mitis = []
    unmitis = []

    for v in var_range:
        tmp_vars = _BETA_GAMMA_OPT.copy()
        tmp_vars[var_idx] = v

        circuit = get_maxcut_qaoa_circuit(
            G, beta=tmp_vars[:p], gamma=tmp_vars[p:],
            transpile_to_basis=True, save_state=False)

        unmiti = mitiq_executor_of_qaoa_maxcut_energy(circuit.copy())
        miti = zne.execute_with_zne(
            circuit=circuit.copy(),
            executor=mitiq_executor_of_qaoa_maxcut_energy,
        )

        mitis.append(miti)
        unmitis.append(unmiti)

    # x = np.cos(2 * 97 * np.pi * full_range) + np.cos(2 * 777 * np.pi * full_range)
    # x = np.cos(2 * np.pi * full_range) # + np.cos(2 * np.pi * full_range)
    # mitis = x[perm]
    # unmitis = x[perm]
    mitis = np.array(mitis)
    unmitis = np.array(unmitis)

    Psi = dct(np.identity(n_pts)) # Build Psi
    
    # unmitis_recon = recon(unmitis, Psi, perm, sparsity)
    # mitis_recon = recon(mitis, Psi, perm, sparsity)
    
    # print('start: solve l1 norm')
    unmitis_recon = recon_by_Lasso(Psi[perm, :], unmitis)
    mitis_recon = recon_by_Lasso(Psi[perm, :], mitis)
    # print('end: solve l1 norm')
    
    # ----- count optima

    max_unmiti = unmitis.max()
    improved_n_optima = np.sum(
        # np.isclose(unmitis_recon, C_opt, atol=C_opt-max_miti) == True
        mitis > max_unmiti
    )
    
    max_unmiti_recon = unmitis_recon.max()
    improved_n_optima_recon = np.sum(
        # np.isclose(mitis_recon, C_opt, atol=abs(C_opt-max_unmiti_recon)) == True
        mitis_recon > max_unmiti_recon
    )

    print('improved_n_optima, recon', improved_n_optima, improved_n_optima_recon)

    # =============== vis unmiti ===============

    _vis_y_and_x_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
        full_range=full_range, var_range=var_range,
        x_recon=mitis_recon, y=mitis,
        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, recon miti',
        save_path=f'{figdir}/zne_varIdx={var_idx}_mitis_recon.png'
    )
    
    _vis_y_and_x_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
        full_range=full_range, var_range=var_range,
        x_recon=unmitis_recon, y=unmitis,
        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, recon unmiti',
        save_path=f'{figdir}/zne_varIdx={var_idx}_unmitis_recon.png'
    )

    _vis_miti_and_unmiti_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
        full_range=full_range,
        mitis_recon=mitis_recon,
        unmitis_recon=unmitis_recon,
        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, reconstructed unmiti and miti',
        save_path=f'{figdir}/zne_varIdx={var_idx}_unmitis_recon_and_mitis_recon.png'
    )

    np.savez_compressed(f"{figdir}/varIdx{var_idx}",
        n_pts=n_pts, n_samples=n_samples, sampling_frac=sampling_frac,
        mitis=mitis, unmitis=unmitis,
        unmitis_recon=unmitis_recon, mitis_recon=mitis_recon,
        perm=perm,
        improved_n_optima=improved_n_optima,
        improved_n_optima_recon=improved_n_optima_recon,
        C_opt=C_opt)

    return


# @DeprecationWarning()
def CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable_and_sampling_frac(
        G: nx.Graph,
        p: int,
        figdir: str,
        var_idx: int, # p==2, 0-3 -> beta0, beta1, gamma0, gamma1
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        sampling_frac: float
    ):
    # ! Convention: First beta, Last gamma

    def mitiq_executor_of_qaoa_maxcut_energy(qc, is_noisy) -> float:
        backend = AerSimulator()
        qc.measure_all()
        # backend = Aer.get_backend('aer_simulator')
        noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model
        counts = backend.run(
            qc,
            shots=2048,
            noise_model=noise_model if is_noisy else None
        ).result().get_counts()
        
        expval = compute_expectation(counts, G)
        return -expval

    assert var_idx >= 0 and var_idx < 2*p
    
    # beta first, gamma later
    b_or_g = 'beta' if var_idx < p else 'gamma'
    bounds = {'beta': [-np.pi/4, np.pi/4], 'gamma': [-np.pi, np.pi]}
    bound = bounds[b_or_g]
    bound_len = bound[1] - bound[0]

    # hyper parameters
    n_pts_per_unit = 36     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    # sparsity = 4 # n_samples // 4       # K-sparse of s per unit
    # n_samples = int(n_pts * sampling_frac) # 12  # num. of random samples per unit
    # window = np.array([1024, 1280]) / n_pts

    # extend by bound_len
    n_pts = np.floor(n_pts_per_unit * bound_len).astype(int)
    # n_samples = np.floor(sparsity * np.log(n_pts / sparsity)).astype(int)
    n_samples = np.ceil(n_pts_per_unit * bound_len * sampling_frac).astype(int)
    # sparsity = np.floor(sparsity * bound_len).astype(int)
    # window = np.floor(window * bound_len).astype(int)

    print(f'n_pts={n_pts}, n_samples={n_samples}')

    # sample P points from N randomly
    perm = np.floor(np.random.rand(n_samples) * n_pts).astype(int)

    full_range = np.linspace(bound[0], bound[1], n_pts)
    # full_range = np.linspace(0, 1, n_pts)
    # var_range = full_range[perm]

    # labels
    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])
    _X_Y_LABELS = [f'beta{i}' for i in range(p)]
    _X_Y_LABELS.extend([f'gamma{i}' for i in range(p)])

    mitis = []
    unmitis = []
    ideals = []

    for v in full_range:
        tmp_vars = _BETA_GAMMA_OPT.copy()
        tmp_vars[var_idx] = v

        circuit = get_maxcut_qaoa_circuit(
            G, beta=tmp_vars[:p], gamma=tmp_vars[p:],
            transpile_to_basis=True, save_state=False)

        ideal = mitiq_executor_of_qaoa_maxcut_energy(circuit.copy(), is_noisy=False)
        unmiti = mitiq_executor_of_qaoa_maxcut_energy(circuit.copy(), is_noisy=True)
        miti = zne.execute_with_zne(
            circuit=circuit.copy(),
            executor=partial(mitiq_executor_of_qaoa_maxcut_energy, is_noisy=True),
        )

        mitis.append(miti)
        unmitis.append(unmiti)
        ideals.append(ideal)

    # x = np.cos(2 * 97 * np.pi * full_range) + np.cos(2 * 777 * np.pi * full_range)
    # x = np.cos(2 * np.pi * full_range) # + np.cos(2 * np.pi * full_range)
    # mitis = x[perm]
    # unmitis = x[perm]
    mitis = np.array(mitis)
    unmitis = np.array(unmitis)
    ideals = np.array(ideals)

    Psi = dct(np.identity(n_pts)) # Build Psi
    
    # print('start: solve l1 norm')
    ideals_recon = recon_by_Lasso(Psi[perm, :], ideals[perm])
    unmitis_recon = recon_by_Lasso(Psi[perm, :], unmitis[perm])
    mitis_recon = recon_by_Lasso(Psi[perm, :], mitis[perm])
    # print('end: solve l1 norm')
    
    # ----- count optima

    max_unmiti = unmitis.max()
    improved_n_optima = np.sum(
        # np.isclose(unmitis_recon, C_opt, atol=C_opt-max_miti) == True
        mitis > max_unmiti
    )
    
    max_unmiti_recon = unmitis_recon.max()
    improved_n_optima_recon = np.sum(
        # np.isclose(mitis_recon, C_opt, atol=abs(C_opt-max_unmiti_recon)) == True
        mitis_recon > max_unmiti_recon
    )

    print('improved_n_optima, recon', improved_n_optima, improved_n_optima_recon)

    # =============== vis unmiti ===============

    _vis_y_and_x_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],

        recon_range=full_range, x_recon=mitis_recon,
        sample_range=full_range[perm], x_sample=mitis[perm],
        full_range=full_range, x_full=mitis,

        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, recon miti',
        save_path=f'{figdir}/zne_varIdx={var_idx}_mitis_recon.png'
    )
    
    _vis_y_and_x_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
        
        recon_range=full_range, x_recon=unmitis_recon,
        sample_range=full_range[perm], x_sample=unmitis[perm],
        full_range=full_range, x_full=unmitis,

        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, recon unmiti',
        save_path=f'{figdir}/zne_varIdx={var_idx}_unmitis_recon.png'
    )
    
    _vis_y_and_x_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],

        recon_range=full_range, x_recon=ideals_recon,
        sample_range=full_range[perm], x_sample=ideals[perm],
        full_range=full_range, x_full=ideals,

        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, recon ideal',
        save_path=f'{figdir}/zne_varIdx={var_idx}_ideal.png'
    )

    _vis_miti_and_unmiti_recon(
        C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
        full_range=full_range,
        mitis_recon=mitis_recon,
        unmitis_recon=unmitis_recon,
        ideal_recon=ideals_recon,
        xlabel=_X_Y_LABELS[var_idx],
        title=f'QAOA energy, nQ{G.number_of_nodes()}, reconstructed unmiti and miti',
        save_path=f'{figdir}/zne_varIdx={var_idx}_unmitis_recon_and_mitis_recon.png'
    )

    # ======== save ==========

    np.savez_compressed(f"{figdir}/varIdx{var_idx}",

        # reconstruct
        mitis=mitis, unmitis=unmitis, ideals=ideals,
        unmitis_recon=unmitis_recon, mitis_recon=mitis_recon, ideals_recon=ideals_recon,

        # parameters
        n_pts=n_pts, n_samples=n_samples, sampling_frac=sampling_frac,
        perm=perm, full_range=full_range,

        # n_optima
        improved_n_optima=improved_n_optima,
        improved_n_optima_recon=improved_n_optima_recon,
        C_opt=C_opt)

    return


# @DeprecationWarning()
def multi_landscapes_and_cnt_optima_and_mitiq_and_MP_and_one_variable_and_CS(
        G: nx.Graph, 
        p: int,
        figdir: str,
        beta_opt: np.ndarray, # converted
        gamma_opt: np.ndarray, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        executor: concurrent.futures.ProcessPoolExecutor,
        sampling_frac: float
    ):

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    params = []
    for i in range(2*p):
        params.append((
            G.copy(),
            p,
            figdir,
            i,
            beta_opt.copy(),
            gamma_opt.copy(),
            noise_model.copy() if noise_model else None,
            params_path.copy(),
            C_opt,
            sampling_frac
        ))
    
    print('choose 10 randomly:', len(params))

    print('start MP')
    # with concurrent.futures.ProcessPoolExecutor() as executor:
        # n_optima_list = executor.map(
        #     lambda x: vis_landscape_heatmap_multi_p_and_count_optima(*x), params, chunksize=16)
    for param in params:
        future = executor.submit(
            # CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable,
            CS_and_one_landscape_and_cnt_optima_and_mitiq_and_one_variable_and_sampling_frac,
            *param
        )
        # print(future.result())
        
    return [], []


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


def wrap_qiskit_optimizer_to_landscape_optimizer(QiskitOptimizer):
    class LandscapeOptimizer(QiskitOptimizer):
    
        """Override some methods based on existing Optimizers.
        Since we have existing landscape, we do not need to actually execute the circuit.

        Dynamic inheritance: https://stackoverflow.com/a/21060094/13392267
        """
        def __init__(self, bounds, landscape, **kwargs) -> None:
            # https://blog.csdn.net/sunny_happy08/article/details/82588749
            print("kwargs", kwargs)
            super(LandscapeOptimizer, self).__init__(**kwargs)
            self.bounds = bounds
            self.landscape = - landscape
            self.landscape_shape = np.array(landscape.shape)

            # https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
            bound_lens = np.apply_along_axis(lambda bound: bound[1] - bound[0], axis=1, arr=bounds)
            print('bound_lens', bound_lens)
            self.grid_lens = bound_lens / self.landscape_shape # element-wise
            print(self.grid_lens)
            self.params_path = []

        # def minimize(self, 
        #     fun: Callable[[POINT], float], 
        #     x0: POINT, 
        #     jac: Optional[Callable[[POINT], POINT]] = None,
        #     bounds: Optional[List[Tuple[float, float]]] = None):
        #     return super().minimize(fun, x0, jac, bounds)

        def minimize(self, 
            fun: Callable[[POINT], float], 
            x0: POINT, 
            jac: Optional[Callable[[POINT], POINT]] = None,
            bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizerResult:
            """Override existing optimizer's minimize function

            """
            def query_fun_value_from_landscape(x: POINT) -> float:
                angles = angles_from_qiskit_format(x)
                angles = angles_to_qaoa_format(angles)

                x = np.concatenate([angles['gamma'], angles['beta']])
                # print('transformed x', x)
                # assert x.shape == self.grid_shapes.shape
                # print(type(bounds))
                relative_indices = np.around((x - self.bounds[:, 0]) / self.grid_lens).astype(int)
                normalized_indices = relative_indices.copy()

                for axis, size in enumerate(self.landscape_shape):
                    # print(axis, size)
                    relative_index = relative_indices[axis]
                    if relative_index >= size:
                        normalized_indices[axis] = relative_index % size
                    elif relative_index < 0:
                        normalized_indices[axis] = relative_index + ((-relative_index - 1) // size + 1) * size

                    if normalized_indices[axis] < 0 or normalized_indices[axis] >= size:
                        print(axis, size, relative_index, normalized_indices[axis])
                        assert False

                # print(normalized_indices)
                # print(self.landscape_shape)
                # obj = *(list(normalized_indices))
                approximate_point = self.landscape[tuple(normalized_indices)]

                # print(approximate_point)
                self.params_path.append(x) # Qiskit format
                return approximate_point
                
            # print(self.callback)
            # print(super().callback)
            res = super().minimize(query_fun_value_from_landscape, x0, jac, bounds)
            print('res', res)
            print(jac)
            return res

    return LandscapeOptimizer


def two_D_CS_p1_recon_with_given_landscapes(
    figdir: str,
    origin: dict,
    full_range: dict,
    # n_pts: dict,
    sampling_frac: float
):
    """Reconstruct landscapes by sampling on given landscapes.

    Args:
        figdir (str): _description_
        origin (dict): _description_
        full_range (dict): _description_
        sampling_frac (float): _description_

    Returns:
        _type_: _description_
    """
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # alpha = 0.1
    # n_pts_per_unit = 36     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    # bounds = {'beta': [-np.pi/4, np.pi/4],
    #           'gamma': [-np.pi, np.pi]}

    # n_samples = {}
    # for label, _ in n_pts.items():
    #     n_samples[label] = np.ceil(n_pts[label] * sampling_frac).astype(int)
    
    # print('bounds: ', bounds)
    # print('n_pts: ', n_pts)
    # print('n_samples: ', n_samples)
    # print('alpha: ', alpha)
    # print('n_pts_per_unit: ', n_pts_per_unit)
    # sample P points from N randomly

    # mitis = []
    # unmitis = []
    # ideals = []

    # _LABELS = ['mitis', 'unmitis', 'ideals']
    _LABELS = []
    for label in origin.keys():
        _LABELS.append(label)

    # x = np.cos(2 * 97 * np.pi * full_range) + np.cos(2 * 777 * np.pi * full_range)
    # x = np.cos(2 * np.pi * full_range) # + np.cos(2 * np.pi * full_range)
    
    print('start: solve l1 norm')
    recon = {label: [] for label in _LABELS}
    
    # for idx_gamma, _ in enumerate(full_range['gamma']):
    #     Psi = dct(np.identity(n_pts['beta']))
    #     perm = np.floor(np.random.rand(n_samples['beta']) * n_pts['beta']).astype(int)

    #     ideals_recon = recon_by_Lasso(Psi[perm, :], origin['ideals'][idx_gamma, perm], alpha)
    #     unmitis_recon = recon_by_Lasso(Psi[perm, :], origin['unmitis'][idx_gamma, perm], alpha)
    #     mitis_recon = recon_by_Lasso(Psi[perm, :], origin['mitis'][idx_gamma, perm], alpha)

    #     recon['ideals'].append(ideals_recon.copy())
    #     recon['unmitis'].append(unmitis_recon.copy())
    #     recon['mitis'].append(mitis_recon.copy())

    # for label, arr in recon.items():
    #     recon[label] = np.array(arr)
    
    # for label in _LABELS:
        # X = origin[label]
    ny, nx = origin[_LABELS[0]].shape

    # extract small sample of signal
    k = round(nx * ny * sampling_frac)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
    # b = np.expand_dims(b, axis=1)

    # create dct matrix operator using kron (memory errors for large ny*nx)
    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
        )
    A = A[ri,:] # same as phi times kron

    # b = X.T.flat[ri]
    for label in _LABELS:
        recon[label] = recon_2D_by_cvxpy(nx, ny, A, origin[label].T.flat[ri])
        # recon[label] = recon_by_Lasso()

    # ideals_recon = recon_2D_by_cvxpy(nx, ny, A, origin['ideals'].T.flat[ri])
    # unmitis_recon = recon_2D_by_cvxpy(nx, ny, A, origin['unmitis'].T.flat[ri])
    # mitis_recon = recon_2D_by_cvxpy(nx, ny, A, origin['mitis'].T.flat[ri])

    # recon['ideals'] = ideals_recon
    # recon['unmitis'] = unmitis_recon
    # recon['mitis'] = mitis_recon

    if figdir:
        _vis_one_D_p1_recon(
            origin_dict=origin,
            recon_dict=recon,
            full_range=full_range,
            bounds=None,
            true_optima=None,
            title='test',
            save_path=f'{figdir}/origin_and_2D_recon_sf{sampling_frac:.3f}.png'
        )

    print('end: solve l1 norm')
    return recon


def vis_optimization_on_p1_landscape(
    figdir,
    params_path
):
    fig, axs = plt.subplots()
    ax = axs
    # fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    # ax = plt.axes()
    # ax.plot(beta_opt, gamma_opt, "ro")
    # ax.plot(*_BETA_GAMMA_OPT[_VAR_INDICE], marker="o", color='red', markersize=5, label='optimal point')
    if params_path != None and len(params_path) > 0:
        xs = []
        ys = []
        for params in params_path:
            tmp_params = np.hstack([params["beta"], params["gamma"]])
            xs.append(tmp_params[_VAR_INDICE[0]])
            ys.append(tmp_params[_VAR_INDICE[1]])

        ax.plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
        ax.plot(xs[0], ys[0], marker="+", color='gray', markersize=7, label="initial point")
        ax.plot(xs[-1], ys[-1], marker="s", color='black', markersize=5, label="last point")
    
    ax.set_ylabel('beta')
    ax.set_xlabel('gamma')
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.legend()
    ax.set_title('QAOA energy')

    # ax.plot(beta_opt, gamma_opt, "ro")
    # fig.colorbar(c, ax=ax)
    fig.savefig(f"{figdir}/opt_path.png")
    plt.close(fig)
    return True


# ================= parameter path vis =================

def _vis_p1_params_path(
        origin_dict,
        recon_dict,
        bounds,
        # gamma_range,
        # beta_range,
        # C_opt, bound, var_opt,
        # full_range,
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
    axs = axs.reshape(-1)

    # beta:
    # X = (bounds['beta'][1] - bounds['beta'][0]) / 

    # gamma:
    # Y = 

    idx = 0
    for label, origin in origin_dict.items():
        recon = recon_dict[label]
        # axs[idx]
        # X, Y = np.meshgrid(var1_range, var2_range) # , indexing='ij')
        # Z = np.array(Z).T
        # c = axs[idx].pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
        
        im = axs[idx].imshow(origin)
        axs[idx].set_title(f"origin, {label}")

        im = axs[idx + 3].imshow(recon)
        axs[idx + 3].set_title(f"recon, {label}")

        if recon_params_path_dict and label in recon_params_path_dict:
            xs = [] # beta
            ys = [] # gamma
            for param in recon_params_path_dict[label]:
                xs.append(param[1])
                ys.append(param[0])

            axs[idx + 3].plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
            axs[idx + 3].plot(xs[0], ys[0], marker="+", color='gray', markersize=7, label="initial point")
            axs[idx + 3].plot(xs[-1], ys[-1], marker="s", color='black', markersize=5, label="last point")
        
        idx += 1

    fig.colorbar(im, ax=[axs[i] for i in range(6)])
    # plt.title(title)
    # plt.subtitle(title)
    fig.savefig(save_path)
    plt.close('all')