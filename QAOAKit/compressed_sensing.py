import time
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
import timeit
import sys, os
from scipy.fftpack import dct, diff, idct
from scipy.optimize import minimize
from sklearn import linear_model

from qiskit.providers.aer.noise import NoiseModel
import concurrent.futures
from mitiq import zne, Observable, PauliString
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
    get_curr_formatted_timestamp,
    noisy_qaoa_maxcut_energy,
    angles_from_qiskit_format,
    maxcut_obj,
    get_adjacency_matrix,
    obj_from_statevector
)
from .noisy_params_optim import (
    compute_expectation,
    get_depolarizing_error_noise_model
)

# vis
import numpy as np
import matplotlib.pyplot as plt
from random import sample


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


# ================== one dimensional CS p==1 ==================

def _vis_one_D_p1_recon(
        origin_dict,
        recon_dict,
        # gamma_range,
        # beta_range,
        # C_opt, bound, var_opt,
        # full_range,
        # mitis_recon,
        # unmitis_recon,
        # ideal_recon,
        # xlabel, 
        title,
        save_path
    ):

    # plt.figure
    plt.rc('font', size=28)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 30))
    axs = axs.reshape(-1)

    idx = 0
    for label, origin in origin_dict.items():
        recon = recon_dict[label]
        # axs[idx]
        # X, Y = np.meshgrid(var1_range, var2_range) # , indexing='ij')
        # Z = np.array(Z).T
        # c = axs[idx].pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
        
        im = axs[idx].imshow(origin)
        axs[idx].set_title(f"origin, {label}", )

        im = axs[idx + 3].imshow(recon)
        axs[idx + 3].set_title(f"recon, {label}")
        idx += 1

    fig.colorbar(im, ax=[axs[i] for i in range(6)])
    # plt.title(title)
    # plt.subtitle(title)
    fig.savefig(save_path)
    plt.close('all')


def _one_D_CS_p1_recon_for_one_point(
    G,
    beta,
    gamma,
    shots
):
    circuit = get_maxcut_qaoa_circuit(
        G, beta=[beta], gamma=[gamma],
        transpile_to_basis=True, save_state=False)

    ideal = _mitiq_executor_of_qaoa_maxcut_energy(circuit.copy(), G, is_noisy=False, shots=shots)
    unmiti = _mitiq_executor_of_qaoa_maxcut_energy(circuit.copy(), G, is_noisy=True, shots=shots)
    miti = zne.execute_with_zne(
        circuit=circuit.copy(),
        executor=partial(_mitiq_executor_of_qaoa_maxcut_energy, G=G, is_noisy=True, shots=shots),
    )

    return ideal, unmiti, miti


def _mitiq_executor_of_qaoa_maxcut_energy(qc, G, is_noisy, shots) -> float:
    backend = AerSimulator()
    qc.measure_all()
    # backend = Aer.get_backend('aer_simulator')
    noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model
    # noise_model = None
    counts = backend.run(
        qc,
        shots=shots,
        noise_model=noise_model if is_noisy else None
    ).result().get_counts()
    
    expval = compute_expectation(counts, G)
    return -expval


def _one_D_CS_p1_recon_for_one_point_mapper(param):
    return _one_D_CS_p1_recon_for_one_point(*param)


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

        ideals_recon = recon_by_Lasso(Psi[perm, :], origin['ideals'][idx_gamma, perm], alpha)
        unmitis_recon = recon_by_Lasso(Psi[perm, :], origin['unmitis'][idx_gamma, perm], alpha)
        mitis_recon = recon_by_Lasso(Psi[perm, :], origin['mitis'][idx_gamma, perm], alpha)

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


def one_D_CS_p1_generate_landscape_task(
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
    n_shots = 1
    n_pts_per_unit = 8     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
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
    # sample P points from N randomly

    # mitis = []
    # unmitis = []
    # ideals = []

    _LABELS = ['mitis', 'unmitis', 'ideals']
    origin = {label: [] for label in _LABELS}

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    
    params = []
    for gamma in full_range['gamma']:
        # mitis = []
        # unmitis = []
        # ideals = []

        for beta in full_range['beta']:
            param = (
                G.copy(),
                beta,
                gamma,
                n_shots
            )
            params.append(param)
            # circuit = get_maxcut_qaoa_circuit(
            #     G, beta=[beta], gamma=[gamma],
            #     transpile_to_basis=True, save_state=False)
            
            # ideal_f = executor.submit(
            #     _mitiq_executor_of_qaoa_maxcut_energy,
            #     circuit.copy(),
            #     G,
            #     False,
            #     n_shots
            # )

            # unmiti_f = executor.submit(
            #     _mitiq_executor_of_qaoa_maxcut_energy,
            #     circuit.copy(),
            #     G,
            #     True,
            #     n_shots
            # )

            # miti_f = executor.submit(
            #     zne.execute_with_zne,
            #     circuit.copy(),
            #     partial(_mitiq_executor_of_qaoa_maxcut_energy, is_noisy=True, G=G, shots=n_shots),
            # )
            
            # mitis.append(miti_f.result())
            # unmitis.append(unmiti_f.result())
            # ideals.append(ideal_f.result())

            # ideal = mitiq_executor_of_qaoa_maxcut_energy(circuit.copy(), is_noisy=False)
            # unmiti = mitiq_executor_of_qaoa_maxcut_energy(circuit.copy(), is_noisy=True)
            # miti = zne.execute_with_zne(
            #     circuit=circuit.copy(),
            #     executor=partial(mitiq_executor_of_qaoa_maxcut_energy, is_noisy=True),
            # )
            # mitis.append(miti)
            # unmitis.append(unmiti)
            # ideals.append(ideal)

        # origin['mitis'].append(mitis.copy())
        # origin['unmitis'].append(unmitis.copy())
        # origin['ideals'].append(ideals.copy())

    print(len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # future = executor.submit(
        #     _one_D_CS_p1_recon_for_one_point,
        #     *params[0]
        # )
        futures = executor.map(
            _one_D_CS_p1_recon_for_one_point_mapper, params
        )
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {end_time - start_time} s")

    for f in futures:
        origin['ideals'].append(f[0])
        origin['unmitis'].append(f[1])
        origin['mitis'].append(f[2])

    for label, arr in origin.items():
        origin[label] = np.array(arr).reshape(n_pts['gamma'], n_pts['beta'])
        print(origin[label].shape)
        
    # x = np.cos(2 * 97 * np.pi * full_range) + np.cos(2 * 777 * np.pi * full_range)
    # x = np.cos(2 * np.pi * full_range) # + np.cos(2 * np.pi * full_range)
    
    # print('start: solve l1 norm')
    # recon = {label: [] for label in _LABELS}
    # for idx_gamma, _ in enumerate(full_range['gamma']):
    #     Psi = dct(np.identity(n_pts['beta']))
    #     perm = np.floor(np.random.rand(n_samples['beta']) * n_pts['beta']).astype(int)

    #     ideals_recon = recon_by_Lasso(Psi[perm, :], origin['ideals'][idx_gamma, perm], alpha)
    #     unmitis_recon = recon_by_Lasso(Psi[perm, :], origin['unmitis'][idx_gamma, perm], alpha)
    #     mitis_recon = recon_by_Lasso(Psi[perm, :], origin['mitis'][idx_gamma, perm], alpha)

    #     recon['ideals'].append(ideals_recon.copy())
    #     recon['unmitis'].append(unmitis_recon.copy())
    #     recon['mitis'].append(mitis_recon.copy())

    # _vis_one_D_p1_recon(
    #     origin_dict=origin,
    #     recon_dict=recon,
    #     title='test',
    #     save_path=f'{figdir}/origin_recon.png'
    # )

    np.savez_compressed(f"{figdir}/data",

        # reconstruct
        origin=origin,
        # recon=recon,
        # mitis=mitis, unmitis=unmitis, ideals=ideals,
        # unmitis_recon=unmitis_recon, mitis_recon=mitis_recon, ideals_recon=ideals_recon,

        # parameters
        n_pts=n_pts,
        # n_samples=n_samples, sampling_frac=sampling_frac,
        # perm=perm,
        full_range=full_range,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        # n_optima
        # improved_n_optima=improved_n_optima,
        # improved_n_optima_recon=improved_n_optima_recon,
        C_opt=C_opt)

    print('end: solve l1 norm')
    
    # ----- count optima

    # max_unmiti = unmitis.max()
    # improved_n_optima = np.sum(
    #     # np.isclose(unmitis_recon, C_opt, atol=C_opt-max_miti) == True
    #     mitis > max_unmiti
    # )
    
    # max_unmiti_recon = unmitis_recon.max()
    # improved_n_optima_recon = np.sum(
    #     # np.isclose(mitis_recon, C_opt, atol=abs(C_opt-max_unmiti_recon)) == True
    #     mitis_recon > max_unmiti_recon
    # )

    # print('improved_n_optima, recon', improved_n_optima, improved_n_optima_recon)

    # =============== vis unmiti ===============

    # _vis_y_and_x_recon(
    #     C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],

    #     recon_range=full_range, x_recon=mitis_recon,
    #     sample_range=full_range[perm], x_sample=mitis[perm],
    #     full_range=full_range, x_full=mitis,

    #     xlabel=_X_Y_LABELS[var_idx],
    #     title=f'QAOA energy, nQ{G.number_of_nodes()}, recon miti',
    #     save_path=f'{figdir}/zne_varIdx={var_idx}_mitis_recon.png'
    # )
    
    # _vis_y_and_x_recon(
    #     C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
        
    #     recon_range=full_range, x_recon=unmitis_recon,
    #     sample_range=full_range[perm], x_sample=unmitis[perm],
    #     full_range=full_range, x_full=unmitis,

    #     xlabel=_X_Y_LABELS[var_idx],
    #     title=f'QAOA energy, nQ{G.number_of_nodes()}, recon unmiti',
    #     save_path=f'{figdir}/zne_varIdx={var_idx}_unmitis_recon.png'
    # )
    
    # _vis_y_and_x_recon(
    #     C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],

    #     recon_range=full_range, x_recon=ideals_recon,
    #     sample_range=full_range[perm], x_sample=ideals[perm],
    #     full_range=full_range, x_full=ideals,

    #     xlabel=_X_Y_LABELS[var_idx],
    #     title=f'QAOA energy, nQ{G.number_of_nodes()}, recon ideal',
    #     save_path=f'{figdir}/zne_varIdx={var_idx}_ideal.png'
    # )

    # _vis_miti_and_unmiti_recon(
    #     C_opt=C_opt, bound=bound, var_opt=_BETA_GAMMA_OPT[var_idx],
    #     full_range=full_range,
    #     mitis_recon=mitis_recon,
    #     unmitis_recon=unmitis_recon,
    #     ideal_recon=ideals_recon,
    #     xlabel=_X_Y_LABELS[var_idx],
    #     title=f'QAOA energy, nQ{G.number_of_nodes()}, reconstructed unmiti and miti',
    #     save_path=f'{figdir}/zne_varIdx={var_idx}_unmitis_recon_and_mitis_recon.png'
    # )

    # ======== save ==========

    # np.savez_compressed(f"{figdir}/varIdx{var_idx}",

    #     # reconstruct
    #     mitis=mitis, unmitis=unmitis, ideals=ideals,
    #     unmitis_recon=unmitis_recon, mitis_recon=mitis_recon, ideals_recon=ideals_recon,

    #     # parameters
    #     n_pts=n_pts, n_samples=n_samples, sampling_frac=sampling_frac,
    #     perm=perm, full_range=full_range,

    #     # n_optima
    #     improved_n_optima=improved_n_optima,
    #     improved_n_optima_recon=improved_n_optima_recon,
    #     C_opt=C_opt)

    return


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

    one_D_CS_p1_generate_landscape_task(*params[0])
        
    return [], []


# ============================ two D CS ====================



def two_D_CS_p1_recon_with_given_landscapes(

):
    pass
