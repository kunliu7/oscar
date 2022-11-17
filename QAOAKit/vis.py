from cProfile import label
from re import A
from tkinter.tix import Tree
from typing import List
import networkx as nx
import numpy as np
import pandas as pd
from pyparsing import col
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
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

from .qaoa import get_maxcut_qaoa_circuit
from .utils import (
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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from random import sample


def vis_landscape(G: nx.Graph, figpath: str):
    # print("test")
    # figpath = 'test'
    def f(x, y):
        return noisy_qaoa_maxcut_energy(G, [x], [y])
    
    # qaoa format, i.e. beta not included
    betas = np.linspace(-np.pi, np.pi, 30)
    gammas = np.linspace(-np.pi/2, np.pi/2, 30)

    Z = []
    for b in betas:
        z = []
        for g in gammas:
            energy = f(b, g) 
            z.append(energy)
            # print(energy)
        Z.append(z.copy())
    X, Y = np.meshgrid(betas, gammas, indexing='ij')
    Z = np.array(Z)
    # print(Z.shape)
    # Z = f(X, Y)

    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig(figpath)
    # print('fig saved')
    return


def vis_landscape_heatmap(G: nx.Graph, figpath: str, beta_opt, gamma_opt):
    """
    heatmap
    https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    """
    def f(x, y):
        return noisy_qaoa_maxcut_energy(G, [x], [y])
    
    # qaoa format, i.e. beta not included
    betas = np.linspace(-np.pi, np.pi, 30)
    gammas = np.linspace(-np.pi/2, np.pi/2, 30)

    Z = []
    for b in betas:
        z = []
        for g in gammas:
            energy = f(b, g) 
            z.append(energy)
            # print(energy)
        Z.append(z.copy())
    X, Y = np.meshgrid(betas, gammas, indexing='ij')
    Z = np.array(Z)

    z_min, z_max = Z.min(), Z.max()

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    ax = plt.axes()
    plt.plot(beta_opt, gamma_opt, "ro")

    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=z_min, vmax=z_max)
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])

    # ax.plot(beta_opt, gamma_opt, "ro")
    fig.colorbar(c, ax=ax)
    fig.savefig(figpath)




def vis_landscape_heatmap_multi_p(
        G: nx.Graph,
        figpath: str,
        var1_idx: int, # $ when p=3, 0~5 maps to \beta_1, \beta_2, \beta_3, \gamma_1, \gamma_2, \gamma_3
        var2_idx: int,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list
    ):
    
    # 0 <= var1_idx < var2_idx < 2*p
    p = len(beta_opt)
    assert var1_idx >= 0 and var1_idx < 2*p - 1
    assert var2_idx > var1_idx and var2_idx < 2*p

    _VAR_INDICE = [var1_idx, var2_idx]
    """
    heatmap
    https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    """
    # def f(x, y):
    #     return noisy_qaoa_maxcut_energy(G, [x], [y])
    
    # qaoa format, i.e. \pi not included
    _BETA_RANGE = np.linspace(-np.pi, np.pi, 30)
    # _BETA_RANGE = angles
    _GAMMA_RANGE = np.linspace(-np.pi/2, np.pi/2, 30)
    
    var1_range = _BETA_RANGE if var1_idx < p else _GAMMA_RANGE
    var2_range = _BETA_RANGE if var2_idx < p else _GAMMA_RANGE
    var1_range = var1_range.copy()
    var2_range = var2_range.copy()

    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])
    _X_Y_LABELS = [f'gamma{i}' for i in range(p)]
    _X_Y_LABELS.extend([f'beta{i}' for i in range(p)])

    Z = []
    for v1 in var1_range:
        z = []
        for v2 in var2_range:
            tmp_vars = _BETA_GAMMA_OPT.copy()
            tmp_vars[_VAR_INDICE] = np.array([v1, v2])
            energy = noisy_qaoa_maxcut_energy(G, tmp_vars[:p], tmp_vars[p:], noise_model=noise_model)
            z.append(energy)
        Z.append(z.copy())
    X, Y = np.meshgrid(var1_range, var2_range) # , indexing='ij')
    Z = np.array(Z).T


    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    # ax = plt.axes()
    # ax.plot(beta_opt, gamma_opt, "ro")
    ax.plot(*_BETA_GAMMA_OPT[_VAR_INDICE], marker="o", color='red', markersize=5, label='optimal point')
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
    
    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    ax.set_ylabel(_X_Y_LABELS[var1_idx])
    ax.set_xlabel(_X_Y_LABELS[var2_idx])
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.legend()
    ax.set_title('QAOA energy')

    # ax.plot(beta_opt, gamma_opt, "ro")
    fig.colorbar(c, ax=ax)
    fig.savefig(figpath)
    plt.close(fig)
    return True
    # plt.clf()


def vis_landscape_multi_p(
        G: nx.Graph, figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list
    ):

    p = len(beta_opt)

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for var1_idx in range(2*p - 1):
        for var2_idx in range(var1_idx+1, 2*p):
            figpath = f'{figdir}/var_indice={var1_idx},{var2_idx}.png'
            vis_landscape_heatmap_multi_p(
                G,
                figpath,
                var1_idx,
                var2_idx,
                beta_opt,
                gamma_opt,
                noise_model,
                params_path
            )
            print(f"fig saved at {figpath}")

    return


def vis_landscape_heatmap_multi_p_and_count_optima(
        G: nx.Graph,
        p: int,
        figdir: str,
        var1_idx: int, # $ when p=3, 0~5 maps to \beta_1, \beta_2, \beta_3, \gamma_1, \gamma_2, \gamma_3
        var2_idx: int,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):
    
    # 0 <= var1_idx < var2_idx < 2*p
    assert var1_idx >= 0 and var1_idx < 2*p - 1
    assert var2_idx > var1_idx and var2_idx < 2*p

    _VAR_INDICE = [var1_idx, var2_idx]
    """
    heatmap
    https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    """
    # def f(x, y):
    #     return noisy_qaoa_maxcut_energy(G, [x], [y])
    
    # qaoa format, i.e. \pi not included
    _BETA_RANGE = np.linspace(-np.pi, np.pi, 30)
    # _BETA_RANGE = angles
    _GAMMA_RANGE = np.linspace(-np.pi/2, np.pi/2, 30)
    
    var1_range = _BETA_RANGE if var1_idx < p else _GAMMA_RANGE
    var2_range = _BETA_RANGE if var2_idx < p else _GAMMA_RANGE
    var1_range = var1_range.copy()
    var2_range = var2_range.copy()

    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])
    _X_Y_LABELS = [f'gamma{i}' for i in range(p)]
    _X_Y_LABELS.extend([f'beta{i}' for i in range(p)])

    cnt_opt = 0
    Z = []
    for v1 in var1_range:
        z = []
        for v2 in var2_range:
            tmp_vars = _BETA_GAMMA_OPT.copy()
            tmp_vars[_VAR_INDICE] = np.array([v1, v2])
            energy = noisy_qaoa_maxcut_energy(G, tmp_vars[:p], tmp_vars[p:], noise_model=noise_model)
            z.append(energy)

            # if p >= 3:
                # p>=3, fixed angles, they are approximately good optima
                # if energy > C_opt - 0.02: # and np.isclose(energy, C_opt, 0.03):
                #     cnt_opt += 1
                # if np.isclose(energy, C_opt, 0.03):
                #     cnt_opt += 1
            # else:
                # p=1,2
            # if energy > C_opt * (1 - 0.03):
            if np.isclose(energy, C_opt, 0.03):
                cnt_opt += 1
            
        Z.append(z.copy())
    X, Y = np.meshgrid(var1_range, var2_range) # , indexing='ij')
    Z = np.array(Z).T

    # we do not want zero, it must be due to the bad threshold of counting
    if cnt_opt == 0:
        cnt_opt = 1

    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    # ax = plt.axes()
    # ax.plot(beta_opt, gamma_opt, "ro")
    
    # print optimal points
    ax.plot(*_BETA_GAMMA_OPT[_VAR_INDICE], marker="o", color='red', markersize=5, label='optimal point')
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
    
    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    ax.set_ylabel(_X_Y_LABELS[var1_idx])
    ax.set_xlabel(_X_Y_LABELS[var2_idx])
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    # ax.legend()
    ax.set_title('QAOA energy')

    # ax.plot(beta_opt, gamma_opt, "ro")
    fig.colorbar(c, ax=ax)
    fig.savefig(f'{figdir}/varIndices={var1_idx},{var2_idx}_nOpt{cnt_opt}.png')
    plt.close(fig)
    return cnt_opt
    # plt.clf()


def vis_one_landscape_and_count_optima_and_mitiq(
        G: nx.Graph,
        p: int,
        figdir: str,
        var1_idx: int, # $ when p=3, 0~5 maps to \beta_1, \beta_2, \beta_3, \gamma_1, \gamma_2, \gamma_3
        var2_idx: int,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):


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
        
    # 0 <= var1_idx < var2_idx < 2*p
    assert var1_idx >= 0 and var1_idx < 2*p - 1
    assert var2_idx > var1_idx and var2_idx < 2*p

    _VAR_INDICE = [var1_idx, var2_idx]
    """
    heatmap
    https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    """
    # def f(x, y):
    #     return noisy_qaoa_maxcut_energy(G, [x], [y])
    
    # qaoa format, i.e. \pi not included
    _BETA_RANGE = np.linspace(-np.pi, np.pi, 30)
    # _BETA_RANGE = angles
    _GAMMA_RANGE = np.linspace(-np.pi/2, np.pi/2, 30)
    
    var1_range = _BETA_RANGE if var1_idx < p else _GAMMA_RANGE
    var2_range = _BETA_RANGE if var2_idx < p else _GAMMA_RANGE
    var1_range = var1_range.copy()
    var2_range = var2_range.copy()

    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])
    _X_Y_LABELS = [f'gamma{i}' for i in range(p)]
    _X_Y_LABELS.extend([f'beta{i}' for i in range(p)])

    miti_cnt_opt = 0
    unmiti_cnt_opt = 0

    miti_Z = []
    unmiti_Z = []
    for v1 in var1_range:
        miti_z = []
        unmiti_z = []
        for v2 in var2_range:
            tmp_vars = _BETA_GAMMA_OPT.copy()
            tmp_vars[_VAR_INDICE] = np.array([v1, v2])

            circuit = get_maxcut_qaoa_circuit(
                G, beta=tmp_vars[:p], gamma=tmp_vars[p:],
                transpile_to_basis=True, save_state=False)
            # copy = circuit.copy()
            # circuit.measure_all()
            # print(circuit.qasm())
            # qiskit.Q
            # circuit = circuit.decompose()
            unmiti = mitiq_executor_of_qaoa_maxcut_energy(circuit.copy())
            miti = zne.execute_with_zne(
                circuit=circuit.copy(),
                executor=mitiq_executor_of_qaoa_maxcut_energy,
            )

            # print('miti: ', miti)

            # energy = unmiti # - miti
            # print(energy)
            # energy = noisy_qaoa_maxcut_energy(G, tmp_vars[:p], tmp_vars[p:], noise_model=noise_model)
            # z.append(energy)
            miti_z.append(miti)
            unmiti_z.append(unmiti)
            

            # if p >= 3:
                # p>=3, fixed angles, they are approximately good optima
                # if energy > C_opt - 0.02: # and np.isclose(energy, C_opt, 0.03):
                #     cnt_opt += 1
                # if np.isclose(energy, C_opt, 0.03):
                #     cnt_opt += 1
            # else:
                # p=1,2
            # if energy > C_opt * (1 - 0.03):
            if np.isclose(miti, C_opt, 0.03):
                miti_cnt_opt += 1

            if np.isclose(unmiti, C_opt, 0.03):
                unmiti_cnt_opt += 1
            
        miti_Z.append(miti_z.copy())
        unmiti_Z.append(unmiti_z.copy())

    X, Y = np.meshgrid(var1_range, var2_range) # , indexing='ij')
    
    # we do not want zero, it must be due to the bad threshold of counting
    if miti_cnt_opt == 0:
        miti_cnt_opt = 1
    
    if unmiti_cnt_opt == 0:
        unmiti_cnt_opt = 1

    # =============== miti ===============
    Z = np.array(miti_Z).T
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    # ax = plt.axes()
    # ax.plot(beta_opt, gamma_opt, "ro")
    
    # print optimal points
    ax.plot(*_BETA_GAMMA_OPT[_VAR_INDICE], marker="o", color='red', markersize=5, label='optimal point')
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
    
    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    ax.set_ylabel(_X_Y_LABELS[var1_idx])
    ax.set_xlabel(_X_Y_LABELS[var2_idx])
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    # ax.legend()
    ax.set_title('QAOA energy')

    # ax.plot(beta_opt, gamma_opt, "ro")
    fig.colorbar(c, ax=ax)
    fig.savefig(f'{figdir}/miti_varIndices={var1_idx},{var2_idx}_nOpt{miti_cnt_opt}.png')
    plt.close(fig)


    # ============= unmiti =============
    Z = np.array(unmiti_Z).T
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    # ax = plt.axes()
    # ax.plot(beta_opt, gamma_opt, "ro")
    
    # print optimal points
    ax.plot(*_BETA_GAMMA_OPT[_VAR_INDICE], marker="o", color='red', markersize=5, label='optimal point')
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
    
    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    ax.set_ylabel(_X_Y_LABELS[var1_idx])
    ax.set_xlabel(_X_Y_LABELS[var2_idx])
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    # ax.legend()
    ax.set_title('QAOA energy')

    # ax.plot(beta_opt, gamma_opt, "ro")
    fig.colorbar(c, ax=ax)
    fig.savefig(f'{figdir}/unmiti_varIndices={var1_idx},{var2_idx}_nOpt{unmiti_cnt_opt}.png')
    plt.close(fig)
    return miti_cnt_opt, unmiti_cnt_opt


def vis_one_landscape_and_count_optima_and_mitiq_and_one_variable(
        G: nx.Graph,
        p: int,
        figdir: str,
        var_idx: int, # p==2, 0-3 -> beta0, beta1, gamma0, gamma1
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):


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

    _BETA_N_SAMPLES = 20
    _GAMMA_N_SAMPLES = 80
    
    # qaoa format, i.e. \pi not included
    # _BETA_RANGE = np.linspace(-np.pi, np.pi, _N_SAMPLES)
    _BETA_RANGE = np.linspace(-np.pi/4, np.pi/4, _BETA_N_SAMPLES)
    # _BETA_RANGE = angles
    # _GAMMA_RANGE = np.linspace(-np.pi/2, np.pi/2, _N_SAMPLES)
    _GAMMA_RANGE = np.linspace(-np.pi, np.pi, _GAMMA_N_SAMPLES)
    
    var_range = _BETA_RANGE if var_idx < p else _GAMMA_RANGE
    var_range = var_range.copy()

    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])
    _X_Y_LABELS = [f'beta{i}' for i in range(p)]
    _X_Y_LABELS.extend([f'gamma{i}' for i in range(p)])

    miti_cnt_opt = 0
    unmiti_cnt_opt = 0

    miti_z = []
    unmiti_z = []

    ymax = float('-inf')
    ymin = float('inf')

    threshold = 0.03

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

        miti_z.append(miti)
        unmiti_z.append(unmiti)

        ymax = max(ymax, miti, unmiti)
        ymin = min(ymin, miti, unmiti)
            
        if np.isclose(miti, C_opt, threshold):
            miti_cnt_opt += 1

        if np.isclose(unmiti, C_opt, threshold):
            unmiti_cnt_opt += 1

    np.savez_compressed(f"{figdir}/data_threshold{threshold}_varIdx{var_idx}",
        mitis=miti_z, unmitis=unmiti_z,
        miti_cnt_opt=miti_cnt_opt, unmiti_cnt_opt=unmiti_cnt_opt)
    
    # =============== vis ===============
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=[10, 10])
    # plt.plot()
    # ax = plt.axes()
    # ax.plot(beta_opt, gamma_opt, "ro")
    # print optimal points
    # ax.vlines(x=_BETA_GAMMA_OPT[var_idx], ymin=ymin, ymax=ymax, colors=('r'), label="optimal point")
    ax.plot(_BETA_GAMMA_OPT[var_idx], C_opt, marker="o", color='red', markersize=5, label='optimal point')
    if params_path != None and len(params_path) > 0:
        pass
        # xs = []
        # ys = []
        # for params in params_path:
        #     tmp_params = np.hstack([params["beta"], params["gamma"]])
        #     xs.append(tmp_params[var_idx])
        #     ys.append(tmp_params[var_idx])

        # ax.plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
        # ax.plot(xs[0], ys[0], marker="+", color='gray', markersize=7, label="initial point")
        # ax.plot(xs[-1], ys[-1], marker="s", color='black', markersize=5, label="last point")
    
    ax.plot(var_range, unmiti_z, marker="o", label="unmitigated")
    ax.plot(var_range, miti_z, marker="o", label="mitigated")
    
    ax.set_ylabel('# optima')
    ax.set_xlabel(_X_Y_LABELS[var_idx])
    ax.set_title(f'QAOA energy, nQ{G.number_of_nodes()}, before and after mitigation')
    ax.legend()

    fig.savefig(f'{figdir}/zne_varIdx={var_idx}_nOpt_unmiti{unmiti_cnt_opt}_miti{miti_cnt_opt}.png')
    plt.close(fig)

    return miti_cnt_opt, unmiti_cnt_opt

# ================================= top function ===========================

def vis_landscape_multi_p_and_and_count_optima(
        G: nx.Graph, 
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    n_optima_list = []
    for var1_idx in range(2*p - 1):
        for var2_idx in range(var1_idx+1, 2*p):
            # figpath = f'{figdir}/var_indice={var1_idx},{var2_idx}.png'
            n_optima = vis_landscape_heatmap_multi_p_and_count_optima(
            # vis_landscape_heatmap_multi_p(
                G,
                p,
                figdir,
                var1_idx,
                var2_idx,
                beta_opt,
                gamma_opt,
                noise_model,
                params_path,
                C_opt
            )
            n_optima_list.append(n_optima)
            # print(f"fig saved at {figpath}")

    return n_optima_list


def vis_landscape_multi_p_and_and_count_optima_MP(
        G: nx.Graph, 
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    n_optima_list = []

    params = []
    for var1_idx in range(2*p - 1):
        for var2_idx in range(var1_idx+1, 2*p):
            # figpath = f'{figdir}/var_indice={var1_idx},{var2_idx}.png'
            # n_optima = vis_landscape_heatmap_multi_p_and_count_optima(
            # vis_landscape_heatmap_multi_p(
            params.append((
                G.copy(),
                p,
                figdir,
                var1_idx,
                var2_idx,
                beta_opt.copy(),
                gamma_opt.copy(),
                noise_model.copy() if noise_model else None,
                params_path.copy(),
                C_opt
            ))
            # n_optima_list.append(n_optima)
            # print(f"fig saved at {figpath}")

    print('start MP')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # n_optima_list = executor.map(
        #     lambda x: vis_landscape_heatmap_multi_p_and_count_optima(*x), params, chunksize=16)
        futures = [executor.submit(vis_landscape_heatmap_multi_p_and_count_optima, *param) for param in params]
        concurrent.futures.wait(futures)
        # for future in concurrent.futures.as_completed(futures):
        #     print(future.result())

    # print(futures)
    n_optima_list = [f.result() for f in futures]
    # print(n_optima_list)

    return n_optima_list



def vis_multi_landscape_and_count_optima_and_mitiq_MP(
        G: nx.Graph, 
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float
    ):

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    params = []
    for var1_idx in range(2*p - 1):
        for var2_idx in range(var1_idx+1, 2*p):
            # figpath = f'{figdir}/var_indice={var1_idx},{var2_idx}.png'
            # n_optima = vis_landscape_heatmap_multi_p_and_count_optima(
            # vis_landscape_heatmap_multi_p(
            params.append((
                G.copy(),
                p,
                figdir,
                var1_idx,
                var2_idx,
                beta_opt.copy(),
                gamma_opt.copy(),
                noise_model.copy() if noise_model else None,
                params_path.copy(),
                C_opt
            ))
            # n_optima_list.append(n_optima)
            # print(f"fig saved at {figpath}")
    
    params = sample(params, min(10, len(params)))

    print('choose 10 randomly:', len(params))

    print('start MP')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # n_optima_list = executor.map(
        #     lambda x: vis_landscape_heatmap_multi_p_and_count_optima(*x), params, chunksize=16)
        futures = [executor.submit(vis_one_landscape_and_count_optima_and_mitiq, *param) for param in params]
        concurrent.futures.wait(futures)
        # for future in concurrent.futures.as_completed(futures):
        #     print(future.result())

    # print(futures)
    miti_n_opt_list = []
    unmiti_n_opt_list = []
    for f in futures:
        miti_n_opt_list.append(f.result()[0])
        unmiti_n_opt_list.append(f.result()[1])
    # print(n_optima_list)

    return miti_n_opt_list, unmiti_n_opt_list


def vis_multi_landscapes_and_count_optima_and_mitiq_MP_and_one_variable(
        G: nx.Graph, 
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        executor: concurrent.futures.ProcessPoolExecutor
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
            C_opt
        ))
    
    print('choose 10 randomly:', len(params))

    print('start MP')
    # with concurrent.futures.ProcessPoolExecutor() as executor:
        # n_optima_list = executor.map(
        #     lambda x: vis_landscape_heatmap_multi_p_and_count_optima(*x), params, chunksize=16)
    for param in params:
        executor.submit(
            vis_one_landscape_and_count_optima_and_mitiq_and_one_variable,
            *param
        )
        # print(future.result())
        
    return [], []


# ================================ 

def vis_two_BPs_p1_recon(
        origin_dict,
        recon_dict,
        # gamma_range,
        # beta_range,
        # C_opt, bound, var_opt,
        bounds,
        full_range,
        # mitis_recon,
        # unmitis_recon,
        # ideal_recon,
        # xlabel, 
        box1,
        box2,
        box1_points,
        box2_points,
        title,
        save_path
    ):

    # plt.figure
    plt.rc('font', size=28)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 30))
    fig.suptitle(title, y=0.92)
    axs = axs.reshape(-1)

    rect1 = patches.Rectangle(
        (box1['beta'][0], box1['gamma'][0]),
         box1['beta'][1] - box1['beta'][0],
         box1['gamma'][1] - box1['gamma'][0],
        linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle(
        (box2['beta'][0], box2['gamma'][0]),
         box2['beta'][1] - box2['beta'][0],
         box2['gamma'][1] - box2['gamma'][0],
        linewidth=1, edgecolor='b', facecolor='none')
    # ,linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    # ax.add_patch(rect1)

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
        axs[idx].set_xlabel('beta')
        axs[idx].set_ylabel('gamma')
        # up down left right
        # axs[idx].Rectange((box1[0], box1[2]), box1[3] - box1[2], box1[0] - box1[1])
        axs[idx].add_patch(copy.deepcopy(rect1))
        axs[idx].add_patch(copy.deepcopy(rect2))

        if box1_points != None:
            xs = []
            ys = []
            for p in box1_points:
                xs.append(p['beta'])
                ys.append(p['gamma'])
            axs[idx].scatter(xs, ys, s=2)

        if box2_points != None:
            xs = []
            ys = []
            for p in box2_points:
                xs.append(p['beta'])
                ys.append(p['gamma'])
            axs[idx].scatter(xs, ys)
                
        # axs[idx].set_xlim(bottom=full_range['beta'][0], top=full_range['beta'][-1])
        # axs[idx].set_xlim(left=bounds['beta'][0], right=bounds['beta'][1])
        # axs[idx].set_ylim(bottom=bounds['gamma'][0], top=bounds['gamma'][1])

        # im = axs[idx + 3].imshow(recon)
        shift = 3
        im = axs[idx + shift].pcolormesh(X, Y, recon)
        axs[idx + shift].set_title(f"recon, {label}")
        axs[idx + shift].set_xlabel('beta')
        axs[idx + shift].set_ylabel('gamma')
        axs[idx + shift].add_patch(copy.deepcopy(rect1))
        axs[idx + shift].add_patch(copy.deepcopy(rect2))
        # axs[idx + 3].set_xlim(left=bounds['beta'][0], right=bounds['beta'][1])
        # axs[idx + 3].set_ylim(bottom=bounds['gamma'][0], top=bounds['gamma'][1])

        idx += 1

    plt.legend()
    fig.colorbar(im, ax=[axs[i] for i in range(6)])
    # plt.title(title)
    # plt.subtitle(title)
    fig.savefig(save_path)
    plt.close('all')


def _vis_recon_distributed_landscape(
        landscapes,
        labels,
        # origin_dict,
        # recon_dict,
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
    if len(landscapes) == 2:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 22))
        # fig.suptitle(title, y=0.8)
    elif len(landscapes) == 3:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 22))
        # fig.suptitle(title, y=0.92)
    elif len(landscapes) == 4:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
        fig.suptitle(title, y=0.92)
    else:
        assert False
    fig.suptitle(title, y=0.92)
    axs = axs.reshape(-1)

    # TODO Check ij and xy
    X, Y = np.meshgrid(full_range['beta'], full_range['gamma'], indexing='ij')

    # c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    for idx, landscape in enumerate(landscapes):
        im = axs[idx].pcolormesh(X, Y, landscape) #, cmap='viridis', vmin=origin.min(), vmax=origin.max())
        axs[idx].set_title(labels[idx])
        axs[idx].set_xlabel('beta')
        axs[idx].set_ylabel('gamma')

    # plt.legend()
    fig.colorbar(im, ax=[axs[i] for i in range(len(landscapes))])
    # plt.title(title)
    # plt.subtitle(title)
    fig.savefig(save_path, bbox_inches='tight')
    print('figure saved to ', save_path)
    plt.close('all')


def vis_landscapes(
        landscapes, # list of np.ndarray
        labels, # list of labels of correlated landscapes
        full_range, # dict, 
        true_optima,
        title,
        save_path, # figure save path
        params_paths, # list of list of parameters correlated to landscapes
        recon_params_path_dict=None,
        origin_params_path_dict=None
    ):

    assert len(landscapes) == len(labels)
    assert len(landscapes) == len(params_paths)

    # print("full_range =", full_range)

    tmp = []
    for ls in landscapes:
        if len(ls.shape) == 4:
            shape = ls.shape
            ls = ls.reshape(shape[0] * shape[1], shape[2] * shape[3])
            print(f"reshape: {shape} -> {ls.shape}")
        elif len(ls.shape) == 2:
            pass
        else:
            raise ValueError()
        tmp.append(ls)

    landscapes = tmp

    # plt.figure
    plt.rc('font', size=28)
    if len(landscapes) == 2:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 22))
    elif len(landscapes) == 3:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    elif len(landscapes) == 4:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
    else:
        assert False

    fig.suptitle(title)
    axs = axs.reshape(-1)

    # TODO Check ij and xy
    X, Y = np.meshgrid(full_range['beta'], full_range['gamma'], indexing='ij')
    # X, Y = np.meshgrid(full_range['beta'], full_range['gamma'], indexing='xy')
    
    # c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    for idx, landscape in enumerate(landscapes):
        im = axs[idx].pcolormesh(X, Y, landscape) #, cmap='viridis', vmin=origin.min(), vmax=origin.max())
        axs[idx].set_title(labels[idx])
        axs[idx].set_xlabel('beta')
        axs[idx].set_ylabel('gamma')
        if isinstance(true_optima, list) or isinstance(true_optima, np.ndarray):
            axs[idx].plot(true_optima[0], true_optima[1], marker="o", color='red', markersize=7, label="true optima")

        params = params_paths[idx]
        if isinstance(params, list) or isinstance(params, np.ndarray):
            xs = [] # beta
            ys = [] # gamma
            for param in params:
                xs.append(param[1])
                ys.append(param[0])

            axs[idx].plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
            axs[idx].plot(xs[0], ys[0], marker="o", color='white', markersize=9, label="initial point")
            axs[idx].plot(xs[-1], ys[-1], marker="s", color='white', markersize=12, label="last point")


    fig.colorbar(im, ax=[axs[i] for i in range(len(landscapes))])
    plt.legend()
    save_dir = os.path.dirname(save_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if os.path.exists(save_path):
    #     print("same path file exists, refuse to overwrite, please check")
    #     return

    fig.savefig(save_path, bbox_inches='tight')
    # plt.show()
    plt.close('all')

    print("save to: ", save_path)