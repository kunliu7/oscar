from cProfile import label
from re import A
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
import pytest
from itertools import groupby
import timeit
import sys, os
from scipy.optimize import minimize
from qiskit.providers.aer.noise import NoiseModel

from qiskit.quantum_info import Statevector

from .qaoa import get_maxcut_qaoa_circuit
from .utils import noisy_qaoa_maxcut_energy, angles_from_qiskit_format

# vis
import numpy as np
import matplotlib.pyplot as plt



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

            if p >= 3:
                # p>=3, fixed angles, they are approximately good optima
                if energy > C_opt + 0.02:
                    cnt_opt += 1
            else:
                # p=1,2
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