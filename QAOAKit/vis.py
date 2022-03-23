from typing import List
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
    _GAMMA_RANGE = np.linspace(-np.pi/2, np.pi/2, 30)
    
    var1_range = _BETA_RANGE if var1_idx < p else _GAMMA_RANGE
    var2_range = _BETA_RANGE if var2_idx < p else _GAMMA_RANGE
    var1_range = var1_range.copy()
    var2_range = var2_range.copy()

    _BETA_GAMMA_OPT = np.hstack([beta_opt, gamma_opt])

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
    ax.plot(*_BETA_GAMMA_OPT[_VAR_INDICE], "ro")
    if params_path != None and len(params_path) > 0:
        for params in params_path:
            params = angles_from_qiskit_format(params)
            params = np.hstack([params["beta"], params["gamma"]])
            # direction is opposite for qiskit params
            # details see Noise_Induced_QAOAKit/QAOAKit/utils.py, angles_to_qiskit_format(angles)
            # tmp_p = np.hstack([params[p:2*p], params[:p]])
            ax.plot(*params[_VAR_INDICE], "s-", color='purple')
    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    ax.set_ylabel("gamma")
    ax.set_xlabel("beta")
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])

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

