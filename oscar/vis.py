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
from qiskit_aer import AerSimulator
from functools import partial
from pathlib import Path
import copy
from itertools import groupby
import timeit
import sys, os
from scipy.optimize import minimize
from qiskit_aer.noise import NoiseModel
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
                xs.append(param[0])
                ys.append(param[1])

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


def vis_landscape_refactored(
    ax,
    landscape, # list of np.ndarray
    label, # list of labels of correlated landscapes
    full_range, # dict, 
    true_optima,
    params_path, # list of list of parameters correlated to landscapes
):

    if len(ls.shape) == 4:
        shape = ls.shape
        ls = ls.reshape(shape[0] * shape[1], shape[2] * shape[3])
        print(f"reshape: {shape} -> {ls.shape}")
    elif len(ls.shape) == 2:
        pass
    else:
        raise ValueError()

    # TODO Check ij and xy
    X, Y = np.meshgrid(full_range['beta'], full_range['gamma'], indexing='ij')
    # X, Y = np.meshgrid(full_range['beta'], full_range['gamma'], indexing='xy')
    
    # c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    im = ax.pcolormesh(X, Y, landscape) #, cmap='viridis', vmin=origin.min(), vmax=origin.max())
    ax.set_title(label)
    ax.set_xlabel('beta')
    ax.set_ylabel('gamma')
    if isinstance(true_optima, list) or isinstance(true_optima, np.ndarray):
        ax.plot(true_optima[0], true_optima[1], marker="o", color='red', markersize=7, label="true optima")

    params = params_path
    if isinstance(params, list) or isinstance(params, np.ndarray):
        xs = [] # beta
        ys = [] # gamma
        for param in params:
            xs.append(param[0])
            ys.append(param[1])

        ax.plot(xs, ys, marker="o", color='purple', markersize=5, label="optimization path")
        ax.plot(xs[0], ys[0], marker="o", color='white', markersize=9, label="initial point")
        ax.plot(xs[-1], ys[-1], marker="s", color='white', markersize=12, label="last point")