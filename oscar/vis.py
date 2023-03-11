import os

import matplotlib.pyplot as plt
import networkx as nx
# vis
import numpy as np

from .utils import noisy_qaoa_maxcut_energy


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
        # , cmap='viridis', vmin=origin.min(), vmax=origin.max())
        im = axs[idx].pcolormesh(X, Y, landscape)
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
    landscapes,  # list of np.ndarray
    labels,  # list of labels of correlated landscapes
    full_range,  # dict,
    true_optima,
    title,
    save_path,  # figure save path
    params_paths,  # list of list of parameters correlated to landscapes
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
        # , cmap='viridis', vmin=origin.min(), vmax=origin.max())
        im = axs[idx].pcolormesh(X, Y, landscape)
        axs[idx].set_title(labels[idx])
        axs[idx].set_xlabel('beta')
        axs[idx].set_ylabel('gamma')
        if isinstance(true_optima, list) or isinstance(true_optima, np.ndarray):
            axs[idx].plot(true_optima[0], true_optima[1], marker="o",
                          color='red', markersize=7, label="true optima")

        params = params_paths[idx]
        if isinstance(params, list) or isinstance(params, np.ndarray):
            xs = []  # beta
            ys = []  # gamma
            for param in params:
                xs.append(param[0])
                ys.append(param[1])

            axs[idx].plot(xs, ys, marker="o", color='purple',
                          markersize=5, label="optimization path")
            axs[idx].plot(xs[0], ys[0], marker="o", color='white',
                          markersize=9, label="initial point")
            axs[idx].plot(xs[-1], ys[-1], marker="s", color='white',
                          markersize=12, label="last point")

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
    landscape,  # list of np.ndarray
    label,  # list of labels of correlated landscapes
    full_range,  # dict,
    true_optima,
    params_path,  # list of list of parameters correlated to landscapes
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
    # , cmap='viridis', vmin=origin.min(), vmax=origin.max())
    im = ax.pcolormesh(X, Y, landscape)
    ax.set_title(label)
    ax.set_xlabel('beta')
    ax.set_ylabel('gamma')
    if isinstance(true_optima, list) or isinstance(true_optima, np.ndarray):
        ax.plot(true_optima[0], true_optima[1], marker="o",
                color='red', markersize=7, label="true optima")

    params = params_path
    if isinstance(params, list) or isinstance(params, np.ndarray):
        xs = []  # beta
        ys = []  # gamma
        for param in params:
            xs.append(param[0])
            ys.append(param[1])

        ax.plot(xs, ys, marker="o", color='purple',
                markersize=5, label="optimization path")
        ax.plot(xs[0], ys[0], marker="o", color='white',
                markersize=9, label="initial point")
        ax.plot(xs[-1], ys[-1], marker="s", color='white',
                markersize=12, label="last point")
