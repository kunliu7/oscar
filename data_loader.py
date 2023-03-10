
import numpy as np
from typing import List, Tuple, Union
import os

from oscar.n_dim_cs import recon_4D_landscape_by_2D
from oscar.compressed_sensing import (
    recon_2D_landscape,
)


"""
Usage:

recon_path, _, _ = get_recon_pathname(ansatz, p, problem, method, noise, cs_seed, sf, data_fname)
recon = get_recon_landscape(ansatz, p, origin, sf, False, recon_path, cs_seed)
"""

def get_recon_landscape(ansatz: str, p: int, origin: np.ndarray, sampling_frac: float,
    force_recon: bool, recon_save_path: str, cs_seed: int, is_save: bool=True
) -> np.ndarray:
    d = len(origin.shape)
    save_dir = os.path.dirname(recon_save_path)
    is_recon = force_recon or not os.path.exists(recon_save_path)

    if not is_recon:
        recon = np.load(recon_save_path, allow_pickle=True)['recon']
        print("recon landscape read from", recon_save_path)
        return recon

    if force_recon:
        print("force recon...")
    else:
        print(f'recon landscape does not exist in {recon_save_path}, start recon...')

    np.random.seed(cs_seed)
    if d == 2:
        recon = recon_2D_landscape(
            origin=origin,
            sampling_frac=sampling_frac
        )
    elif d == 4:
        recon = recon_4D_landscape_by_2D(
            origin=origin,
            sampling_frac=sampling_frac
        )
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(recon_save_path, recon=recon, sampling_frac=sampling_frac)
    print("not exists, save to", save_dir)
    
    return recon


def load_IBM_or_sim_grid_search_data(
    n_qubits: int, p: int, ansatz: str, problem: str, method: str, noise: str,
    beta_step: int, gamma_step: int, seed: int, miti_method: str=''
) -> Tuple[dict, str, str]:
    if 'ibm' in method:
        data, fname, fdir = load_IBM_grid_search_data(
            n_qubits, p, ansatz, problem, method, noise, beta_step, gamma_step, seed
        )
    else:
        data, fname, fdir = load_grid_search_data(
            n_qubits, p, ansatz, problem, method, noise, beta_step, gamma_step, seed, miti_method
        )

    return data, fname, fdir


def load_IBM_grid_search_data(
    n_qubits: int, p: int, ansatz: str, problem: str, method: str, noise: str,
    beta_step: int, gamma_step: int, seed: int
) -> Tuple[dict, str, str]:
    assert ansatz == 'qaoa'
    assert problem == 'maxcut'
    assert n_qubits == 6
    assert p == 1
    assert method in ['ibm_lagos', 'ibm_perth']
    assert noise in ['ideal_sim', 'noisy_sim', 'real']
    assert seed == 1

    fname = f"{problem}-{method}-{noise}-n={n_qubits}-{p=}-{seed=}-{beta_step}-{gamma_step}"
    save_dir = f"figs/ibm/IBM_Exp_2048/{fname}.npz"
    data = np.load(
        save_dir,
        allow_pickle=True
    )

    beta_bound = np.pi / 4 / p
    gamma_bound = np.pi / 2 / p

    full_range = {
        'beta': np.linspace(-beta_bound, beta_bound, beta_step),
        'gamma': np.linspace(-gamma_bound, gamma_bound, gamma_step)
    } 

    data_dict = dict(data)
    if 'ibm_H_ls' in data_dict:
        print("ibm data read from", save_dir)
        new_data = {
            'data': data_dict['ibm_H_ls'],
        }
    else:
        new_data = data_dict
    
    new_data['full_range'] = full_range
    return new_data, fname, save_dir


# def get_IBM_recon_landscape(
#     n_qubits: int, p: int, ansatz: str, problem: str, method: str, noise: str,
#     beta_step: int, gamma_step: int, seed: int, shots: int, cs_seed: int,
#     sampling_frac: float, force_recon: bool, is_save: bool=True
# )


def load_ibm_data(
    mid: int,
    seed: int,
    shots: int=2048
) -> dict:
    if shots == 2048:
        save_dir = f"figs/ibm/IBM_Exp_2048/ls-M-{mid}-seed-{seed}.npz"
    elif shots == 1024:
        save_dir = f"figs/ibm/Experiments/ls-M-{mid}-seed-{seed}.npz"
    else:
        raise ValueError("shots must be 1024 or 2048")

    data = np.load(
        save_dir,
        allow_pickle=True
    )

    p = 1
    beta_bound = np.pi / 4 / p
    gamma_bound = np.pi / 2 / p
    bs = 50
    gs = 100

    full_range = {
            'beta': np.linspace(-beta_bound, beta_bound, bs),
            'gamma': np.linspace(-gamma_bound, gamma_bound, gs)
        }

    landscapes = dict(data)
    print("ibm data read from", save_dir)
    new_data = {
        'data': landscapes['ibm_H_ls'],
        'full_range': full_range
    }
    return new_data
    

def load_grid_search_data(
    n_qubits: int,
    p: int,
    ansatz: str,
    problem: str,
    method: str,
    noise: str,
    beta_step: int,
    gamma_step: int,
    seed: int,
    miti_method: str=""
) -> Tuple[dict, str, str]:
    """Load grid search landscape data.

    Returns:
        Tuple[dict, str, str]: data; data filename; data directory.
    """
    # abbv.
    nq = n_qubits
    bs = beta_step
    gs = gamma_step

    data_dir = f"figs/grid_search/{ansatz}/{problem}/{method}-{noise}-p={p}"
    if ansatz == 'twolocal' or ansatz == 'uccsd':
        fname = f"{problem}-{method}-{noise}-n={nq}-p={p}-seed={seed}-{bs}.npz"
    elif ansatz == 'qaoa':
        fname = f"{problem}-{method}-{noise}-n={nq}-p={p}-seed={seed}-{bs}-{gs}.npz"
    else:
        raise ValueError(f"not support {ansatz} {problem} {noise}")

    data_path = f"{data_dir}/{fname}"
    print(f"\nread grid search data from {data_path}\n")
    data = np.load(
        data_path,
        allow_pickle=True
    )

    data = dict(data)

    # 统一 p=1 和 p=2 的 full_ranges
    beta_bound = data['beta_bound']
    gamma_bound = data['gamma_bound']

    if ansatz == 'qaoa':
        full_range = {
            'beta': np.linspace(-beta_bound, beta_bound, bs),
            'gamma': np.linspace(-gamma_bound, gamma_bound, gs)
        }

        full_ranges = []
        for _ in range(p):
            full_ranges.append(full_range['beta'])
        for _ in range(p):
            full_ranges.append(full_range['gamma'])

        data['full_ranges'] = full_ranges
        data['full_range'] = full_range
        if p == 1:
            data['plot_range'] = data['full_range'].copy()
        elif p == 1:
            data['plot_range'] = {
                'beta': list(range(bs ** p)),
                'gamma': list(range(gs ** p))
            }
        else:
            data['plot_range'] = None


    elif ansatz == 'twolocal':
        # assert n_qubits % 2 == 0
        n_half = n_qubits // 2
        shape0 = data['data'].shape[0]
        # data['data'] = data['data'].reshape((shape0 ** n_half, shape0 ** n_half))

        data['full_range'] = {
            'beta': np.linspace(-beta_bound, beta_bound, bs),
            # 'gamma': np.linspace(-gamma_bound, gamma_bound, gs)
        }
        data['plot_range'] = { 
            'beta': list(range(shape0)),
            'gamma': list(range(shape0))
        }
    elif ansatz == 'uccsd':
        data['full_range'] = {
            'beta': np.linspace(-beta_bound, beta_bound, bs),
            # 'gamma': np.linspace(-gamma_bound, gamma_bound, gs)
        }
        data['plot_range'] = None

    """
    np.savez_compressed(
        f"{dirpath}/{args.problem}-{backend_config}-{n=}-{p=}-{seed=}-{args.beta_steps}-{args.gamma_steps}",
        data=data,
        time=time,
        offset=offset,
        beta_bound=beta_bound,
        gamma_bound=gamma_bound,
        grid=grid,
        optimization_result=optimization_result,
        **args.__dict__,
    )
    """

    return data, fname, data_dir


def get_recon_pathname(ansatz: str, p:int, problem: str, method: str,
    noise: str, cs_seed: str, sampling_frac: float, origin_fname: str
) -> Tuple[str, str, str]:
    """

    Returns:
        Tuple[str, str, str]: path, filename, directory
    """
    sf = sampling_frac
    # 和Tianyi的目录结构保持一致
    recon_dir = f"figs/grid_search_recon/{ansatz}/{problem}/{method}-{noise}-p={p}"
    recon_fname = f"recon-cs_seed={cs_seed}-sf={sf:.3f}-{origin_fname}"
    recon_path = f"{recon_dir}/{recon_fname}"

    return recon_path, recon_fname, recon_dir


def get_interpolation_path_filename(
    n_qubits: int,
    p: int,
    problem: str,
    method: str,
    noise: str,
    optimizer: str,
    maxiter: int,
    initial_point: List[float],
    seed: int,
    miti_method: str=""
) -> Tuple[str, str, str]:
    n = n_qubits
    if isinstance(initial_point, np.ndarray):
        init_pt = list(initial_point)
    elif isinstance(initial_point, list):
        init_pt = initial_point
    else:
        init_pt = None
        print(initial_point)
        # raise ValueError()
    
    # assert len(init_pt) == 2 * p
    data_dir = f"figs/opt_on_recon_landscape/{problem}/{method}-{noise}-p={p}" 
    fname = f"{problem}-{method}-{noise}-{n=}-{p=}-{seed=}-{optimizer}-{maxiter=}-{init_pt}.npz"

    data_path = f"{data_dir}/{fname}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_path, fname, data_dir


def load_optimization_path(
    n_qubits: int,
    p: int,
    problem: str,
    method: str,
    noise: str,
    optimizer: str,
    lr: float,
    maxiter: int,
    initial_point: List[int],
    seed: int,
    miti_method: str=""
) -> Tuple[dict, str, str]:
    n = n_qubits
    if isinstance(initial_point, np.ndarray):
        init_pt = list(initial_point)
    elif isinstance(initial_point, list):
        init_pt = initial_point
    else:
        init_pt = None
        # raise ValueError()
    
    # assert len(init_pt) == 2 * p
    data_dir = f"figs/optimization/{problem}/{method}-{noise}-p={p}" 

    fname = f"{problem}-{method}-{noise}-{n=}-{p=}-{seed=}-{optimizer}-{maxiter=}-{init_pt}"
    if lr:
        fname += f"-{lr}"

    fname += ".npz"
    data_path = f"{data_dir}/{fname}"
    data = np.load(data_path, allow_pickle=True)
    print("opt data load from", data_path)

    return data['optimizer_path']

