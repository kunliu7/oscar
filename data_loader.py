
import numpy as np
from typing import Tuple

def hello() -> Tuple[dict, str, str]:
    """_summary_

    Returns:
        Tuple[dict, str, str]: _description_
    """

def load_grid_search_data(
    n_qubits: int,
    p: int,
    problem: str,
    method: str,
    noise: str,
    beta_step: int,
    gamma_step: int,
    seed: int,
) -> Tuple[dict, str, str]:
    """Load grid search landscape data.

    Returns:
        Tuple[dict, str, str]: data; data filename; data directory.
    """
    # abbv.
    nq = n_qubits
    bs = beta_step
    gs = gamma_step

    data_dir = f"figs/grid_search/{problem}/{method}-{noise}-p={p}"
    if noise == 'depolar-0.001-0.02':
        fname = f"{problem}-{method}-{noise}-n={nq}-p={p}-seed={seed}-{bs}-{gs}.npz"
    else:
        fname = f"{method}-{noise}-n={nq}-p={p}-seed={seed}-{bs}-{gs}.npz"

    data_path = f"{data_dir}/{fname}"
    print("read data from ", data_path)
    data = np.load(
        data_path,
        allow_pickle=True
    )

    data = dict(data)

    # 统一 p=1 和 p=2 的 full_ranges
    beta_bound = data['beta_bound']
    gamma_bound = data['gamma_bound']

    full_range = {
        'beta': np.linspace(-beta_bound, gamma_bound, bs),
        'gamma': np.linspace(-gamma_bound, gamma_bound, gs)
    }

    full_ranges = []
    for _ in range(p):
        full_ranges.append(full_range['beta'])
    for _ in range(p):
        full_ranges.append(full_range['gamma'])

    data['full_ranges'] = full_ranges
    data['full_range'] = full_range

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

