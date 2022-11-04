
import numpy as np


def load_grid_search_data(
    n_qubits: int,
    p: int,
    problem: str,
    method: str,
    noise: str,
    beta_step: int,
    gamma_step: int,
    seed: int,
):
    # abbv.
    nq = n_qubits
    bs = beta_step
    gs = gamma_step

    data_dir = f"figs/grid_search/{problem}/{method}-{noise}"
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

    return data

