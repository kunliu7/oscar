import argparse
import networkx as nx
import numpy as np
from qiskit_optimization.applications import Maxcut
from oscar.optimizer_wrapper import wrap_qiskit_optimizer_to_landscape_optimizer
from qiskit_aer.noise import NoiseModel
from qiskit.algorithms.optimizers.optimizer import Optimizer as QiskitOptimizer
from typing import List, Tuple
from qiskit.algorithms.optimizers import (
    ADAM,
    COBYLA,
    L_BFGS_B,
    SPSA
)
from qiskit.algorithms import QAOA
from qiskit_aer import AerSimulator
import os

from data_loader import get_interpolation_path_filename, get_recon_landscape, get_recon_pathname, load_grid_search_data, load_optimization_path

from oscar.noise_models import (
    get_depolarizing_error_noise_model,
)

from oscar.vis import (
    vis_landscapes
)

from oscar.utils import (
    arraylike_to_str,
    shift_parameters
)


def get_point_val(G: nx.Graph, p: int, last_pt: np.ndarray, noise_model: NoiseModel):
    maxcut = Maxcut(G)
    problem = maxcut.to_quadratic_program()
    C, offset = problem.to_ising()

    # qinst = AerSimulator(method='statevector', noise_model=noise_model)
    qinst = AerSimulator(shots=1024, noise_model=noise_model)

    qaoa = QAOA(
        reps=p,
        quantum_instance=qinst,
    )

    qaoa._check_operator_ansatz(C)
    energy_evaluation, expectation = qaoa.get_energy_evaluation(
        C, return_expectation=True
    )

    return energy_evaluation(last_pt)


def get_minimum_by_QAOA(G: nx.Graph, p: int, qiskit_init_pt: np.ndarray,
                        noise_model: NoiseModel, optimizer: QiskitOptimizer):
    """Get minimum cost value by random initialization.

    Returns:
        eigenvalue, qiskit_init_pt, params, values
    """
    maxcut = Maxcut(G)
    problem = maxcut.to_quadratic_program()
    C, offset = problem.to_ising()

    counts = []
    values = []
    params = []

    def cb_store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    qinst = AerSimulator(shots=1024, noise_model=noise_model)

    print("noise model", noise_model)

    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        initial_point=qiskit_init_pt,
        quantum_instance=qinst,
        callback=cb_store_intermediate_result
    )
    # print(qaoa.ansatz)
    result = qaoa.compute_minimum_eigenvalue(C)
    eigenvalue = result.eigenvalue.real

    print("offset                 :", offset)
    print("QAOA minimum           :", eigenvalue)
    print("Real minimum (+offset) :", eigenvalue + offset)

    return eigenvalue, qiskit_init_pt, params, values


def batch_eval_opt_on_recon_ls(n: int, seed_range: List[int], noise: str, opt: str):
    r"""Evaluate optimization on reconstructed landscape for different seeds.

    """
    p = 1
    if len(seed_range) == 1:
        seeds = list(range(seed_range[0]))
    elif len(seed_range) == 2:
        seeds = list(range(seed_range[0], seed_range[1]))
    else:
        raise NotImplementedError()

    if opt == 'ADAM':
        maxiter = 10000
    elif opt == 'COBYLA':
        maxiter = 1000
    elif opt == 'SPSA':
        maxiter = 100
    else:
        raise ValueError()

    miti_method = None

    intp_paths = []
    initial_points = []
    # for seed, opt in itertools.product(seeds, opts):
    for seed in seeds:
        print(f"{n=}, {seed=}, {opt=}")

        intp_path, initial_point = optimize_on_p1_reconstructed_landscape(
            n, p, seed, noise, miti_method,
            None, opt, None, maxiter, False
        )

        initial_points.append(initial_point)
        intp_paths.append(intp_path.copy())

    assert len(intp_paths) == len(seeds)
    print("n: ", n)
    print("seeds: ", seeds)
    print("opt: ", opt)

    save_dir = f"figs/second_optimize"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/opt_on_recon-{n=}-noise={noise}-seeds={arraylike_to_str(seeds)}-opt={opt}-{maxiter=}"
    print("save to", save_path)

    np.savez_compressed(
        save_path,
        save_path=save_path,
        seeds=seeds,
        initial_points=initial_points,
        intp_paths=intp_paths,
        opt=opt,
        maxiter=maxiter
    )


def optimize_on_p1_reconstructed_landscape(
    n: int, p: int, seed: int, noise: str, miti_method: str,
    initial_point: List[float], opt_name: str, lr: float, maxiter: int, is_sim: bool
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Optimize on p=1 reconstructed landscape with interpolation.

    Args:
        initial_point (List[float]): [beta, gamma]

    Returns:
        List[np.ndarray]: list of points on optimization path
        np.ndarray: initial point
    """
    noise_cfgs = noise.split('-')
    if noise_cfgs[0] == 'ideal':
        noise_model = None
    elif noise_cfgs[0] == 'depolar':
        noise_model = get_depolarizing_error_noise_model(
            float(noise_cfgs[1]), float(noise_cfgs[2]))
        print(noise_cfgs)
    else:
        raise NotImplementedError()

    problem = 'maxcut'
    method = 'sv'
    cs_seed = n
    assert p == 1

    sf = 0.05
    if p == 1:
        bs = 50
        gs = 100
    elif p == 2:
        bs = 12
        gs = 15
    else:
        raise NotImplementedError()
    ansatz = 'qaoa'
    is_vis = False
    data, data_fname, data_dir = load_grid_search_data(
        n_qubits=n, p=p, ansatz=ansatz, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
    )

    beta_bound = data['beta_bound']
    gamma_bound = data['gamma_bound']
    plot_range = data['plot_range']
    origin = data['data']

    recon_path, recon_fname, recon_dir = get_recon_pathname(
        ansatz, p, problem, method, noise, cs_seed, sf, data_fname)
    print("tend to save to", recon_path)
    recon = get_recon_landscape(
        ansatz, p, origin, sf, False, recon_path, cs_seed)

    G = nx.random_regular_graph(3, n, seed)
    maxcut = Maxcut(G)
    # print(maxcut.get_gset_result)
    maxcut_problem = maxcut.to_quadratic_program()
    C, offset = maxcut_problem.to_ising()

    bounds = np.array([
        [-beta_bound, beta_bound],
        [-gamma_bound, gamma_bound],
    ])

    print("bounds:", bounds)
    print("landscape shape:", recon.shape)

    # if initial_point is not provided, then randomly generate one
    print("initial point:", initial_point)
    if not initial_point:
        rng = np.random.default_rng(seed)
        # initial_point = np.array(initial_point)
        beta = rng.uniform(-beta_bound, beta_bound)
        gamma = rng.uniform(-gamma_bound, gamma_bound)

        initial_point = np.array([beta, gamma])
        print(f"{seed=}, {initial_point=}, {beta_bound=}, {gamma_bound=}")

    # ---------------- data prepare -------------------

    intp_path_path, intp_path_fname, intp_path_dir = get_interpolation_path_filename(
        n, p, problem, method, noise, opt_name, maxiter, initial_point, seed, miti_method)
    is_data_existed = os.path.exists(intp_path_path)

    if opt_name == 'ADAM':
        raw_optimizer = ADAM
    elif opt_name == 'SPSA':
        raw_optimizer = SPSA
    elif opt_name == 'COBYLA':
        raw_optimizer = COBYLA
    elif opt_name == 'L_BFGS_B':
        raw_optimizer = L_BFGS_B
    else:
        raise NotImplementedError()

    opt_params = {}
    if lr:
        opt_params['lr'] = lr
    if maxiter:
        opt_params['maxiter'] = maxiter
    print("optimizer: ", opt_name)
    optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(
        raw_optimizer
    )(
        bounds=bounds,
        landscape=recon,
        fun_type='INTERPOLATE_QISKIT',
        **opt_params
    )

    # opts = [ADAM,
    # AQGD,
    # CG,
    # COBYLA,
    # L_BFGS_B,
    # GSLS,
    # GradientDescent,
    # NELDER_MEAD,
    # NFT,
    # P_BFGS,
    # POWELL,
    # SLSQP,
    # SPSA,
    # QNSPSA,
    # TNC,
    # SciPyOptimizer]

    qinst = AerSimulator()  # meaningless, do not actually activate
    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        initial_point=initial_point,
        quantum_instance=qinst,
        # callback=cb_store_intermediate_result
    )
    result = qaoa.compute_minimum_eigenvalue(C)
    # print(qaoa.optimal_params)
    print("recon landscape minimum     :", result.eigenvalue)
    print("QAOA energy + offset        :", - (result.eigenvalue + offset))

    params = optimizer.params_path
    # ! there is some problem deriving the values directly
    # e.g. for ADAM, the values are NOT one-to-one corresponding to the params
    # intp_vals = optimizer.vals
    print("len of params:", len(params))

    params = [
        shift_parameters(_params, bounds)
        for _params in params
    ]

    # for some reason we forget to save the point corresponding to the initial point
    # when generating the data, so we need to compute it manually
    # but it takes time, so we only do it if needed
    if is_sim:
        # _, _, circ_path, _ = get_minimum_by_QAOA(G, p, initial_point, None, raw_optimizer(lr=lr, maxiter=maxiter))
        circ_path = load_optimization_path(
            n, p, problem, method, noise, opt_name, lr, maxiter, initial_point, seed, miti_method)
        print("len of circuit simulation path:", len(circ_path))
        circ_vals = []
        for ipt, pt in enumerate(circ_path):
            print(f"\r{ipt} th / {len(circ_path)}", end="")
            circ_vals.append(get_point_val(G, p, pt, None))
    else:
        circ_path = None
        circ_vals = None

    print(initial_point)

    save_path = intp_path_path
    intp_path_fname_base = os.path.splitext(intp_path_fname)[0]
    print("params save to =", save_path)
    np.savez_compressed(
        save_path,
        opt_name=opt_name,
        initial_point=initial_point,
        intp_path=params,  # interpolation
        # intp_vals=intp_vals,
        circ_path=circ_path,
        circ_vals=circ_vals
    )

    if is_vis:
        vis_landscapes(
            landscapes=[recon, origin],
            labels=["Interpolate", "Circuit Sim."],
            full_range=plot_range,
            true_optima=None,
            title="Origin and recon",
            save_path=f'{intp_path_dir}/vis-{intp_path_fname_base}.png',
            params_paths=[params, circ_path]
        )

    return params, initial_point


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', type=int, help="Number of qubits.", required=True)
    parser.add_argument('--seed_range', type=int, nargs="+", help="1 or 2 parameters;"
                        "if 1, range(args.seed_range[0]);"
                        "if 2, range(*args.seed_range)", required=True)
    parser.add_argument('--noise', type=str, required=True)
    parser.add_argument('--opt', type=str, required=True)
    args = parser.parse_args()

    batch_eval_opt_on_recon_ls(args.n, args.seed_range, args.noise, args.opt)
