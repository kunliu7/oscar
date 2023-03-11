import argparse
import numpy as np
from scipy.ndimage import laplace
from typing import Callable, List, Optional, Tuple
from scipy.spatial.distance import (
    cosine
)

from oscar.vis import(
    vis_landscapes
)

from oscar.utils import (
    get_curr_formatted_timestamp,
)

from oscar.compressed_sensing import (
    cal_recon_error
)

from data_loader import get_recon_landscape, get_recon_pathname, load_grid_search_data


def cal_multi_errors(a, b):
    diff = {}
    a = a.reshape(-1)
    b = b.reshape(-1)
    # diff['L2-norm'] = np.linalg.norm(a - b)
    diff['SqrtMSE'] = cal_recon_error(a, b, 'RMSE')
    # diff['1-NCC'] = 1 - cal_recon_error(a, b, "CROSS_CORRELATION")
    diff['COS'] = cosine(a, b)
    return diff


def cal_gap(C_opt, full, recon):
    min_full = np.max(full)
    min_recon = np.max(recon)
    print("C_opt: ", C_opt)
    print(f"min_full: {min_full}, C_opt - min_full: {C_opt - min_full}")
    print(f"min_miti_recon: {min_recon}, C_opt - min_recon: {C_opt - min_recon}")


def grad_naive(ls):
    assert len(ls.shape) == 2
    a = ls[:, :-1]
    b = ls[:, 1:]
    # print(a.shape)
    # print(b.shape)
    g1 = a - b

    a = ls[1:, :]
    b = ls[:-1, :]
    # print(a.shape)
    # print(b.shape)
    g2 = a - b
    return [g1, g2]


def cal_laplace(ls):
    return laplace(ls)


def second_derivative(ls: np.ndarray) -> float:
    # print(ls.shape)
    g1 = np.diff(ls, n=2, axis=0)
    g2 = np.diff(ls, n=2, axis=1)

    # f[:-2] - 2* f[1:-1] + f[2:]
    g1 = ls[:, :-2] - 2 * ls[:, 1:-1] + ls[:, 2:]
    g2 = ls[:-2, :] - 2 * ls[1:-1, :] + ls[2:, :]

    # print(g1.shape)
    # print(g2.shape)

    smooth = (g1**2).sum() + (g2**2).sum()
    smooth /= 4
    return smooth


def cal_smoothness(ls: np.ndarray) -> float:
    """
    sd(diff(x))/abs(mean(diff(x)))
    sd(diff(y))/abs(mean(diff(y)))
    """
    # print(ls.shape)
    # grad = np.gradient(ls)
    # grad = grad_naive(ls)
    grad = cal_laplace(ls)
    std_of_grad = [np.abs(g).std() for g in grad]
    abs_mean = [np.abs(g).mean() for g in grad]

    std_of_grad = np.array(std_of_grad)
    abs_mean = np.array(abs_mean)
    # print(std_of_grad.shape)
    # print(abs_mean.shape)

    smoothness = std_of_grad / (abs_mean) # + 0.01)
    return smoothness.mean()


def cal_variance(ls) -> float:
    return np.var(ls)


def var_of_grad(ls) -> list:
    grad = np.gradient(ls)

    var_of_grad = [g.var() for g in grad]
    return var_of_grad


def cal_barren_plateaus(ls) -> float:
    return np.mean(var_of_grad(ls))


def compare_by_matrics(ls1, ls2, metrics):
    rsts = []
    for met in metrics:
        met1 = met(ls1)
        met2 = met(ls2)
        rst = [met1, met2, met1 - met2]
        
    rsts.append(rst)
    return rsts


def print_table(noisy, richard, linear, metric):
    lss = [noisy, richard, linear]
    # for met in metrics:
    vals = []
    for ls in lss:
        vals.append(metric(ls))

    print(metric.__name__, vals)


def compare_with_ideal_landscapes(ideal, ls1, ls2):
    diff1 = ideal - ls1
    diff2 = ideal - ls2

    ids = np.abs(diff1) < np.abs(diff2)     # indices that Richardson is closer to ideal than linear

    # print("n_pts =", np.sum(ids))
    print("abs(diff1) <  abs(diff2), num of such points =", np.sum(ids == True))
    print("abs(diff1) >= abs(diff2), num of such points =", np.sum(ids == False))
    
    print("var diff1 =", np.var(diff1))
    print("var diff2 =", np.var(diff2))

    print("var abs(diff1) =", np.var(np.abs(diff1)))
    print("var abs(diff2) =", np.var(np.abs(diff2)))


def vis_case_compare_mitigation_method(check: bool=False):
    is_reconstructed = False 
    is_test = check

    method = 'sv'
    problem = 'maxcut'
    miti_method1 = 'zne-RichardsonFactory'
    miti_method2 = 'zne-LinearFactory'

    # noise-3
    p1 = 0.001
    p2 = 0.02
    n_qubits = 16

    noise = f'depolar-{p1}-{p2}'

    # n_qubits = 16
    cs_seed = n_qubits
    p = 1
    sf = 0.05
    seed = 0
    if p == 2:
        bs = 12
        gs = 15
    elif p == 1:
        bs = 50
        gs = 100
    else:
        raise ValueError("Invalid depth of QAOA")
    ansatz = 'qaoa'

    if is_test:
        ideal_data1, ideal_data_fname, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method1
        )
        ideal_data2, ideal_data_fname, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method2
        )
        full_range = ideal_data1['full_range']
        vis_landscapes(
            # landscapes=[origin['unmitis'], miti1, miti2, miti1_recon, miti2_recon],
            landscapes=[ideal_data1['data'], ideal_data2['data']],
            labels=[miti_method1, miti_method2],
            full_range=full_range,
            true_optima=None,
            title="Compare different ZNE configs and reconstruction",
            save_path="paper_figs/debug_miti.png",
            params_paths=[None, None]
        )
        return
    else:
        ideal_data, _, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
            noise='ideal', beta_step=bs, gamma_step=gs, seed=seed
        )
        full_range = ideal_data['full_range']
        ideal = ideal_data['data']
        
        noisy_data, noisy_data_fname, _ = load_grid_search_data(
            n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed
        )
        noisy = noisy_data['data']
    
        noisy_recon_path, _, _ = get_recon_pathname(ansatz, p, problem, method, noise, cs_seed, sf, noisy_data_fname)
        noisy_recon = get_recon_landscape(ansatz, p, noisy, sf, is_reconstructed, noisy_recon_path, cs_seed)
 
    if not is_reconstructed:
        timestamp = get_curr_formatted_timestamp()
    else:
        timestamp = "2022-11-07_13:55:52_OK" # TODO

    # -------- derive miti1 data

    miti1_data, miti1_data_fname, _ = load_grid_search_data(
        n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method1
    )
    miti1 = miti1_data['data']
    # mitigation_method1 = miti1_data['mitigation_method']
    # mitigation
    print(miti1_data['mitigation_method'], miti1_data['mitigation_config'])
    # print(mitigation_config1)

    # exit()
    recon1_path, _, _ = get_recon_pathname(ansatz, p, problem, method, noise, cs_seed, sf, miti1_data_fname)
    # recon1_path = f"figs/recon_p2_landscape/{timestamp}/recon-sf={sf:.3f}-cs_seed={cs_seed}-{miti1_data_fname}"
    miti1_recon = get_recon_landscape(ansatz, p, miti1, sf, is_reconstructed, recon1_path, cs_seed)

    # -------- derive miti2 data

    print("\n\n")

    miti2_data, miti2_data_fname, _ = load_grid_search_data(
        n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
        noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method2
    )
    miti2 = miti2_data['data']

    recon2_path, _, _ = get_recon_pathname(ansatz, p, problem, method, noise, cs_seed, sf, miti2_data_fname)
    # recon2_path = f"figs/recon_p2_landscape/{timestamp}/recon-sf={sf:.3f}-cs_seed={cs_seed}-{miti2_data_fname}"
    miti2_recon = get_recon_landscape(ansatz, p, miti2, sf, is_reconstructed, recon2_path, cs_seed)
    
    print(miti2_data['mitigation_method'], miti2_data['mitigation_config'])


    metrics = [second_derivative, cal_smoothness, cal_barren_plateaus, cal_variance]
    for metric in metrics:
        print("origin: ", end="")
        print_table(noisy, miti1, miti2, metric)

        print("recon: ", end="")
        print_table(noisy_recon, miti1_recon, miti2_recon, metric)
        print("")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', default=False)
    args = parser.parse_args()
    vis_case_compare_mitigation_method(args.check)