import argparse
from typing import List
import numpy as np
from scipy.fftpack import dctn, fftn
from scipy.ndimage import fourier_shift
import os

from data_loader import load_grid_search_data


def nonzero_ratio(ls: np.ndarray, metric: str = 'DCT'):
    if metric == 'DCT':
        f = dctn(ls)
    elif metric == 'FFT':
        f = np.fft.fftn(ls)
    elif metric == 'PS':
        f = fftn(ls)
        d = len(ls.shape)
        shift = (ls.shape[i] // 2 for i in range(d))
        f = fourier_shift(f, shift)
        f = np.abs(f)**2
        f /= f.max()

    nonzero_ids = np.argwhere(np.abs(f) > 1e-6)
    nonzero_ratio = len(nonzero_ids) * 1.0 / np.prod(ls.shape)

    # print("Using DCT:")
    # print("original shape            =", ls.shape)
    # print("original min, max         =", ls.min(), ls.max())
    # print("after transform, shape    =", f.shape)
    # print("after transform, min, max =", f.min(), f.max())
    # print("# nonzero, nonzero_ratio  =", len(nonzero_ids), nonzero_ratio)

    return nonzero_ratio


def measure_sparsity(
    p: int, ansatz: str, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list, device: str
):
    method = 'sv'
    miti_method = ''

    labels = ['DCT', 'FFT', 'PS']
    ratios = {label: [] for label in labels}

    if p == 1:
        bs = 50  # beta step
        gs = 2 * bs
    elif p == 2:
        bs = 12
        gs = 15
    elif p == 0:
        bs = 14
        gs = 100

    if len(n_seeds) == 1:
        seeds = list(range(n_seeds[0]))
    elif len(n_seeds) == 2:
        seeds = list(range(n_seeds[0], n_seeds[1]))

    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    for n_qubits in n_qubits_list:
        for seed in seeds:
            data, data_fname, data_dir = load_grid_search_data(
                n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
                noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
            )

            for label in labels:
                ratio = nonzero_ratio(data['data'], metric=label)
                ratios[label].append(ratio)

    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)

    for label in labels:
        print(f"{label} ratio =", ratios[label])
        ratios[label] = np.array(ratios[label]).reshape(
            len(n_qubits_list), len(seeds))

    save_dir = f"figs/sparsity/"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    save_fname = f"ns={n_qubits_list}-seeds={seeds}-ansatz={ansatz}-problem={problem}-noise={noise}-{p=}"
    save_path = os.path.join(save_dir, save_fname)
    print(f"data save to {save_path}")
    np.savez_compressed(
        save_path,
        **ratios,
        labels=labels,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns', type=int, nargs='+',
                        help="Your aims, vis, opt", required=True)
    parser.add_argument(
        '--p', type=int, help="Your aims, vis, opt", required=True)
    parser.add_argument('--noise', type=str,
                        help="Your aims, vis, opt", required=True)
    parser.add_argument('--problem', type=str,
                        help="Your aims, vis, opt", required=True)
    parser.add_argument('--n_seeds', type=int, nargs='+',
                        help="Your aims, vis, opt", required=True)
    parser.add_argument('--ansatz', type=str,
                        help="Your aims, vis, opt", required=True)
    parser.add_argument('--device', type=str,
                        help="Your aims, vis, opt", default=None)
    args = parser.parse_args()

    measure_sparsity(p=args.p, ansatz=args.ansatz, problem=args.problem,
                     noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns, device=args.device)
