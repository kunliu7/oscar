import argparse
import os
import numpy as np
from typing import List

from oscar.vis import (
    _vis_recon_distributed_landscape,
)

from oscar.utils import (
    get_curr_formatted_timestamp,
)

from oscar.compressed_sensing import (
    cal_recon_error,
    two_D_CS_p1_recon_with_distributed_landscapes,
)

from data_loader import load_IBM_or_sim_grid_search_data
from sklearn.linear_model import LinearRegression


def ndarray2D_to_set_of_tuples(a):
    ids = []

    for idx in a:
        ids.append(tuple(idx))

    ids = set(ids)
    print("ndarray to set:", len(ids))
    return ids


def T_flatten(a: np.ndarray):
    assert len(a.shape) == 2
    return a.copy().T.flat[:]


def inv_T_flatten(a: np.ndarray, shape: tuple):
    assert len(a.shape) == 1 and len(shape) == 2
    return a.reshape(shape[::-1]).T


def normalize_by_linear_regression(ls1, ls2, n_pts, ri):
    shape = ls1.shape
    print("shape =", shape)
    assert (ls1 == inv_T_flatten(T_flatten(ls1), ls1.shape)).all()
    fls1 = T_flatten(ls1)
    fls2 = T_flatten(ls2)

    rng = np.random.default_rng(7)

    # pick n_pts random points from ri to train NCM
    # note that ri is fixed for normalization and parallel reconstruction
    ids = rng.choice(ri, n_pts, replace=False)
    print(ids.shape)

    y = fls1[ids]
    x = fls2[ids]
    print(x.shape)
    print(y.shape)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model = model.fit(x, y)

    r_sq = model.score(x, y)
    print('coefficient of determination(𝑅²) :', r_sq)
    # coefficient of determination(𝑅²) : 0.715875613747954
    print('intercept:', model.intercept_)
    # （标量） 系数b0 intercept: 5.633333333333329 -------this will be an array when y is also 2-dimensional
    print('slope:', model.coef_)

    # ls2_flat = ls2.reshape(-1)
    ls2_flat = ls2.flatten().reshape(-1, 1)
    y_pred = model.predict(ls2_flat)
    print(y_pred.shape)
    ls2_normalized = y_pred.reshape(shape)

    return ls1, ls2_normalized


def reconstruct_by_distributed_landscapes_two_noisy_simulations_top(
    n_qubits_list: List[int], p: int, sf: float, noise1: str, noise2: str, seed: int,
    method1: str, method2: str,
    normalize: str, norm_frac: float, ratios: List[float],
    error_type: str, recon_dir: str
):
    is_vis = False
    is_existing_recon = True if isinstance(recon_dir, str) else False
    if not is_existing_recon:
        signature = get_curr_formatted_timestamp()
        recon_dir = f"figs/recon_distributed_landscape/{signature}"
        if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)
    else:
        print("recon_dir specified:", recon_dir)

    errors1 = []
    errors2 = []
    # def get_data(noise: str):
    #     r"""
    #     Deal with the data from IBM and grid search together.
    #     """
    #     if noise == 'ibm-1':
    #         assert seed == 1, "seed should be 1 for ibm-1"
    #         noisy_data = load_ibm_data(mid=1, seed=seed)
    #     elif noise == 'ibm-2':
    #         assert seed == 1, "seed should be 1 for ibm-2"
    #         noisy_data = load_ibm_data(mid=2, seed=seed)
    #     elif os.path.exists(noise): # ! special case
    #         assert seed == 1 and p == 1, "seed should be 1 and p should be 1 for local data"
    #         path = noise
    #         noisy_data = np.load(path, allow_pickle=True)
    #         noisy_data = dict(noisy_data)
    #         noisy_data['full_range'] = None # ! special case, only for visualization
    #     else:
    #         ansatz = 'qaoa'
    #         noisy_data, _, _ = load_grid_search_data(
    #             n_qubits=n_qubits, p=p, ansatz=ansatz,
    #             problem=problem, method=method,
    #             noise=noise, beta_step=50, gamma_step=100, seed=seed,
    #         )

    #     return noisy_data

    problem = 'maxcut'
    ansatz = 'qaoa'
    bs = 50
    gs = 100
    for n_qubits in n_qubits_list:
        # only method and noise are different
        noisy_data1, _, _ = load_IBM_or_sim_grid_search_data(
            n_qubits=n_qubits, p=p, ansatz=ansatz, method=method1,
            problem=problem, seed=seed,
            noise=noise1, beta_step=bs, gamma_step=gs,
        )

        noisy_data2, _, _ = load_IBM_or_sim_grid_search_data(
            n_qubits=n_qubits, p=p, ansatz=ansatz, method=method2,
            problem=problem, seed=seed,
            noise=noise2, beta_step=bs, gamma_step=gs,
        )

        if noisy_data1['full_range'] is not None:
            full_range = noisy_data1['full_range']
        else:
            full_range = noisy_data2['full_range']

        noisy1 = noisy_data1['data']
        noisy2 = noisy_data2['data']

        # choose random points for reconstruction.
        rng = np.random.default_rng(0)
        landscape_shape = noisy1.shape
        n_pts = np.prod(landscape_shape)
        k = round(sf * n_pts)
        random_indices = rng.choice(np.prod(n_pts), k, replace=False)

        # --------- data prepared OK, check if want to normalize -----------
        if normalize == 'linear':
            print("normalized by linear regression")
            n_normalize = round(norm_frac * sf * n_pts)

            # ! Use points from `random_indices` to normalize
            noisy1, noisy2 = normalize_by_linear_regression(
                noisy1, noisy2, n_normalize, random_indices)
            datas = [noisy1, noisy2]
        elif normalize == None:
            print("do not normalize")
            datas = [noisy1, noisy2]
        else:
            raise NotImplementedError()

        for data in datas:
            print(data.shape)

        ratios_cfg1 = ratios
        print(ratios_cfg1)

        for ratio in ratios_cfg1:
            ratios = [ratio, 1-ratio]

            recon_fname = f"recon-n={n_qubits}-{p=}-ratios={ratios}-sf={sf:.3f}-norm={normalize}-nf={norm_frac:.3f}"
            if not is_existing_recon:
                if p == 1:
                    recon = two_D_CS_p1_recon_with_distributed_landscapes(
                        origins=datas,
                        sampling_frac=sf,
                        ratios=[ratio, 1-ratio],
                        ri=random_indices
                    )
                elif p == 2:
                    raise NotImplementedError()

                np.savez_compressed(f"{recon_dir}/{recon_fname}", recon=recon)

            else:
                recon = np.load(f"{recon_dir}/{recon_fname}.npz",
                                allow_pickle=True)['recon']

            if is_vis:
                _vis_recon_distributed_landscape(
                    landscapes=datas + [recon],
                    labels=[noise1, noise2, f'recon. by {ratios} of each'],
                    full_range=full_range,
                    bounds=None,
                    true_optima=None,
                    title=f'reconstruct distributed landscapes, sampling fraction: {sf:.3f}',
                    save_path=f"{recon_dir}/{recon_fname}.png"
                )

            # error1 = np.linalg.norm(noisy1 - recon)
            error1 = cal_recon_error(
                noisy1.reshape(-1), recon.reshape(-1), error_type)
            errors1.append(error1)

            # error2 = np.linalg.norm(noisy2 - recon)
            error2 = cal_recon_error(
                noisy2.reshape(-1), recon.reshape(-1), error_type)
            errors2.append(error2)

            # print(f"reconstruct error 1: {error1}; error 2: {error2}")

    print('ratio cfg:', ratios_cfg1)
    print("Sqrt MSE between recon and noise1's original:", errors1)
    print("Sqrt MSE between recon and noise2's original:", errors2)

    errors1 = np.array(errors1).reshape(len(n_qubits_list), len(ratios_cfg1))
    errors2 = np.array(errors2).reshape(len(n_qubits_list), len(ratios_cfg1))

    print('')

    save_path = f"{recon_dir}/dist_recon_errors-ns={n_qubits_list}-p={p}-sf={sf:.3f}-n1={noise1}-n2={noise2}-m1={method1}-m2={method2}-r={ratios_cfg1}-norm={normalize}-nf={norm_frac:.3f}-error={error_type}"
    print(f"recon errors save to {save_path}")
    np.savez_compressed(
        save_path,
        n_qubits_list=n_qubits_list,
        errors1=errors1,
        errors2=errors2,
        ratios=ratios_cfg1
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', type=int, help="QAOA circuit depth.", required=True)
    parser.add_argument(
        '--noise1', help="Noise type of the first landscape.", type=str)
    parser.add_argument(
        '--noise2', help="Noise type of the second landscape.", type=str)
    parser.add_argument('--seed', type=int,
                        help="Seed of the problem instance.", required=True)
    parser.add_argument('--sf', type=float,
                        help="Sampling fraction.", required=True)
    parser.add_argument('--normalize', type=str,
                        help="Normalized method. `linear` only.", default=None)
    parser.add_argument('--norm_frac', type=float,
                        help="Percentage of points to train Noise Compensation Model.", default=0)
    parser.add_argument('--ns', type=int, nargs='+',
                        help="List of #Qubits.", required=True)
    parser.add_argument('--error', type=str,
                        help="Type of error.", required=True)
    parser.add_argument('--recon_dir', type=str,
                        help="Reconstruction landscape directory.", default=None)
    parser.add_argument('--ratios', type=float, nargs='+',
                        help="Ratios of samples from landscape 1 used to normalized.", default=[0.2, 0.5, 0.8, 1.0])
    parser.add_argument(
        '--method1', help="Method to generate the landscape 1.", type=str, default='sv')
    parser.add_argument(
        '--method2', help="Method to generate the landscape 2.", type=str, default='sv')

    args = parser.parse_args()

    reconstruct_by_distributed_landscapes_two_noisy_simulations_top(
        n_qubits_list=args.ns, p=args.p, sf=args.sf, noise1=args.noise1, noise2=args.noise2,
        seed=args.seed, method1=args.method1, method2=args.method2,
        normalize=args.normalize, norm_frac=args.norm_frac, ratios=args.ratios,
        error_type=args.error, recon_dir=args.recon_dir
    )
