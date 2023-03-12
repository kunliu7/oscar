from oscar.vis import (
    vis_landscapes,
)
from oscar.compressed_sensing import (
    cal_recon_error,
)
import argparse
import itertools
from typing import List
import numpy as np
import os
from scipy.spatial.distance import (
    cosine
)
from data_loader import load_grid_search_data, get_recon_landscape


def recon_landscapes_varying_qubits_and_instances(
    p: int, ansatz: str, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list,
    bs: int, gs: int, error_type: str
):
    is_vis = False
    is_force_recon = False
    method = 'sv'
    miti_method = ''
    mses = []
    coss = []

    if ansatz == 'qaoa':
        if gs == None or bs == None:
            if p == 1:
                bs = 50  # beta step
                gs = 2 * bs
            elif p == 2:
                bs = 12
                gs = 15
            else:
                raise ValueError('p must be 1 or 2 for qaoa')

    sfs = np.arange(0.01, 0.11, 0.02)
    if len(n_seeds) == 1:
        seeds = list(range(n_seeds[0]))
    elif len(n_seeds) == 2:
        seeds = list(range(n_seeds[0], n_seeds[1]))

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)

    for n_qubits in n_qubits_list:
        cs_seed = n_qubits  # ! compare horizontally

        for seed in seeds:
            data, data_fname, data_dir = load_grid_search_data(
                n_qubits=n_qubits, p=p, ansatz=ansatz, problem=problem, method=method,
                noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
            )

            plot_range = data['plot_range']

            # 和Tianyi代码使用相同目录结构
            recon_dir = f"figs/grid_search_recon/{ansatz}/{problem}/{method}-{noise}-p={p}"

            for sf in sfs:
                recon_fname = f"recon-cs_seed={cs_seed}-sf={sf:.3f}-{data_fname}"
                recon_path = f"{recon_dir}/{recon_fname}"

                origin = data['data']
                recon = get_recon_landscape(ansatz, p, origin, sf, is_force_recon,
                                            recon_path, cs_seed)

                mse = cal_recon_error(origin.reshape(-1),
                                      recon.reshape(-1), error_type)
                # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
                cos = cosine(origin.reshape(-1), recon.reshape(-1))
                mses.append(mse)
                coss.append(cos)

                # ncc = cal_recon_error()
                print("RMSE: ", mse)
                print("Cosine: ", cos)

                base_recon_fname = os.path.splitext(recon_fname)[0]
                if is_vis:
                    vis_landscapes(
                        landscapes=[origin, recon],
                        labels=["origin", "recon"],
                        full_range={
                            "beta": plot_range['beta'],
                            "gamma": plot_range['gamma']
                        },
                        true_optima=None,
                        title="Origin and recon",
                        save_path=f'{recon_dir}/vis/vis-{base_recon_fname}.png',
                        params_paths=[None, None]
                    )

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)

    print("mse =", mses)
    print("cos =", coss)
    mses = np.array(mses)
    coss = np.array(coss)

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    # timestamp = get_curr_formatted_timestamp()
    recon_error_save_dir = f"{recon_dir}/recon_error_ns={n_qubits_list}-seeds={seeds}-sfs={sfs}-error={error_type}"
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    print(f"recon error data save to {recon_error_save_dir}")
    np.savez_compressed(
        recon_error_save_dir,
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs
    )

@DeprecationWarning
def gen_heapmap_by_varying_sampling_fraction_and_beta_step(
    p: int, problem: str, noise: str, n_seeds: List[int], n_qubits_list: list, error_type: str
):
    is_recon = False

    method = 'sv'
    miti_method = ''
    mses = []
    coss = []

    sfs = np.arange(0.01, 0.11, 0.02)
    if len(n_seeds) == 1:
        seeds = list(range(n_seeds[0]))
    elif len(n_seeds) == 2:
        seeds = list(range(n_seeds[0], n_seeds[1]))

    bss = range(25, 76, 5)

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)
    print("bss =", bss)

    # 和Tianyi代码使用相同目录结构
    recon_dir = f"figs/grid_search_recon/{problem}/{method}-{noise}-p={p}"
    for n_qubits, seed, sf, bs in itertools.product(n_qubits_list, seeds, sfs, bss):
        cs_seed = n_qubits  # ! compare horizontally
        gs = 2 * bs

        data, data_fname, data_dir = load_grid_search_data(
            n_qubits=n_qubits, p=p, problem=problem, method=method,
            noise=noise, beta_step=bs, gamma_step=gs, seed=seed, miti_method=miti_method
        )

        plot_range = data['plot_range']

        recon_fname = f"recon-cs_seed={cs_seed}-sf={sf:.3f}-{data_fname}"
        recon_path = f"{recon_dir}/{recon_fname}"

        origin = data['data']
        recon = get_recon_landscape(p, origin, sf, is_recon,
                                    recon_path, cs_seed)

        mse = cal_recon_error(origin.reshape(-1),
                              recon.reshape(-1), error_type)
        # ncc = cal_recon_error(landscape.reshape(-1), recon.reshape(-1), "CROSS_CORRELATION")
        cos = cosine(origin.reshape(-1), recon.reshape(-1))
        mses.append(mse)
        coss.append(cos)

        # ncc = cal_recon_error()
        print("NRMSE: ", mse)
        print("Cosine: ", cos)

        base_recon_fname = os.path.splitext(recon_fname)[0]
        vis_landscapes(
            landscapes=[origin, recon],
            labels=["origin", "recon"],
            full_range={
                "beta": plot_range['beta'],
                "gamma": plot_range['gamma']
            },
            true_optima=None,
            title="Origin and recon",
            save_path=f'{recon_dir}/vis/vis-{base_recon_fname}.png',
            params_paths=[None, None]
        )

    print("noise =", noise)
    print("n qubits list =", n_qubits_list)
    print("seeds =", seeds)
    print("sfs =", sfs)

    print("mse =", mses)
    print("cos =", coss)
    mses = np.array(mses)
    coss = np.array(coss)

    mses = mses.reshape(len(n_qubits_list), len(seeds), len(sfs), len(bss))
    coss = coss.reshape(len(n_qubits_list), len(seeds), len(sfs), len(bss))
    print("mse's shape =", mses.shape)
    print("cos's shape =", coss.shape)
    # timestamp = get_curr_formatted_timestamp()
    recon_error_save_dir = f"{recon_dir}/recon_error_ns={n_qubits_list}-seeds={seeds}-sfs={sfs}-bss={bss}-error={error_type}"
    print(f"recon error data save to {recon_error_save_dir}")
    np.savez_compressed(
        recon_error_save_dir,
        mses=mses,
        coss=coss,
        n_qubits_list=n_qubits_list,
        seeds=seeds,
        sfs=sfs,
        bss=bss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aim', type=str,
                        help="Your aims to run the script.", required=True)
    parser.add_argument('--ns', type=int, nargs='+',
                        help="list of #Qubits.", required=True)
    parser.add_argument('--bs', type=int, help="Beta steps.", default=None)
    parser.add_argument('--gs', type=int, help="Gamma steps.", default=None)
    parser.add_argument('-p', type=int, help="#Layers.", required=True)
    parser.add_argument('--noise', type=str, help="Noise type.", required=True)
    parser.add_argument('--problem', type=str, help="Problem.", required=True)
    parser.add_argument('--n_seeds', type=int, nargs='+',
                        help="Seed of instance. If there is only 1 number, reconstruct the specified instance;"
                        "if there are two, reconstruct the range between n_seeds[0] and n_seeds[1].", required=True)
    parser.add_argument(
        '--error', type=str, help="Type of error that used to compute reconstruction error.", required=True)
    parser.add_argument('--ansatz', type=str,
                        help="Type of ansatz.", required=True)

    args = parser.parse_args()

    if args.aim == 'heatmap':
        raise NotImplementedError("Deprecated.")
        gen_heapmap_by_varying_sampling_fraction_and_beta_step(p=args.p, problem=args.problem,
                                                               noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns, error_type=args.error)
    elif args.aim == 'final':
        recon_landscapes_varying_qubits_and_instances(
            p=args.p, ansatz=args.ansatz, problem=args.problem,
            noise=args.noise, n_seeds=args.n_seeds, n_qubits_list=args.ns,
            bs=args.bs, gs=args.gs, error_type=args.error)
    else:
        raise NotImplementedError()
