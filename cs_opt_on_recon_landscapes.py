from cs_second_optimize import optimize_on_p1_reconstructed_landscape
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help="#qubits", required=True)
    parser.add_argument('-p', type=int, help="#layers", required=True)
    parser.add_argument('--noise', type=str,
                        help="noise config", required=True)
    parser.add_argument('--miti', type=str,
                        help="mitigation config", default=None)
    parser.add_argument('--seed', type=int,
                        help="seed of instance", required=True)
    parser.add_argument('--lr', type=float, help="learning rate", default=None)
    parser.add_argument('--maxiter', type=int,
                        help="max iteration of optimization", default=None)
    parser.add_argument('--init_pt', type=float, nargs="+",
                        help="[beta, gamma]", required=True)
    # parser.add_argument('--error', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--check', action="store_true",
                        help="check code", default=False)
    parser.add_argument('--opt', type=str,
                        help="optimizer Name.", required=True)
    parser.add_argument('--get_vals', help="get values corresponding to parameters on the path",
                        action='store_true', default=False)
    args = parser.parse_args()

    optimize_on_p1_reconstructed_landscape(
        args.n, args.p, args.seed, args.noise, args.miti, args.init_pt,
        args.opt, args.lr, args.maxiter, args.get_vals
    )
