from cs_second_optimize import optimize_on_p1_reconstructed_landscape
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ns', type=int, nargs='+', help="Your aims, vis, opt", required=True)
    parser.add_argument('-n', type=int, help="Your aims, vis, opt", required=True)
    parser.add_argument('-p', type=int, help="Your aims, vis, opt", required=True)
    # parser.add_argument('--method', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--noise', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--miti', type=str, help="Your aims, vis, opt", default=None)
    parser.add_argument('--seed', type=int, help="Your aims, vis, opt", required=True)
    parser.add_argument('--lr', type=float, help="Your aims, vis, opt", default=None)
    parser.add_argument('--maxiter', type=int, help="Your aims, vis, opt", default=None)
    parser.add_argument('--init_pt', type=float, nargs="+", help="[beta, gamma]", required=True)
    # parser.add_argument('--error', type=str, help="Your aims, vis, opt", required=True)
    parser.add_argument('--check', action="store_true", help="Your aims, vis, opt", default=False)
    parser.add_argument('--opt', type=str, help="Optimizer Name.", required=True)
    parser.add_argument('--get_vals', help="Get values on the path.", action='store_true', default=False)
    args = parser.parse_args()

    optimize_on_p1_reconstructed_landscape(
        args.n, args.p, args.seed, args.noise, args.miti, args.init_pt,
        args.opt, args.lr, args.maxiter, args.get_vals
    )