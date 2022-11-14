import argparse
import os
from functools import partial

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.algorithms import VQE, QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit.utils import QuantumInstance
from qiskit.opflow import AerPauliExpectation, PauliExpectation, StateFn, CircuitStateFn
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut, SKModel
from qiskit_optimization.translators import from_docplex_mp
from qiskit.utils import algorithm_globals
from docplex.mp.model import Model
from mitiq.zne import (
    execute_with_zne,
    LinearFactory,
    RichardsonFactory,
    PolyFactory,
    ExpFactory,
    AdaExpFactory,
)

from benchmark import Benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str, default="maxcut")
parser.add_argument("-n", type=int, default=16)
parser.add_argument("-p", type=int, default=1)
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-b", "--backend", type=str, default="sv")
parser.add_argument("--cpu", default=False, action="store_true")
parser.add_argument("--no-aer", dest="aer", default=True, action="store_false")
parser.add_argument("--noise", type=str, default="ideal")
parser.add_argument("--p1", type=float, default=0.001)
parser.add_argument("--p2", type=float, default=0.005)
parser.add_argument("--beta-steps", type=int, default=50)
parser.add_argument("--gamma-steps", type=int, default=100)
parser.add_argument("--mitiq", type=str, default=None)
parser.add_argument("--mitiq-config", type=int, default=0)

args = parser.parse_args()

n = args.n
p = args.p
seed = args.seed
backend_config = args.backend.lower()
noise = args.noise.lower()

if backend_config == "sv":
    method = "statevector"
else:
    raise NotImplementedError(f"Backend {backend_config} not implemented yet")

device = "CPU" if args.cpu else "GPU"
backend_config += f" {noise}"

if noise == "ideal":
    noise_model = None
elif noise == "depolar":
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(args.p1, 1), ['id', 'rz', 'sx', 'u2', 'u3']
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(args.p2, 2), ['cx']
    )
    backend_config += f" {args.p1} {args.p2}"
else:
    raise NotImplementedError(f"Noise model {args.noise} not implemented yet")

backend = AerSimulator(
    method=method,
    device=device,
    noise_model=noise_model,
    fusion_enable=args.problem == "maxcut" or noise != "depolar" or n < 17,
    # blocking_enable=True,
    # blocking_qubits=30,
)
backend_config = backend_config.replace(" ", "-")

if args.problem == "maxcut":
    graph = nx.random_regular_graph(3, n, seed)
    problem = Maxcut(graph).to_quadratic_program()
elif args.problem == "skmodel":
    problem = SKModel(n, seed).to_quadratic_program()
elif args.problem == "partition":
    number_set = np.random.default_rng(seed).integers(2 * n, size=n) + 1
    mdl = Model(name="Number partitioning")
    x = {i: mdl.binary_var(name=f"x_{i}") for i in range(n)}
    mdl.minimize(
        mdl.sum(num * (-2 * x[i] + 1) for i, num in enumerate(number_set)) ** 2
    )
    problem = from_docplex_mp(mdl)
else:
    raise NotImplementedError(f"{args.problem} problem is not implemented")

H, offset = problem.to_ising()

algorithm_globals.random_seed = seed
quantum_instance = QuantumInstance(
    backend=backend,
    seed_simulator=seed,
    seed_transpiler=seed,
    optimization_level=0 if noise != "ideal" else None,
)

algorithm = QAOA(
    SPSA(),
    reps=p,
    quantum_instance=quantum_instance,
    expectation=AerPauliExpectation() if args.aer else None,
)
algorithm._check_operator_ansatz(H)
energy_evaluation, expectation = algorithm.get_energy_evaluation(
    H, return_expectation=True
)

mitiq_config = None
if args.mitiq:
    if args.mitiq.lower() == "zne":
        factory_configs = [
            RichardsonFactory(scale_factors=[1, 2, 3]),
            LinearFactory(scale_factors=[1, 3]),
            PolyFactory(scale_factors=[1, 1.5, 2, 2.5, 3], order=2),
            ExpFactory(scale_factors=[1, 2, 3], asymptote=0.5),
            AdaExpFactory(steps=5, asymptote=0.5),
        ]
        mitiq_config = factory_configs[args.mitiq_config]
        execute_with_mitiq = partial(execute_with_zne, factory=mitiq_config)
    else:
        raise NotImplementedError(f"{args.mitiq} mitigation is not implemented yet")

    def energy_evaluation_with_mitiq(parameters: np.ndarray) -> float:
        qc = quantum_instance.transpile(algorithm.ansatz.bind_parameters(parameters))[0]

        def executor(circuit: QuantumCircuit) -> float:
            observable_meas = expectation.convert(StateFn(H, is_measurement=True))
            expect_op = observable_meas.compose(CircuitStateFn(circuit)).reduce()
            sampled_expect_op = algorithm._circuit_sampler.convert(expect_op)
            return np.real(sampled_expect_op.eval())

        return execute_with_mitiq(qc, executor)

    energy_evaluation = energy_evaluation_with_mitiq

data, time = [], []
beta_bound = np.pi / 4 / p
gamma_bound = np.pi / 2 / p

grid = np.array(
    np.meshgrid(
        *np.linspace([-beta_bound] * p, [beta_bound] * p, args.beta_steps, axis=1),
        *np.linspace([-gamma_bound] * p, [gamma_bound] * p, args.gamma_steps, axis=1),
        indexing="ij",
    )
)
for params in grid.transpose((*range(1, 2 * p + 1), 0)).reshape(-1, 2 * p):
    benchmark = Benchmark(
        # f"{backend_config}-{n=}-{p=}-{seed=}-{args.beta_steps}-{args.gamma_steps}",
        profile_time=False,
        profile_memory=False,
        printout=False,
        include_git_hash=False,
    )
    with benchmark:
        data.append(energy_evaluation(params))
    time.append(benchmark.data["time"])

data = np.array(data).reshape(([args.beta_steps] * p) + ([args.gamma_steps] * p))
time = np.array(time).reshape(([args.beta_steps] * p) + ([args.gamma_steps] * p))
print(
    f"Total time for {backend_config} {n=} {p=} {seed=} {args.beta_steps}x{args.gamma_steps}: ",
    np.sum(time),
)

# optimization_result = MinimumEigenOptimizer(algorithm).solve(problem)

dirpath = f"data/grid_search/{args.problem}/{backend_config}-{p=}"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

savepath = f"{dirpath}/{args.problem}-{backend_config}-{n=}-{p=}-{seed=}-{args.beta_steps}-{args.gamma_steps}"
if args.mitiq:
    savepath += f"-{args.mitiq.lower()}-{mitiq_config.__class__.__name__}"

np.savez_compressed(
    savepath,
    data=data,
    time=time,
    offset=offset,
    beta_bound=beta_bound,
    gamma_bound=gamma_bound,
    grid=grid,
    # optimization_result=optimization_result,
    mitigation_method=args.mitiq,
    mitigation_config=mitiq_config.__class__.__name__,
    **args.__dict__,
)
