import itertools
import time
import networkx as nx
import numpy as np
from qiskit_aer import AerSimulator
from functools import partial
import copy

from qiskit_aer.noise import NoiseModel
import concurrent.futures
from mitiq.zne.zne import execute_with_zne



from .qaoa import get_maxcut_qaoa_circuit
from .utils import (
    get_curr_formatted_timestamp
)
from .noisy_params_optim import (
    compute_expectation
)

# vis
import numpy as np

# qiskit Landscape optimizer



# ================== CS p==1 ==================

def _get_ideal_unmiti_miti_value_for_one_point(
    G,
    beta: list,
    gamma: list,
    shots: int,
    noise_model,
    ignore_miti: bool=False,
    ignore_unmiti: bool=False,
    ignore_ideal: bool=False,
    mitigation_params: dict=None
):
    """Generate ideal, unmitigated and mitigated energy for given one point.
    """
    circuit = get_maxcut_qaoa_circuit(
        G, beta=beta, gamma=gamma,
        # transpile_to_basis=True, save_state=False)
        transpile_to_basis=False, save_state=False)

    print("!")
    if ignore_ideal:
        ideal = 0
    else:
        ideal = _executor_of_qaoa_maxcut_energy(
            circuit.copy(), G, noise_model=None, shots=shots)
    
    if ignore_unmiti:
        unmiti = 0
    else:
        unmiti = _executor_of_qaoa_maxcut_energy(
            circuit.copy(), G, noise_model=copy.deepcopy(noise_model), shots=shots)
    
    if ignore_miti:
        miti = 0
    else:
        if isinstance(mitigation_params, dict):
            factory = mitigation_params['factory']
        else:
            factory = None

        miti = execute_with_zne(
            circuit.copy(),
            executor=partial(
                _executor_of_qaoa_maxcut_energy, G=G, noise_model=noise_model, shots=shots),
            factory=factory
        )
    # print(miti)

    return ideal, unmiti, miti


def _executor_of_qaoa_maxcut_energy(qc, G, noise_model, shots) -> float:
    """Generate mitigated QAOA MaxCut energy. For minimize.
    """
    backend = AerSimulator()
    qc.measure_all()
    # backend = Aer.get_backend('aer_simulator')
    # noise_model = get_depolarizing_error_noise_model(p1Q=0.001, p2Q=0.005) # ! base noise model
    # noise_model = None
    counts = backend.run(
        qc,
        shots=shots,
        # noise_model=noise_model if is_noisy else None
        noise_model=noise_model
    ).result()

    counts = counts.get_counts()
    # print(counts)
    
    expval = compute_expectation(counts, G)
    # print(expval)
    # ! for minimize
    return -expval
    # return expval


def _get_ideal_unmiti_miti_value_for_one_point_wrapper_for_concurrency(param):
    return _get_ideal_unmiti_miti_value_for_one_point(*param)


def gen_p1_landscape(
        G: nx.Graph,
        p: int,
        figdir: str,
        beta_opt: np.array, # converted
        gamma_opt: np.array, # converted
        noise_model: NoiseModel,
        params_path: list,
        C_opt: float,
        mitigation_params: dict=None,
        bounds = {'beta': [-np.pi/4, np.pi/4],
                 'gamma': [-np.pi, np.pi]},
        n_shots: int=2048,
        n_pts_per_unit: int=36
    ):
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # n_shots = 2
    # n_pts_per_unit = 2     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    # bounds = {'beta': [-np.pi/4, np.pi/4],
    #           'gamma': [-np.pi, np.pi]}

    n_pts = {}
    # n_samples = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit * bound_len).astype(int)
        # n_samples[label] = np.ceil(n_pts_per_unit * bound_len * sampling_frac).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    # print('n_samples: ', n_samples)
    # print('alpha: ', alpha)
    print('n_pts_per_unit: ', n_pts_per_unit)
    # sample P points from N randomly

    _LABELS = ['mitis', 'unmitis', 'ideals']
    origin = {label: [] for label in _LABELS}

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    params = []
    for gamma in full_range['gamma']:
        for beta in full_range['beta']:
            param = (
                G.copy(),
                [beta],
                [gamma],
                n_shots,
                copy.deepcopy(noise_model),
                False,
                True,
                True,
                copy.deepcopy(mitigation_params)
            )
            params.append(param)
    
    print('totally num. need to calculate energies:', len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # ! submit will shows exception, while map does not
        # future = executor.submit(
        #     _get_ideal_unmiti_miti_value_for_one_point_p1_wrapper_for_concurrency,
        #     *params[0]
        # )
        futures = executor.map(
            _get_ideal_unmiti_miti_value_for_one_point_wrapper_for_concurrency, params
        )
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {(end_time - start_time) / 3600} h")

    for f in futures:
        # print(f)
        origin['ideals'].append(f[0])
        origin['unmitis'].append(f[1])
        origin['mitis'].append(f[2])

    for label, arr in origin.items():
        origin[label] = np.array(arr).reshape(n_pts['gamma'], n_pts['beta'])
        print(origin[label].shape)

    np.savez_compressed(f"{figdir}/data",
        # landscapes
        origin=origin,

        # parameters
        n_pts=n_pts,
        full_range=full_range,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        C_opt=C_opt)

    print('generated landscapes data is saved')
    return
# ------------------------- generate gradient of landscape -------------------

def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=1):
    """
    We compute the gradient with the numeric differentiation in the parallel way,
    around the point x_center.

    Args:
        x_center (ndarray): point around which we compute the gradient
        f (func): the function of which the gradient is to be computed.
        epsilon (float): the epsilon used in the numeric differentiation.
        max_evals_grouped (int): max evals grouped
    Returns:
        grad: the gradient computed

    """
    forig = f(*((x_center,)))
    grad = []
    ei = np.zeros((len(x_center),), float)
    todos = []
    for k in range(len(x_center)):
        ei[k] = 1.0
        d = epsilon * ei
        todos.append(x_center + d)
        ei[k] = 0.0

    counter = 0
    chunk = []
    chunks = []
    length = len(todos)
    # split all points to chunks, where each chunk has batch_size points
    for i in range(length):
        x = todos[i]
        chunk.append(x)
        counter += 1
        # the last one does not have to reach batch_size
        if counter == max_evals_grouped or i == length - 1:
            chunks.append(chunk)
            chunk = []
            counter = 0

    for chunk in chunks:  # eval the chunks in order
        parallel_parameters = np.concatenate(chunk)
        todos_results = f(parallel_parameters)  # eval the points in a chunk (order preserved)
        if isinstance(todos_results, float):
            grad.append((todos_results - forig) / epsilon)
        else:
            for todor in todos_results:
                grad.append((todor - forig) / epsilon)

    return np.array(grad)


def qaoa_maxcut_energy_2b_wrapped(x, qc, G, is_noisy, shots):
    # x = [beta, gamma]
    n = x.shape[0]
    beta = x[:n]
    gamma = x[n:]
    energy = _executor_of_qaoa_maxcut_energy(qc.copy(), G, is_noisy=is_noisy, shots=shots)
    return energy


def _p1_grad_for_one_point(
    G,
    beta: float,
    gamma: float,
    shots: int
):
    qc = get_maxcut_qaoa_circuit(
        G, beta=[beta], gamma=[gamma],
        transpile_to_basis=True, save_state=False)

    x0 = np.array([beta, gamma])
    fun = partial(qaoa_maxcut_energy_2b_wrapped,
        qc=qc,
        G=G,
        is_noisy=False,
        shots=shots
    )
    grad = gradient_num_diff(x_center=x0, f=fun, epsilon=1e-10)
    return grad


def _p1_grad_for_one_point_mapper(param):
    return _p1_grad_for_one_point(*param)


def p1_generate_grad(
    G: nx.Graph,
    p: int,
    figdir: str,
    beta_opt: np.array, # converted
    gamma_opt: np.array, # converted
    noise_model: NoiseModel,
    params_path: list,
    C_opt: float
):
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # alpha = 0.1
    n_shots = 2048
    n_pts_per_unit = 36     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    bounds = {'beta': [-np.pi/4, np.pi/4],
              'gamma': [-np.pi, np.pi]}

    n_pts = {}
    # n_samples = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit * bound_len).astype(int)
        # n_samples[label] = np.ceil(n_pts_per_unit * bound_len * sampling_frac).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    # print('n_samples: ', n_samples)
    # print('alpha: ', alpha)
    print('n_pts_per_unit: ', n_pts_per_unit)

    _LABELS = ['mitis', 'unmitis', 'ideals']
    origin = {label: [] for label in _LABELS}

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    params = []
    for gamma in full_range['gamma']:
        for beta in full_range['beta']:
            param = (
                G.copy(),
                beta,
                gamma,
                n_shots
            )
            params.append(param)
    
    print(len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # future = executor.submit(
        #     _p1_grad_for_one_point_mapper,
        #     params[0]
        # )
        futures = executor.map(
            _p1_grad_for_one_point_mapper, params
        )
    # print(future.result())
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {end_time - start_time} s")

    grads = []
    for f in futures:
        grads.append(f)

    grads = np.array(grads).reshape(n_pts['gamma'], n_pts['beta'], 2*p)

    # for f in futures:
        # origin['ideals'].append(f[0])
        # origin['unmitis'].append(f[1])
        # origin['mitis'].append(f[2])

    # for label, arr in origin.items():
        # origin[label] = np.array(arr).reshape(n_pts['gamma'], n_pts['beta'])
        # print(origin[label].shape)
        
    np.savez_compressed(f"{figdir}/grad_data",
        # ! gradient
        grads=grads,

        # ! reconstruct
        # origin=origin,
        # recon=recon,
        # mitis=mitis, unmitis=unmitis, ideals=ideals,
        # unmitis_recon=unmitis_recon, mitis_recon=mitis_recon, ideals_recon=ideals_recon,

        # ! parameters
        n_pts=n_pts,
        # n_samples=n_samples, sampling_frac=sampling_frac,
        # perm=perm,
        full_range=full_range,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        # ! n_optima
        # improved_n_optima=improved_n_optima,
        # improved_n_optima_recon=improved_n_optima_recon,
        C_opt=C_opt)

    return


# ================== generate p=2 landscape ==================

def gen_p2_landscape(
    G: nx.Graph,
    p: int,
    figdir: str,
    beta_opt: np.array, # converted
    gamma_opt: np.array, # converted
    noise_model: NoiseModel,
    params_path: list,
    C_opt: float,
    bounds = {'beta': [-np.pi/4, np.pi/4],
                'gamma': [-np.pi, np.pi]},
    n_shots: int=2048,
    n_pts_per_unit: int=36
):
    # ! Convention: First beta, Last gamma
    
    # hyper parameters
    # n_shots = 2
    # n_pts_per_unit = 2     # num. of original points per unit == 4096, i.e. resolution rate = 1 / n
    
    # beta first, gamma later
    # bounds = {'beta': [-np.pi/4, np.pi/4],
    #           'gamma': [-np.pi, np.pi]}

    n_pts = {}
    # n_samples = {}
    for label, bound in bounds.items():
        bound_len = bound[1] - bound[0]
        n_pts[label] = np.floor(n_pts_per_unit * bound_len).astype(int)
        # n_samples[label] = np.ceil(n_pts_per_unit * bound_len * sampling_frac).astype(int)
    
    print('bounds: ', bounds)
    print('n_pts: ', n_pts)
    # print('n_samples: ', n_samples)
    # print('alpha: ', alpha)
    print('n_pts_per_unit: ', n_pts_per_unit)
    # sample P points from N randomly

    _LABELS = ['mitis', 'unmitis', 'ideals']
    origin = {label: [] for label in _LABELS}

    full_range = {
        'gamma': np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_pts['gamma']),
        'beta': np.linspace(bounds['beta'][0], bounds['beta'][1], n_pts['beta'])
    }

    # in order of beta1 beta2 gamma1 gamma2
    full_ranges = []
    for ip in range(p):
        full_ranges.append(full_range['beta'].copy())
    
    for ip in range(p):
        full_ranges.append(full_range['gamma'].copy())

    # full_ranges = [
    #     full_range['beta'].copy(),
    #     full_range['beta'].copy(),
    #     full_range['gamma'].copy(),
    #     full_range['gamma'].copy()
    # ]

    params = []
    # former 2 are betas, latter 2 are gammas
    for beta2_gamma2 in itertools.product(*full_ranges):
        param = (
            G.copy(),
            beta2_gamma2[:p], # beta
            beta2_gamma2[p:], # gamma
            n_shots,
            copy.deepcopy(noise_model),
            True
        )
        params.append(param)
    
    print('totally number of points need to calculate energies:', len(params))

    start_time = time.time()
    print("start time: ", get_curr_formatted_timestamp())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # ! submit will shows exception, while map does not
        # future = executor.submit(
        #     _get_ideal_unmiti_miti_value_for_one_point_p1_wrapper_for_concurrency,
        #     *params[0]
        # )
        futures = executor.map(
            _get_ideal_unmiti_miti_value_for_one_point_wrapper_for_concurrency, params
        )
    print("end time: ", get_curr_formatted_timestamp())
    end_time = time.time()

    print(f"full landscape time usage: {(end_time - start_time) / 3600} h")

    for f in futures:
        # print(f)
        origin['ideals'].append(f[0])
        origin['unmitis'].append(f[1])
        origin['mitis'].append(f[2])

    shape = []
    for ip in range(p):
        shape.append(n_pts['gamma'])
    for ip in range(p):
        shape.append(n_pts['beta'])

    for label, arr in origin.items():
        origin[label] = np.array(arr).reshape(*shape)
        print(origin[label].shape)

    np.savez_compressed(f"{figdir}/data",
        # landscapes
        origin=origin,

        # parameters
        shape=shape,
        n_pts=n_pts,
        full_range=full_range,
        full_ranges=full_ranges,
        bounds=bounds,
        n_shots=n_shots,
        n_pts_per_unit=n_pts_per_unit,

        C_opt=C_opt)

    print('generated landscapes data is saved')
    return