import copy
import pynauty
import networkx as nx
import numpy as np
from functools import partial
from qiskit_aer import AerSimulator
import re
import time
from oscar.qaoa import get_maxcut_qaoa_circuit


def get_adjacency_dict(G):
    """Returns adjacency dictionary for G
    G must be a networkx graph
    Return format: { n : [n1,n2,...], ... }
    where [n1,n2,...] is a list of neighbors of n
    ignores all attributes
    """
    adjacency_dict = {}
    for n, neigh_dict in G.adjacency():
        neigh_list = []
        for neigh, attr_dict in neigh_dict.items():
            neigh_list.append(neigh)
        adjacency_dict[n] = neigh_list
    return adjacency_dict


def get_pynauty_certificate(G):
    """Get pynauty certificate for G

    Parameters
    ----------
    G : networkx.Graph
        Unweighted graph to compute certificate for

    Returns
    -------
    cert : binary
        isomorphism certificate for G
    """
    g = pynauty.Graph(
        number_of_vertices=G.number_of_nodes(),
        directed=nx.is_directed(G),
        adjacency_dict=get_adjacency_dict(G),
    )
    return pynauty.certificate(g)


def isomorphic(G1, G2):
    """Tests if two unweighted graphs are isomorphic using pynauty
    Ignores all attributes
    """
    g1 = pynauty.Graph(
        number_of_vertices=G1.number_of_nodes(),
        directed=nx.is_directed(G1),
        adjacency_dict=get_adjacency_dict(G1),
    )
    g2 = pynauty.Graph(
        number_of_vertices=G2.number_of_nodes(),
        directed=nx.is_directed(G2),
        adjacency_dict=get_adjacency_dict(G2),
    )
    return pynauty.isomorphic(g1, g2)


def angles_to_qaoa_format(angles):
    """Converts from format in graph2angles
    into the format used by qaoa.py
    get_maxcut_qaoa_circuit(G, angles['beta'], angles['gamma'])
    """
    res = copy.deepcopy(angles)
    res["beta"] = beta_to_qaoa_format(res["beta"])
    res["gamma"] = gamma_to_qaoa_format(res["gamma"])
    return res


def angles_from_qaoa_format(gamma, beta):
    res = {}
    res["beta"] = beta / np.pi
    res["gamma"] = gamma / (-np.pi/2)
    return res


def beta_to_qaoa_format(beta):
    """Converts from format in graph2angles
    into the format used by qaoa.py
    """
    return np.pi * np.array(beta)


def gamma_to_qaoa_format(gamma):
    """Converts from format in graph2angles
    into the format used by qaoa.py
    get_maxcut_qaoa_circuit(G, angles['beta'], angles['gamma'])
    """
    return -np.pi * np.array(gamma) / 2


def angles_to_qiskit_format(angles):
    """Converts from format in graph2angles
    into the format used by QAOAAnsatz
    """
    return np.concatenate(
        [[-np.pi * g, np.pi * b]
            for g, b in zip(angles["gamma"], angles["beta"])]
    )


def angles_from_qiskit_format(angles):
    """Converts from the format used by QAOAAnsatz
    into the format in graph2angles
    """
    res = {}
    assert len(angles) % 2 == 0
    res["gamma"] = list(x / (-np.pi) for x in angles[::2])
    res["beta"] = list(x / np.pi for x in angles[1::2])
    return res


def qaoa_format_to_qiskit_format(gamma, beta) -> np.ndarray:
    return np.concatenate(
        [[2*g, b] for g, b in zip(gamma, beta)]
    )


def qiskit_format_to_qaoa_format_arr(arr: np.ndarray) -> np.ndarray:
    """Converts from format used by Qiskit's QAOAAnsatz
    to QAOA format used by qaoa.py, in an array where
    [:p] are gammas and [p:] are betas.

    Args:
        arr (np.array): Qiskit format

    Returns:
        np.ndarray: [:p] are gammas and [p:] are betas
    """
    angles = angles_from_qiskit_format(arr)
    angles = angles_to_qaoa_format(angles)

    arr = np.concatenate([angles['gamma'], angles['beta']])
    return arr


def angles_to_qtensor_format(angles):
    """Converts from format in graph2angles
    into the format used by QTensor
    """
    return {"gamma": [-g / 2 for g in angles["gamma"]], "beta": angles["beta"]}


def read_graph_from_file(f, expected_nnodes=None):
    """Read a graph in a format used by qaoa-dataset-version1

    Parameters
    ----------
    f : file-like object
        Handler for the file to read from
    expected_nnodes : int, default None
        Number of nodes to expect
        If passed, a check will be performed
        to confirm that the actual number of nodes
        matches the expectation

    Returns
    -------
    G : networkx.Graph
    graph_id : int
        ID of the graph
    """
    f.readline(-1)  # first line is blank
    line_with_id = f.readline(-1)  # second line has graph number and order
    graph_id, graph_order = [
        int(x) for x in re.split(" |, |. |.\n", line_with_id) if x.isdigit()
    ]
    if expected_nnodes is not None:
        assert graph_order == expected_nnodes
    G = nx.Graph()
    for n in range(graph_order):
        G.add_nodes_from([n])
    edge_id = 0
    # third line is first row of upper triangle of adjacency matrix (without the diagonal element)
    for n in range(graph_order - 1):
        adj_str = f.readline(-1)
        for m in range(graph_order - 1 - n):
            q_num = n + m + 1
            if adj_str[m] == "1":
                G.add_edge(n, q_num, edge_id=edge_id)
                edge_id += 1
    return G, graph_id


def brute_force(obj_f, num_variables, minimize=False):
    """Get the maximum of a function by complete enumeration
    Returns the maximum value and the extremizing bit string
    """
    if minimize:
        best_cost_brute = float("inf")
        def compare(x, y): return x < y
    else:
        best_cost_brute = float("-inf")
        def compare(x, y): return x > y
    bit_strings = (
        (
            (
                np.array(range(2 ** num_variables))[:, None]
                & (1 << np.arange(num_variables))
            )
        )
        > 0
    ).astype(int)
    for x in bit_strings:
        cost = obj_f(np.array(x))
        if compare(cost, best_cost_brute):
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute


#############################
# QAOA utils
############################


def state_num2str(basis_state_as_num, nqubits):
    return "{0:b}".format(basis_state_as_num).zfill(nqubits)


def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)


def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2 ** nqubits, dtype=complex)
    for basis_state in range(2 ** nqubits):
        adjusted_state[state_reverse(
            basis_state, nqubits)] = state[basis_state]
    return adjusted_state


def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real ** 2 + val.imag ** 2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def precompute_energies(obj_f, nbits):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector
    """
    bit_strings = (
        ((np.array(range(2 ** nbits))[:, None] & (1 << np.arange(nbits)))) > 0
    ).astype(int)

    return np.array([obj_f(x) for x in bit_strings])


def obj_from_statevector(sv, obj_f, precomputed_energies=None):
    """Compute objective from Qiskit statevector
    For large number of qubits, this is slow.
    """
    if precomputed_energies is None:
        qubit_dims = np.log2(sv.shape[0])
        if qubit_dims % 1:
            raise ValueError(
                "Input vector is not a valid statevector for qubits.")
        qubit_dims = int(qubit_dims)
        # get bit strings for each element of the state vector
        # https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        bit_strings = (
            ((np.array(range(sv.shape[0]))[:, None]
             & (1 << np.arange(qubit_dims)))) > 0
        ).astype(int)

        return sum(
            obj_f(bit_strings[kk]) * (np.abs(sv[kk]) ** 2) for kk in range(sv.shape[0])
        )
    else:
        amplitudes = np.array(
            [np.abs(sv[kk]) ** 2 for kk in range(sv.shape[0])])
        return precomputed_energies.dot(amplitudes)


def maxcut_obj(x, w):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return np.sum(w * X)


def get_adjacency_matrix(G):
    n = G.number_of_nodes()
    w = np.zeros([n, n])

    for e in G.edges():
        if nx.is_weighted(G):
            w[e[0], e[1]] = G[e[0]][e[1]]["weight"]
            w[e[1], e[0]] = G[e[0]][e[1]]["weight"]
        else:
            w[e[0], e[1]] = 1
            w[e[1], e[0]] = 1
    return w


def qaoa_maxcut_energy(G, beta, gamma, precomputed_energies=None):
    """Computes MaxCut QAOA energy for graph G
    qaoa format (`angles_to_qaoa_format`) used for beta, gamma
    """
    if precomputed_energies is None:
        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    else:
        obj = None
    qc = get_maxcut_qaoa_circuit(G, beta, gamma)
    backend = AerSimulator(method="statevector")
    sv = backend.run(qc).result().get_statevector()
    return obj_from_statevector(sv, obj, precomputed_energies=precomputed_energies)


def noisy_qaoa_maxcut_energy(G, beta, gamma, precomputed_energies=None, noise_model=None):
    """Computes MaxCut QAOA energy for graph G
    qaoa format (`angles_to_qaoa_format`) used for beta, gamma
    """
    if precomputed_energies is None:
        obj = partial(maxcut_obj, w=get_adjacency_matrix(G))
    else:
        obj = None
    qc = get_maxcut_qaoa_circuit(G, beta, gamma)
    backend = AerSimulator(method="statevector")
    sv = backend.run(
        qc,
        noise_model=noise_model
    ).result().get_statevector()
    return obj_from_statevector(sv, obj, precomputed_energies=precomputed_energies)


def beta_shift_sector(beta: np.ndarray, left: float = -np.pi/4, period: float = np.pi/2):
    shifted = shift_sector(beta, left, period)
    return shifted


def gamma_shift_sector(gamma: np.ndarray, left: float = -np.pi, period: float = np.pi):
    shifted = shift_sector(gamma, left, period)
    return shifted


def shift_sector(values: np.ndarray, left: float, period: float):
    right = left + period
    shifted = values.copy()
    for i, b in enumerate(values):
        if b > right:
            while b > right:
                b -= period
        elif b < left:
            while b < left:
                b += period

        shifted[i] = b
    return shifted


def shift_parameters(x: np.ndarray, bounds: np.ndarray):
    assert x.shape[0] == bounds.shape[0]

    _x = x - bounds[:, 0]
    shifted = _x.copy()
    for axis, bound in enumerate(bounds):
        bound_len = bound[1] - bound[0]

        relative = _x[axis]
        # assert relative >= 0 and relative < bound_len
        if relative >= bound_len:
            shifted[axis] = relative % bound_len
        elif relative < 0:
            shifted[axis] = relative + \
                ((-relative) // bound_len + 1) * bound_len
            if np.isclose(shifted[axis], bound_len):
                shifted[axis] = 0.0

        if shifted[axis] < 0 or shifted[axis] >= bound_len:
            print(axis, bound_len, relative, shifted[axis])
            assert False

    shifted = shifted + bounds[:, 0]
    return shifted


def get_curr_formatted_timestamp(format: str = None):
    s = ''
    if format == None:
        s = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    else:
        s = time.strftime(format, time.localtime())

    return s


def arraylike_to_str(a):
    return ",".join(list(map(str, a)))
