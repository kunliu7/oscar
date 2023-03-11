#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper functions.

Most of them are copied from [QAOAKit](https://github.com/QAOAKit/QAOAKit).
Some of them are not used in the current version of OSCAR.
"""

import copy
import pynauty
import networkx as nx
import numpy as np
from functools import partial
from qiskit_aer import AerSimulator
import re
import time


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
