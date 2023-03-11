import imp
from re import L
import re
import time
from turtle import left, right
from typing import Callable, List, Optional, Tuple
import networkx as nx
import numpy as np
import cvxpy as cvx
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit import Aer
import qiskit
from qiskit_aer import AerSimulator
from functools import partial
from pathlib import Path
import copy
import timeit
import sys, os
from scipy.fftpack import dct, diff, idct
from scipy.optimize import minimize
from sklearn import linear_model

from qiskit_aer.noise import NoiseModel
import concurrent.futures
from mitiq import zne, Observable, PauliString
from mitiq.interface.mitiq_qiskit.qiskit_utils import (
    execute,
    execute_with_noise,
    execute_with_shots_and_noise
)

from mitiq.interface import convert_to_mitiq

from qiskit.quantum_info import Statevector
from sympy import beta, per

from .qaoa import get_maxcut_qaoa_circuit
from .utils import (
    angles_to_qaoa_format,
    get_curr_formatted_timestamp,
    noisy_qaoa_maxcut_energy,
    angles_from_qiskit_format,
    maxcut_obj,
    get_adjacency_matrix,
    obj_from_statevector,
    qaoa_maxcut_energy,
    shift_parameters
)
from .noisy_params_optim import (
    compute_expectation,
    get_depolarizing_error_noise_model
)
from .interpolate import (
    approximate_fun_value_by_2D_interpolation,
    approximate_fun_value_by_2D_interpolation_qiskit
)

# vis
import numpy as np
import matplotlib.pyplot as plt
from random import sample

# qiskit Landscape optimizer
from qiskit.algorithms.optimizers import (
    Optimizer, OptimizerResult
)

from qiskit.algorithms.optimizers.optimizer import POINT

def get_numerical_derivative(fun: Optional[Callable[[POINT], POINT]], eps=1e-10):
    jac = Optimizer.wrap_function(Optimizer.gradient_num_diff, (fun, eps))
    return jac

def wrap_qiskit_optimizer_to_landscape_optimizer(QiskitOptimizer):
    class LandscapeOptimizer(QiskitOptimizer):
    
        """Override some methods based on existing Optimizers.
        Since we have existing landscape, we do not need to actually execute the circuit.

        Dynamic inheritance: https://stackoverflow.com/a/21060094/13392267
        """
        def __init__(self, bounds=None, landscape=None, fun_type='FUN', fun=None, **kwargs) -> None:
            # https://blog.csdn.net/sunny_happy08/article/details/82588749
            print("kwargs", kwargs)
            super(LandscapeOptimizer, self).__init__(**kwargs)
            self.fun_type = fun_type
            self.params_path = []
            self.vals = []
            if self.fun_type == 'FUN':
                self.fun = fun
                return
            
            self.bounds = bounds
            self.landscape = landscape
            self.landscape_shape = np.array(landscape.shape)

            # https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
            bound_lens = np.apply_along_axis(lambda bound: bound[1] - bound[0], axis=1, arr=bounds)
            print('bound_lens', bound_lens)
            self.grid_lens = bound_lens / self.landscape_shape # element-wise
            print("grid_lens", self.grid_lens)


        # def minimize(self, 
        #     fun: Callable[[POINT], float], 
        #     x0: POINT, 
        #     jac: Optional[Callable[[POINT], POINT]] = None,
        #     bounds: Optional[List[Tuple[float, float]]] = None):
        #     return super().minimize(fun, x0, jac, bounds)

        # def qiskit_format_to_qaoa_format_arr(self, x: POINT):
        @staticmethod
        def qiskit_format_to_qaoa_format_arr(x: POINT):
            angles = angles_from_qiskit_format(x)
            angles = angles_to_qaoa_format(angles)

            x = np.concatenate([angles['gamma'], angles['beta']])
            return x


        def approximate_fun_value(self, x: POINT) -> float:
            x = self.qiskit_format_to_qaoa_format_arr(x)

            x = shift_parameters(x, self.bounds)
            val = approximate_fun_value_by_2D_interpolation(
                x=x,
                landscape=self.landscape,
                bounds=self.bounds
            )

            self.params_path.append(x)
            print("appro", val)
            self.vals.append(val[0])
            return val[0]
        

        def approximate_fun_value_qiskit(self, x: POINT) -> float:
            """
            Args:
                x (POINT): gamma, beta
            """
            x = np.array(x)
            x = shift_parameters(x, self.bounds)
            val = approximate_fun_value_by_2D_interpolation_qiskit(
                x=x,
                landscape=self.landscape,
                bounds=self.bounds
            )

            self.params_path.append(x)
            # print("appro", val)
            self.vals.append(val[0])
            return val[0]

        def _fun(self, x: POINT) -> float:
            x = self.qiskit_format_to_qaoa_format_arr(x)
            # angles = angles_from_qiskit_format(x)
            # angles = angles_to_qaoa_format(angles)

            val = self.fun(x)
            self.params_path.append(x)
            return val


        def query_fun_value_from_landscape(self, x: POINT) -> float:        
            angles = angles_from_qiskit_format(x)
            angles = angles_to_qaoa_format(angles)

            x = np.concatenate([angles['gamma'], angles['beta']])
            print('transformed x', x)
            # assert x.shape == self.grid_shapes.shape
            # print(type(bounds))
            relative_indices = np.around((x - self.bounds[:, 0]) / self.grid_lens).astype(int)
            normalized_indices = relative_indices.copy()

            for axis, size in enumerate(self.landscape_shape):
                # print(axis, size)
                relative_index = relative_indices[axis]
                if relative_index >= size:
                    normalized_indices[axis] = relative_index % size
                elif relative_index < 0:
                    normalized_indices[axis] = relative_index + ((-relative_index - 1) // size + 1) * size

                if normalized_indices[axis] < 0 or normalized_indices[axis] >= size:
                    print(axis, size, relative_index, normalized_indices[axis])
                    assert False

            # print(normalized_indices)
            # print(self.landscape_shape)
            # obj = *(list(normalized_indices))
            approximate_point = self.landscape[tuple(normalized_indices)]

            # print(approximate_point)
            self.params_path.append(x) # Qiskit format
            return approximate_point


        def minimize(self, 
            fun: Callable[[POINT], float], 
            x0: POINT, 
            jac: Optional[Callable[[POINT], POINT]] = None,
            bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizerResult:
            """Override existing optimizer's minimize function

            """
            def query_fun_value_from_landscape(x: POINT) -> float:
                angles = angles_from_qiskit_format(x)
                angles = angles_to_qaoa_format(angles)

                x = np.concatenate([angles['gamma'], angles['beta']])
                # print('transformed x', x)
                # assert x.shape == self.grid_shapes.shape
                # print(type(bounds))
                relative_indices = np.around((x - self.bounds[:, 0]) / self.grid_lens).astype(int)
                normalized_indices = relative_indices.copy()

                for axis, size in enumerate(self.landscape_shape):
                    # print(axis, size)
                    relative_index = relative_indices[axis]
                    if relative_index >= size:
                        normalized_indices[axis] = relative_index % size
                    elif relative_index < 0:
                        normalized_indices[axis] = relative_index + ((-relative_index - 1) // size + 1) * size

                    if normalized_indices[axis] < 0 or normalized_indices[axis] >= size:
                        print(axis, size, relative_index, normalized_indices[axis])
                        assert False

                # print(normalized_indices)
                # print(self.landscape_shape)
                # obj = *(list(normalized_indices))
                approximate_point = self.landscape[tuple(normalized_indices)]

                # print(approximate_point)
                self.params_path.append(x) # Qiskit format
                return approximate_point

            
            if self.fun_type == 'INTERPOLATE':
                res = super().minimize(self.approximate_fun_value, x0, jac, bounds)
            if self.fun_type == 'INTERPOLATE_QISKIT':
                res = super().minimize(self.approximate_fun_value_qiskit, x0, jac, bounds)
            elif self.fun_type == 'FUN':
                res = super().minimize(self._fun, x0, jac, bounds)
            else:
                # TODO replace query_fun_value_from_landscape by self.query_fun_value_from_landscape, and test
                res = super().minimize(query_fun_value_from_landscape, x0, jac, bounds)
            print('res', res)
            print(jac)
            return res



    return LandscapeOptimizer

