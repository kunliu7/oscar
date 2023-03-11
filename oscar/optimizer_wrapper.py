from typing import Callable, List, Optional, Tuple

import numpy as np
from qiskit.algorithms.optimizers import OptimizerResult
from qiskit.algorithms.optimizers.optimizer import POINT

from .interpolate import (approximate_fun_value_by_2D_interpolation,
                          approximate_fun_value_by_2D_interpolation_qiskit)
from .utils import (angles_from_qiskit_format, angles_to_qaoa_format,
                    shift_parameters)



def wrap_qiskit_optimizer_to_landscape_optimizer(QiskitOptimizer):
    """Override `minimize` of existing Qiskit Optimizers. 

    Since we have existing landscape, we do not need to actually execute the circuit simulation.
    Based on the `fun_type` parameter, we can choose the implementation of the `f` function.

    We use the dynamic inheritance (https://stackoverflow.com/a/21060094/13392267) to
    do the override.
    """
    class LandscapeOptimizer(QiskitOptimizer):
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
            bound_lens = np.apply_along_axis(
                lambda bound: bound[1] - bound[0], axis=1, arr=bounds)
            print('bound_lens', bound_lens)
            self.grid_lens = bound_lens / self.landscape_shape  # element-wise
            print("grid_lens", self.grid_lens)

        def minimize(self,
                     fun: Callable[[POINT], float],
                     x0: POINT,
                     jac: Optional[Callable[[POINT], POINT]] = None,
                     bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizerResult:
            """Override existing optimizer's minimize function.

            Notice that the parameter `fun` is not used here.
            We use our own `fun` function based on the `fun_type` parameter.
            """
            if self.fun_type == 'INTERPOLATE':
                res = super().minimize(self.approximate_fun_value, x0, jac, bounds)
            if self.fun_type == 'INTERPOLATE_QISKIT':
                res = super().minimize(self.approximate_fun_value_qiskit, x0, jac, bounds)
            elif self.fun_type == 'FUN':
                res = super().minimize(self._fun, x0, jac, bounds)
            else:
                res = super().minimize(self.query_fun_value_from_landscape, x0, jac, bounds)
            print('res', res)
            print(jac)
            return res

        @staticmethod
        def qiskit_format_to_qaoa_format_arr(x: POINT):
            """Convert the qiskit format to qaoa format.

            These are different formats used by different circuit implementations. 
            See the `angles_from_qiskit_format`
            and `angles_to_qaoa_format` functions for more details.

            """
            angles = angles_from_qiskit_format(x)
            angles = angles_to_qaoa_format(angles)

            x = np.concatenate([angles['gamma'], angles['beta']])
            return x

        @DeprecationWarning
        def approximate_fun_value(self, x: POINT) -> float:
            x = self.qiskit_format_to_qaoa_format_arr(x)
            x = shift_parameters(x, self.bounds)
            val = approximate_fun_value_by_2D_interpolation(
                x=x,
                landscape=self.landscape,
                bounds=self.bounds
            )

            self.params_path.append(x)
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
            self.vals.append(val[0])
            return val[0]

        @DeprecationWarning
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
            # print('transformed x', x)
            # assert x.shape == self.grid_shapes.shape
            # print(type(bounds))
            relative_indices = np.around(
                (x - self.bounds[:, 0]) / self.grid_lens).astype(int)
            normalized_indices = relative_indices.copy()

            for axis, size in enumerate(self.landscape_shape):
                # print(axis, size)
                relative_index = relative_indices[axis]
                if relative_index >= size:
                    normalized_indices[axis] = relative_index % size
                elif relative_index < 0:
                    normalized_indices[axis] = relative_index + \
                        ((-relative_index - 1) // size + 1) * size

                if normalized_indices[axis] < 0 or normalized_indices[axis] >= size:
                    print(axis, size, relative_index,
                          normalized_indices[axis])
                    assert False

            approximate_point = self.landscape[tuple(normalized_indices)]
            self.params_path.append(x)  # Qiskit format
            return approximate_point

    return LandscapeOptimizer
