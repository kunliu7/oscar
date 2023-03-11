from typing import Callable, List, Optional, Tuple

import numpy as np
from qiskit.algorithms.optimizers import OptimizerResult
import qiskit.algorithms.optimizers.optimizer as qiskit_optimizer
from qiskit.algorithms.optimizers.optimizer import POINT, Optimizer

from .interpolate import (approximate_fun_value_by_2D_interpolation,
                          approximate_fun_value_by_2D_interpolation_qiskit)
from .utils import (angles_from_qiskit_format, angles_to_qaoa_format,
                    shift_parameters)


def wrap_qiskit_optimizer_to_landscape_optimizer(QiskitOptimizer) -> Optimizer:
    """Override `minimize` of existing Qiskit Optimizers. 

    Since we have existing landscape, we do not need to actually execute the circuit simulation.
    Based on the `fun_type` parameter, we can choose the implementation of the `f` function.

    We use the dynamic inheritance (https://stackoverflow.com/a/21060094/13392267) to
    do the override.

    Examples:
        >>> from qiskit.algorithms.optimizers import COBYLA
        >>> optimizer = wrap_qiskit_optimizer_to_landscape_optimizer(COBYLA)

    Args:
        QiskitOptimizer (Optimizer): Qiskit Optimizer class

    Returns:
        LandscapeOptimizer (Optimizer): Optimizer class with overridden `minimize` function.
    """
    class LandscapeOptimizer(QiskitOptimizer):
        def __init__(self, bounds=None, landscape=None, fun_type='INTERPOLATE_QISKIT', fun=None, **kwargs) -> None:
            print("Landscape Optimizer's kwargs", kwargs)
            assert fun_type in [
                'INTERPOLATE_QISKIT'], f'fun_type {fun_type} is not implemented.'

            # Init the parent class of LandscapeOptimizer, which is a QiskitOptimizer
            super(LandscapeOptimizer, self).__init__(**kwargs)

            self.fun_type = fun_type
            self.params_path = []
            self.vals = []
            self.bounds = bounds
            self.landscape = landscape
            self.landscape_shape = np.array(landscape.shape)

            # https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
            bound_lens = np.apply_along_axis(
                lambda bound: bound[1] - bound[0], axis=1, arr=bounds)
            self.grid_lens = bound_lens / self.landscape_shape  # element-wise

            print('bound_lens', bound_lens)
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
            if self.fun_type == 'INTERPOLATE_QISKIT':
                res = super().minimize(self.approximate_fun_value_qiskit, x0, jac, bounds)
            else:
                raise NotImplementedError(
                    f'fun_type {self.fun_type} is not implemented.')

            return res

        def approximate_fun_value_qiskit(self, x: POINT) -> float:
            """
            Args:
                x (POINT): gammas, betas
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

    return LandscapeOptimizer
