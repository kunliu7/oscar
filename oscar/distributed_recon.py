import numpy as np
from numpy import ndarray
from typing import Tuple




from sklearn.linear_model import LinearRegression


def T_flatten(a: np.ndarray):
    assert len(a.shape) == 2
    return a.copy().T.flat[:]


def inv_T_flatten(a: np.ndarray, shape: tuple):
    assert len(a.shape) == 1 and len(shape) == 2
    return a.reshape(shape[::-1]).T


def normalize_by_linear_regression(
    ls1: ndarray, ls2: ndarray, n_pts: int, ri: ndarray
) -> Tuple[ndarray, ndarray]:
    """Pick n_pts random points from ri to train a linear model.

    Use the samples from the two landscapes
    located at the same positions in the grid,
    and train a linear model transforming samples from landscape 2 to landscape 1.

    Args:
        ls1 (ndarray): landscape 1
        ls2 (ndarray): landscape 2
        n_pts (int): number of points to train the linear model
        ri (ndarray): random indices

    Returns:
        Tuple[ndarray, ndarray]: landscape 1, normalized landscape 2
    """
    shape = ls1.shape
    print("shape =", shape)
    assert (ls1 == inv_T_flatten(T_flatten(ls1), ls1.shape)).all()
    fls1 = T_flatten(ls1)
    fls2 = T_flatten(ls2)

    rng = np.random.default_rng(7)

    # pick n_pts random points from ri to train NCM
    # note that ri is fixed for normalization and parallel reconstruction
    ids = rng.choice(ri, n_pts, replace=False)
    print(ids.shape)

    y = fls1[ids]
    x = fls2[ids]
    print(x.shape)
    print(y.shape)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model = model.fit(x, y)

    r_sq = model.score(x, y)
    print('coefficient of determination (R^2) :', r_sq)
    print('intercept:', model.intercept_)

    # ! this will be an array when y is also 2-dimensional
    print('slope:', model.coef_)

    ls2_flat = ls2.flatten().reshape(-1, 1)
    y_pred = model.predict(ls2_flat)
    print(y_pred.shape)
    ls2_normalized = y_pred.reshape(shape)

    return ls1, ls2_normalized