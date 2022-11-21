
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
from scipy.fftpack import dct, idct
from scipy.optimize import minimize
from sklearn import linear_model
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import griddata, interp2d, interpn, RectBivariateSpline


def approximate_fun_value_by_2D_interpolation(
    x: np.ndarray, landscape: np.ndarray, bounds: np.ndarray):
    # ! x: [gamma, beta]
    assert x.shape == (2, )
    assert bounds.shape == (2, 2)

    shape = landscape.shape
    gamma = np.linspace(bounds[0][0], bounds[0][1], shape[0])
    beta = np.linspace(bounds[1][0], bounds[1][1], shape[1])
    
    # print(shape)

    # ! indexing = 'xy', https://zhuanlan.zhihu.com/p/33579211
    # f = interp2d(x=beta, y=gamma, z=landscape, kind='cubic') # ! landscape.shape = (len(gamma), len(beta)) = (y-axis, x-axis)
    # guess = f(x=x[1], y=x[0]) # ! f(beta, gamma)
    
    f = RectBivariateSpline(x=gamma, y=beta, z=landscape)
    guess = f(x=x[0], y=x[1]) # ! f(beta, gamma)

    return guess[0]


def approximate_fun_value_by_2D_interpolation_qiskit(
    x: np.ndarray, landscape: np.ndarray, bounds: np.ndarray):
    # ! x: [beta, gamma]
    assert x.shape == (2, )
    assert bounds.shape == (2, 2)

    shape = landscape.shape
    beta = np.linspace(bounds[0][0], bounds[0][1], shape[0])
    gamma = np.linspace(bounds[1][0], bounds[1][1], shape[1])
    
    # print(shape)

    # ! indexing = 'xy', https://zhuanlan.zhihu.com/p/33579211
    # f = interp2d(x=beta, y=gamma, z=landscape, kind='cubic') # ! landscape.shape = (len(gamma), len(beta)) = (y-axis, x-axis)
    # guess = f(x=x[1], y=x[0]) # ! f(beta, gamma)
    
    f = RectBivariateSpline(x=beta, y=gamma, z=landscape)
    guess = f(x=x[0], y=x[1]) # ! f(beta, gamma)

    return guess[0]