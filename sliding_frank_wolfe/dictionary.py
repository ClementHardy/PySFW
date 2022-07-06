
from numba import jit
import numpy as np





class dictionary:
    def __init__(self,function,derivatives, jacobian = None):
        func = func
        derivatives = derivatives
        if jacobian != None:
            jacobian = jacobian


@jit(nopython=True)
def cauchy(parameters, x):
    b = parameters[0]
    c = parameters[1]
    return 1. / (1 + ((x - b) ** 2 / c ** 2))

@jit(nopython=True)
def derivCauchyB(parameters, x):
    b = parameters[0]
    c = parameters[1]
    return -2 * (b - x) / (c ** 2) * np.power(1 + ((b - x) / c) ** 2, -2)

@jit(nopython=True)
def derivCauchyC(parameters, x):
    b = parameters[0]
    c = parameters[1]
    return 2 * (((x - b) ** 2) / (c ** 3)) * np.power(1 + ((b - x) / c) ** 2, -2)

@jit(nopython=True)
def expo(parameters, x):
    b = parameters[0]
    c = parameters[1]
    return np.exp(-1 * ((x - b) ** 2) / c ** 2)

@jit(nopython=True)
def derivExpoB(parameters, x):
    b = parameters[0]
    c = parameters[1]
    return 2 * (x - b) / (c ** 2) * np.exp(-1 * ((x - b) ** 2) / (c**2))


@jit(nopython=True)
def derivExpoC(parameters, x):
    b = parameters[0]
    c = parameters[1]
    return 2 *(((x - b)**2) / (c**3)) * np.exp(-1 * ((x - b)**2) / (c**2))

@jit(nopython=True)
def derivExpo(index_parameter, parameters, x):
    if index_parameter == 0:
        res = derivExpoB(parameters, x)
    if index_parameter == 1:
        res = derivExpoC(parameters, x)
    return res

@jit(nopython=True)
def derivCauchy(index_parameter, parameters, x):
    if index_parameter == 0:
        res = derivCauchyB(parameters, x)
    if index_parameter == 1:
        res = derivCauchyC(parameters, x)
    return res



