
import numpy as np
from numba import jit,prange
from scipy.optimize import minimize
from scipy.optimize import Bounds



@jit(nopython=True, fastmath=True)
def objective_eta(x, Y, times, normalized, func):
    values = func(x, times)
    if normalized:
        values = values / np.linalg.norm(values, ord=2)
    return -1 * np.sum(np.dot(values, Y)**2)

@jit(nopython=True, fastmath=True)
def jac_eta(x,number_parameters, Y, times, normalized, func, deriv_func):
    res = np.zeros(number_parameters)
    v = func(x, times)
    v_norm = np.linalg.norm(v, ord=2)
    for index_parameter in range(number_parameters):
        if normalized:
            deriv_values = deriv_func(index_parameter, x, times)
            dv = deriv_values/ v_norm - v * np.sum(
                 deriv_values* v) / v_norm**3
            res[index_parameter] = -2 * np.sum(np.dot(dv, Y) *  np.dot(v/v_norm, Y))
        else:
            dv = deriv_func(index_parameter,x, times)
            res[index_parameter] = -2 * np.sum(np.dot(dv, Y) * np.dot(v, Y))
    return res


def locate_new_spike(data, times, normalized, size_grids, lower_bounds, upper_bounds, func, deriv_func):
    bounds = Bounds(lower_bounds, upper_bounds)
    value = np.inf
    number_parameters = len(lower_bounds)
    grids =[]
    for ii in range(number_parameters):
        grids.append(np.linspace(lower_bounds[ii],upper_bounds[ii],size_grids[ii]))
    meshgrid = np.meshgrid(*grids)
    flatten_meshgrid = []
    for jj in range(len(meshgrid)):
        flatten_meshgrid.append(meshgrid[jj].flatten())
    for jjj in range(len(flatten_meshgrid[0])):
        x0 = []
        for iii in range(number_parameters):
            x0.append(flatten_meshgrid[iii][jjj])
        res = minimize(lambda x: objective_eta(x, data, times, normalized, func), x0, method='L-BFGS-B',
                       jac=lambda x: jac_eta(x,number_parameters, data, times, normalized, func, deriv_func),
                       bounds=bounds)
        if value > res.fun:
            sol = res.x
            value = res.fun
    return sol

