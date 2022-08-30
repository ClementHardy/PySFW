
import numpy as np
from numba import jit,prange
from scipy.optimize import minimize
from scipy.optimize import Bounds



@jit(nopython=True, fastmath=True)
def objective_eta(x, Y, times, normalized, func):
    '''
    This function computes the objective function of the subproblem that adds a new peak at each SFW iteration.

    Parameters
    ----------
    x : array, shape(d,)
        parameters of the parametric function.
    Y : array, shape(p,n)
        array of n signals distretized on p points.
    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.
    normalized : bool,
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.
    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    Returns
    -------
    float,
        the value of the objective function
    '''
    values = func(x, times)
    if normalized:
        values = values / np.linalg.norm(values, ord=2)
    return -1 * np.sum(np.dot(values, Y)**2)

@jit(nopython=True, fastmath=True)
def jac_eta(x,number_parameters, Y, times, normalized, func, deriv_func):
    '''
    This function computes the Jacobian of the objective function of the subproblem that adds a new peak at each SFW
    iteration.

    Parameters
    ----------
    x : array, shape(d,)
        parameters of the parametric function.
    number_parameters : int,
        must be equal to d.
    Y : array, shape(p,n)
        array of n signals discretized on p points.
    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.
    normalized : bool,
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.
    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    deriv_func : callable
        derivative of the parametric function "func" with respect to the parameter of index "index_parameter".
        deriv_func(index_parameter, parameters, x) -> float, where "x" is either a float or an array of float,
        "parameters" is an array of shape (d,).

    Returns
    -------
    res : array, shape(d,)
        Jacobian of the objective function.
    '''
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
    '''
    This function finds the solution of the subproblem that adds a new peak at each SFW iteration.

    Parameters
    ----------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    normalized : bool,
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.

    size_grids : array, shape (d,)
        The k-th coordinate of "size_grids" corresponds to the size of the initialization grid for the k-th parameter
        used to locate a new spike at each SFW iteration.

    lower_bounds : array, shape(d,)
        lower_bounds on the parameters of the parametric functions. The k-th coordinate of the array "lower_bounds"
        corresponds to the lower_bound on the k-th dimension of the parameter.

    upper_bounds : array, shape(d,)
        upper_bounds on the parameters of the parametric functions. The k-th coordinate of the array "upper_bounds"
        corresponds to the upper_bound on the k-th dimension of the parameter.

    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    deriv_func : callable
        derivative of the parametric function "func" with respect to the parameter of index "index_parameter".
        deriv_func(index_parameter, parameters, x) -> float, where "x" is either a float or an array of float,
        "parameters" is an array of shape (d,).

    Returns
    -------
    sol : array, shape(d,)
        solution of the subproblem
    '''
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

