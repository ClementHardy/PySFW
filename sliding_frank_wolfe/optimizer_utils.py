import numpy as np
from numba import jit,prange
from sliding_frank_wolfe.tools import build_Phi
from scipy.optimize import minimize
from scipy.optimize import Bounds








@jit(nopython=True, fastmath=True)
def objectiveFunc(values_parameter,index_parameter, A, parameters, data, times, order_base, normalized, func):
    n, k0 = A.shape
    k = k0 - order_base
    new_parameters = parameters.copy()
    new_parameters[:,index_parameter] = values_parameter
    F = build_Phi(times, new_parameters, k, order_base, normalized, func)
    return np.sum((data - np.dot(F, np.transpose(A))) ** 2)


@jit(nopython=True, parallel=True,fastmath=True)
def JacObjectiveFunc(values_parameter,index_parameter, A,parameters, data, times, order_base, normalized, func, deriv_func):
    new_parameters = parameters.copy()
    new_parameters[:,index_parameter] = values_parameter
    k = len(parameters)
    res = np.zeros(k)
    F = build_Phi(times, new_parameters, k, order_base, normalized, func)
    if normalized == False:
        temp = 2 * (data - np.dot(F, np.transpose(A)))
        for j in range(k):
            res[j] = np.sum(np.outer(-1 * deriv_func(index_parameter, new_parameters[j], times), A[:, j]) * temp )
    else:
        temp = 2 * (data - np.dot(F, np.transpose(A)))
        for ii in range(k):
            deriv_values = deriv_func(index_parameter,new_parameters[ii], times)
            func_values = func(new_parameters[ii], times)
            norm = np.linalg.norm(func_values, ord=2)
            deriv_normalized = deriv_values  / norm - func_values  * np.sum( deriv_values  * func_values ) /norm**3
            res[ii] = np.sum(np.outer(-1 * deriv_normalized, A[:, ii]) * temp )

    return res

def nlls_step_jac_decomp(data, times, A, parameters, order_base, bounds, normalized, func, deriv_func):
    number_parameters = parameters.shape[1]
    new_parameters = parameters.copy()
    for i in range(number_parameters):
        if bounds[i] is None:
            new_res = minimize(lambda x: objectiveFunc(x,i, A, new_parameters, data, times, order_base, normalized, func), parameters[:,i],
                             method='L-BFGS-B',
                             jac=lambda x: JacObjectiveFunc(x,i, A, new_parameters, data, times, order_base, normalized, func,
                                                              deriv_func))
        else:
            new_res = minimize(lambda x: objectiveFunc(x, i,A, new_parameters, data, times, order_base, normalized, func), parameters[:,i],
                             method='L-BFGS-B',
                             jac=lambda x: JacObjectiveFunc(x,i, A, new_parameters, data, times, order_base, normalized, func,
                                                         deriv_func), bounds=Bounds(bounds[i][0], bounds[i][1]))
        new_parameters[:,i] = np.array(new_res.x)
    return new_parameters
