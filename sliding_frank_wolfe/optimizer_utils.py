import numpy as np
from numba import jit,prange
from sliding_frank_wolfe.tools import build_Phi
from scipy.optimize import minimize
from scipy.optimize import Bounds








@jit(nopython=True, fastmath=True)
def objectiveFunc(values_parameter,index_parameter, A, parameters, data, times, order_base, normalized, func):
    '''
    objective function of the nonlinear subproblem appearing in the SFW procedure.

    Parameters
    ----------
    values_parameter : array, shape(k,)
        initialization of the components of index "index_parameter" of the parameters of the k parametric functions.

    index_parameter :  int,
        index of the dimension of the parameter space over which the optimization is performed. Must be inferior to
        the dimension d of the parameter space.

    A : array, shape(n,k)
    array of size (n,k) corresponding to the linear
        coefficients in the mixture of  k parametric functions used to approximate the n signals.

    parameters : array, shape(k,d)
        current parameters of the k parametric functions used to approximate the n signals.
         Each parametric function is parametrized by a parameter of dimension d.

    data : array, shape (p,n)
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
        value of the objective function.
    '''
    n, k0 = A.shape
    k = k0 - order_base
    new_parameters = parameters.copy()
    new_parameters[:,index_parameter] = values_parameter
    F = build_Phi(times, new_parameters, k, order_base, normalized, func)
    return np.sum((data - np.dot(F, np.transpose(A))) ** 2)


@jit(nopython=True, parallel=True,fastmath=True)
def JacObjectiveFunc(values_parameter,index_parameter, A,parameters, data, times, order_base, normalized, func, deriv_func):
    '''

    Parameters
    ----------
    values_parameter : array, shape(k,)
        initialization of the components of index "index_parameter" of the parameters of the k parametric functions to
        be optimized.

    index_parameter :  int,
        index of the dimension of the parameter space (of dimension d) over which the optimization is performed.

    A : array, shape(n,k)
    array of size (n,k) corresponding to the linear
        coefficients in the mixture of  k parametric functions used to approximate the n signals.


    parameters : array, shape(k,d)
        current parameters of the k parametric functions. Each parametric function is parametrized by a parameter
        of dimension d.

    data : array, shape (p,n)
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

    deriv_func : callable
        derivative of the parametric function "func" with respect to the parameter of index "index_parameter".
        deriv_func(index_parameter, parameters, x) -> float, where "x" is either a float or an array of float,
        "parameters" is an array of shape (d,).

    Returns
    -------
    array, shape(k,)
        Jacobian of the objective function.
    '''
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
    '''
    This function solves a nonlinear optimization problem appearing in the SFW algorithm.
    The optimization is not performed on the k (number of parametric functions) * d (dimension of the parameter space)
    parameters. Indeed, it is performed successively on the dimension of the parameter space.
    Parameters
    ----------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    A : array of size (n,k) corresponding to the linear
        coefficients in the mixture of  k parametric functions used to approximate the n signals.

    parameters :  array, shape(k,d)
        parameters of the k parametric functions. Each parametric function is parametrized by a parameter of dimension d.

    bounds : array, shape(d,2,k)
        bound[i,0,j] corresponds to the lower bounds for the i-th dimension of the k-th parametric function.
        bound[i,1,j] corresponds to the upper bounds for the i-th dimension of the k-th parametric function.

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

    '''
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
