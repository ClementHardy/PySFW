import  numpy as np
from numba import jit
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sliding_frank_wolfe.tools import build_Phi
#from sklearn.linear_model import LinearRegression, Lasso
#from sklearn.linear_model import


def objectiveRegGroupLasso(x, data, F, lbda, order_base):
    '''
    This function computes the value of the objective function of the group-Lasso optimization problem (subproblem in the
     SFW algorithm).
    Parameters
    ----------
    x : array, shape(n*k,)
        vector of linear coefficients. x.reshape((n, k)) gives an array of size (n,k) corresponding to the linear
        coefficients in the mixture of  the k parametric functions used to approximate the n signals.

    data : array, shape (p,n)
        array of n signals distretized on p points.

    F : array, shape(p,k)
        array containing k parametric functions discretized on p points.

    lbda : float
        regularization parameter of the optimization problem.

    Returns
    -------
    float,
        value of the objective function of the group-Lasso problem.

    '''
    p, k = F.shape
    n = data.shape[1]
    group_norm = np.sum(np.linalg.norm(x.reshape((n, k))[:, :k - order_base], ord=2, axis=0))
    return (1. / (p * n)) * np.linalg.norm(data - F.dot(np.transpose(x.reshape((n, k)))),
                                           ord='fro') ** 2 + lbda * group_norm


def JacRegGroupLasso(x, Y, F, lbda, order_base):
    '''
    This function computes the value of the Jacobian of the objective function of the group-Lasso problem.

    Parameters
    ----------
    x : array, shape(n*k,)
        vector of linear coefficients. x.reshape((n, k)) gives a table of size (n,k) corresponding to the linear
        coefficients in the mixture of the k parametric functions used to approximate the n signals.

    data : array, shape (p,n)
        array of n signals distretized on p points.

    F : array, shape(p,k)
        array containing k parametric functions  discretized on p points.

    lbda : float
        regularization parameter of the optimization problem.

    Returns
    -------
    array, shape (n*k,)
        Jacobian of the objective function
    '''
    p, n = Y.shape
    k = F.shape[1]
    A = Y - F.dot(np.transpose(x.reshape((n, k))))
    norm = np.linalg.norm(x.reshape((n, k))[:, :k - order_base], ord=2, axis=0)
    temp = np.zeros(len(norm))
    index_non_zero = np.nonzero(norm)
    for index in index_non_zero:
        temp[index] = 1. / norm[index]
    where_are_NaNs = np.isnan(temp)
    temp[where_are_NaNs] = 0
    where_are_infs = np.isinf(temp)
    temp[where_are_infs] = 0
    B = lbda * (x.reshape((n, k))[:, :k - order_base] * temp)

    return -2 * (1. / (p * n)) * (np.dot(np.transpose(A), F)).flatten() + (
        np.hstack((B, np.zeros((n, order_base))))).flatten()


def regressionGroupLasso(X, Y, F, lbda, order_base, positive=False):
    '''
    This function solve the goup-Lasso optimization problem.
    Parameters
    ----------
    X : array, shape(n,k)
    array of size (n,k) corresponding to an initialization of the linear
        coefficients in the mixture of  k parametric functions used to approximate the n signals.

    Y : array, shape (p,n)
        array of n signals distretized on p points.

    F : array, shape(p,k)
        array containing k parametric functions  discretized on p points.

    lbda : float
        regularization parameter of the optimization problem.

    positive : Bool,
        if True the coefficients in the linear combination of parametric functions used to approximate the signals
        are required to be non negative.

    Returns
    -------
    array, shape(n,k)
        array of size (n,k) corresponding to the linear coefficients found by the procedure to approximate
        n signals by a mixture of  k parametric functions.
    '''
    n, k = X.shape
    if positive:
        temp = np.hstack((np.zeros((n, k - order_base)), np.ones((n, order_base))))
        for i in range(1, order_base + 1):
            temp[:, k - i] = -1 * np.inf * temp[:, k - i]
        lower = temp.flatten()
        upper = np.inf * np.ones(n * k)
        bounds = Bounds(lower, upper)
        res = minimize(lambda x: objectiveRegGroupLasso(x, Y, F, lbda, order_base), X.flatten(), method='L-BFGS-B',
                       jac=lambda x: JacRegGroupLasso(x, Y, F, lbda, order_base), bounds=bounds)
    if not (positive):
        res = minimize(lambda x: objectiveRegGroupLasso(x, Y, F, lbda, order_base), X.flatten(), method='L-BFGS-B',
                       jac=lambda x: JacRegGroupLasso(x, Y, F, lbda, order_base))
    return (res.x).reshape((n, k))

def group_lasso_step(data, times, A, parameters, lbda, order_base, normalized, func, positive):
    '''
    This function performs the linear step of the SFW algorithm. It solves a group-Lasso optimization problem.

    Parameters
    ----------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    A : array, shape(n,k)
    array of size (n,k) corresponding to an initialization of the linear
        coefficients in the mixture of  k parametric functions used to approximate the n signals.

    parameters : array, shape(k,d)
        parameters of k parametric functions. Each parametric function is parametrized by a parameter of dimension d.

    lbda : float
        regularization parameter of the optimization problem.

    normalized : bool,
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.

    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    positive : Bool,
        if True the coefficients in the linear combination of parametric functions used to approximate the signals
        are required to be non negative.

    Returns
    -------
    array, shape(n,k)
        array of size (n,k) corresponding to the linear coefficients found by the procedure to approximate
        n signals by a mixture of  k parametric functions.
    '''
    k = parameters.shape[0]
    F = build_Phi(times, parameters, k, order_base, normalized, func)
    return regressionGroupLasso(A, data, F, lbda, order_base, positive)

