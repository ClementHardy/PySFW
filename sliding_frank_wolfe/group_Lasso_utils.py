import  numpy as np
from numba import jit
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sliding_frank_wolfe.tools import build_Phi
#from sklearn.linear_model import LinearRegression, Lasso
#from sklearn.linear_model import


def objectiveRegGroupLasso(x, data, F, lbda, order_base):
    p, k = F.shape
    n = data.shape[1]
    group_norm = np.sum(np.linalg.norm(x.reshape((n, k))[:, :k - order_base], ord=2, axis=0))
    return (1. / (p * n)) * np.linalg.norm(data - F.dot(np.transpose(x.reshape((n, k)))),
                                           ord='fro') ** 2 + lbda * group_norm


def JacRegGroupLasso(x, Y, F, lbda, order_base):
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
    k = parameters.shape[0]
    F = build_Phi(times, parameters, k, order_base, normalized, func)
    return regressionGroupLasso(A, data, F, lbda, order_base, positive)

