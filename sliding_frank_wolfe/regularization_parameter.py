from sliding_frank_wolfe.SFW_algorithm import SFW
import numpy as np
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.group_Lasso_utils import regressionGroupLasso
import matplotlib.pyplot as plt


def find_lambda(data, times, range_lbda, lower_bounds, upper_bounds, func, deriv_func, threshold=1e-5, merging_threshold=1e-5, rank="full", size_grids = None,  normalized=True, epsilon = 1e-2, max_iter=100, size_mesh=1000):
    '''
    This function performs the SFW algorithm for different values of the regularization parameter.

    Parameters
    ----------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    range_lbda : array, shape (N,)
        array containing different values for the regularization parameter.

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

    threshold : float, must be non negative
        threshold on the 2-norm of the linear coefficients associated to a parametric function
         under which the parametric function is removed from the optimization.

    merging_threshold : float, must be non negative
        threshold on the 2-norm of the difference between two parametric functions
         under which the parametric functions are merged.

    rank : "full" or an non negative int <= optimSFW.linear_parameters.shape[1] , if rank is not "full" the algorithm run
     a SVD decomposition on the matrix of linear coefficients to return a matrix of rank "rank".

    size_grids : array, shape (d,)
        The k-th coordinate of "size_grids" corresponds to the size of the initialization grid for the k-th parameter
        used to locate a new spike at each Frank-Wolfe iteration.

    normalized : bool,
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.

    epsilon : float, must be non negative
        epsilon is the tolerance for the stopping criteria used in the Frank-Wolfe algorithm.

    max_iter : int,
        maximal number of Frank_Wolfe iterations allowed.

    size_mesh : int,
        number of points in the mesh on the parameter space over which the stopping criteria is checked.

    Returns
    -------
    err : array, shape(N,)
        values of the data fidelty term of the objective function at the end of the SFW algorithm
        for the different values of the regularization parameter.

    err_penalize : array, shape(N,)
        values of the objective function (data fidelty term + penalty ) of the optimization problem  at the end of the SFW algorithm
        for the different values of the regularization parameter.

    nb_spikes : array(N,)
        number of peaks (or spikes) found by the SFW algorithm for the different values of the regularization parameter.
    '''
    err = np.zeros(len(range_lbda))
    err_penalized = np.zeros(len(range_lbda))
    nb_spikes = np.zeros(len(range_lbda))
    p, n = data.shape
    np.random.shuffle(np.transpose(data))
    for i in range(len(range_lbda)):
        res_optim = SFW(data, times, range_lbda[i], lower_bounds, upper_bounds,
                        func, deriv_func, threshold=threshold,
                        merging_threshold=merging_threshold, rank=rank,
                        size_grids=size_grids, normalized=normalized,
                        epsilon=epsilon, max_iter=max_iter, size_mesh=size_mesh)
        A = res_optim.linear_coefficients
        parameters = res_optim.dictionary_parameters
        k = np.shape(A)[1]
        nb_spikes[i] = k
        F = build_Phi(times, parameters, k, 0, normalized, func)
        err[i] = (1. / (p * n)) * np.linalg.norm(data - F.dot(np.transpose(A.reshape((n, k)))), ord='fro') ** 2
        group_norm = np.sum(np.linalg.norm(A.reshape((n, k)), ord=2, axis=0))
        err_penalized[i] = (1. / (p * n)) * np.linalg.norm(data - F.dot(np.transpose(A.reshape((n, k)))),
                                                           ord='fro') ** 2 + range_lbda[i] * group_norm
    return err, err_penalized, nb_spikes
