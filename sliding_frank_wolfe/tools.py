
import numpy as np
from numba import jit
from scipy.spatial.distance import cdist
from numpy import triu_indices






@jit(nopython=True)
def build_Phi(times,parameters, k, order_base, normalized,func):
    """
    Build a dictionary of k (normalized, if normalized = True) parametric functions discretized on p points (= len(times)) whose parameters
    belong to the table parameters "parameters".

    Parameters
    ----------
    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    parameters : array, shape(k,d)
        array of size k (number of parametric functions) * d (dimension of the parameter space).

    k : int,
        number of parametric functions.

    order_base : int,
        degree of the additional polynomials in the dictionary.

    normalized : bool
    if normalized == True, the parametric functions discretized on p points used to approximate the signals
    are normalized with respect to the 2-norm.

    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    Returns
    -------
    F : array, shape (p, k + order_base)
     array containing the k parametric functions and the the polynomials x^0,...,x^order_base discretized on p points.
    """
    p = times.size
    F = np.ones((p, k + order_base))
    for j in range(k):
        F[:, j] = func(parameters[j], times)
    for m in range(order_base):
        F[:, m + k] = times ** m
    if normalized == True:
        norms = np.ones(k)
        for i in range(k):
            norms[i] = np.linalg.norm(F[:, i], ord=2)
        F = F / np.hstack((norms, np.ones(order_base)))
    return F


def trunc_rank(A, rank='full'):
    """
    Parameters
    ----------
    A, array, shape(n,k)

    rank, int

    Returns
    -------
    This function returns a matrix that approximates A by keeping only the first eigenvalues (up to the rank-th) of A in the Singular Value
    Decomposition (SVD). When rank is equal to 'full', it returns A.

    """
    if rank != 'full':
        u, s, v = np.linalg.svd(A)
        if rank < len(s):
            s2 = np.zeros(len(s))
            s2[:rank] = s[:rank].copy()
            if len(s) < u.shape[1]:
                A = u[:, :len(s)] @ np.diag(s2) @ v
            else:
                A = u @ np.diag(s2) @ v[:len(s), :]
    return A


def merge_function_Phi(merging_threshold, A, parameters, times, lower_bounds, upper_bounds, normalized, func):
    """
    This function merges the parametric functions used to approximate the data that are too close. Two functions are
    merged when the 2-norm of the difference between the vectors corresponding of their values on a grid is under a
    threshold "merging_threshold". A new parameter is generated at random (uniformly between the upper and lower bounds).

    Parameters
    ----------
    merging_threshold : float, must be non negative,
        threshold  under which two parametric functions are merged.

    A : array, shape(n,k)
        array containing the linear coefficients in the mixture of k parametric functions used to approximate the
        n signals. The k-th column corresponds to the n linear coefficients associated to the k-th parametric function.

    parameters : array, shape(k,d)
        array containing the k d-dimensional parameters associated to the k parametric functions used to approximate the
        data.
    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    lower_bounds : array, shape(d,)
        lower_bounds on the parameters of the parametric functions. The k-th coordinate of the array "lower_bounds"
        corresponds to the lower_bound on the k-th dimension of the parameter.

    upper_bounds : array, shape(d,)
        upper_bounds on the parameters of the parametric functions. The k-th coordinate of the array "upper_bounds"
        corresponds to the upper_bound on the k-th dimension of the parameter.

    normalized : bool
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.

    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    Returns
    -------
    A : array, shape(n,k)
        array obtained after the merging and containing the linear coefficients  in the mixture of k parametric functions
        used to approximate the n signals. The k-th column corresponds to the n linear coefficients associated to
        the k'-th parametric function.

    parameters : array, shape(k,d)
        array containing the k d-dimensional parameters associated to the k parametric functions used to approximate
        the data after the merging.
    """
    k, number_parameters = parameters.shape
    F = build_Phi(times, parameters, k, 0, normalized, func) #build the dictionary of k parametric functions discretized on p points
    dist_matrix = cdist(np.transpose(F), np.transpose(F), 'minkowski', p=2) #compute de distance matrix between the parametric functions
    assert (len(dist_matrix) == k)
    index_upper_triangle = triu_indices(len(dist_matrix))
    assert (merging_threshold >= 0)
    dist_matrix[index_upper_triangle] = 2 * merging_threshold + 1
    index_matrix = np.argwhere(dist_matrix < merging_threshold)
    for j in range(len(index_matrix)):
        F = build_Phi(times, parameters, k, 0, normalized, func)
        if np.linalg.norm(F[:, index_matrix[j][0]] - F[:, index_matrix[j][1]], ord=2) <= merging_threshold:
            A[:, index_matrix[j][0]] = A[:, index_matrix[j][0]] + A[:, index_matrix[j][1]]
            compt = 0
            var_bool = True
            while compt < 3 and var_bool:
                for jj in range(number_parameters):
                    alea = np.random.rand()
                    parameters[:,jj][index_matrix[j][1]] = (1 - alea) * lower_bounds[jj] + alea * upper_bounds[jj] #generate a new parameter at random
                compt += 1
                F = build_Phi(times, parameters, k, 0, normalized, func)
                if np.linalg.norm(F[:,index_matrix[j][0]] - F[:,index_matrix[j][1]], ord= 2) > merging_threshold:
                    var_bool = False
            if compt == 3:
                print("Warning : the merging threshold might be too high!")
    return A, parameters
