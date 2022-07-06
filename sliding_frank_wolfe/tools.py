
import numpy as np
from numba import jit
from scipy.spatial.distance import cdist
from numpy import triu_indices






@jit(nopython=True)
def build_Phi(times,parameters, k, order_base, normalized,func):
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
    k, number_parameters = parameters.shape
    F = build_Phi(times, parameters, k, 0, normalized, func)
    dist_matrix = cdist(np.transpose(F), np.transpose(F), 'minkowski', p=2)
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
                    parameters[:,jj][index_matrix[j][1]] = (1 - alea) * lower_bounds[jj] + alea * upper_bounds[jj]
                compt += 1
                F = build_Phi(times, parameters, k, 0, normalized, func)
                if np.linalg.norm(F[:,index_matrix[j][0]] - F[:,index_matrix[j][1]], ord= 2) > merging_threshold:
                    var_bool = False
            if compt == 3:
                print("Warning : the merging threshold might be too high!")
    return A, parameters
