from sliding_frank_wolfe.SFW_algorithm import SFW
import numpy as np
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.group_Lasso_utils import regressionGroupLasso
import matplotlib.pyplot as plt


def cross_validation_lambda(data, times, size_partition, range_lbda, lower_bounds, upper_bounds, func, deriv_func, threshold=1e-5, merging_threshold=1e-5, rank="full", size_grids = None,  normalized= True, epsilon = 1e-2, max_iter = 100, step_mesh = 1e-1):
    err = np.zeros((len(range_lbda), size_partition))
    p, n = data.shape
    np.random.shuffle(np.transpose(data))
    step = n // size_partition
    data = data[:, :step * size_partition]
    for i in range(len(range_lbda)):
        for j in range(size_partition):
            data_test = data[:, j * step:(j + 1) * step]
            data_train = np.hstack((data[:, :j * step], data[:, :(j + 1) * step]))
            res_optim = SFW(data_train, times, range_lbda[i], lower_bounds, upper_bounds,
                                                        func, deriv_func, threshold=threshold,
                                                        merging_threshold=merging_threshold, rank=rank,
                                                        size_grids= size_grids, normalized=normalized,
                                                        epsilon = epsilon, max_iter=max_iter, step_mesh = step_mesh)

            A_train = res_optim.linear_coefficients
            parameters_train = res_optim.dictionary_parameters
            k_train = np.shape(A_train)[1]
            F_train = build_Phi(times, parameters_train, k_train, 0, normalized, func)
            A = np.abs(np.random.normal(0, 10, size=(step, k_train)))
            A_test = regressionGroupLasso(A, data_test, F_train, 0, 0, False)
            p_test, n_test = data_test.shape
            k_test = np.shape(A_test)[1]
            err[i, j] = (1. / (p_test * n_test)) * np.linalg.norm(
                data_test - F_train.dot(np.transpose(A_test.reshape((n_test, k_test)))), ord='fro') ** 2
    mean_err = np.sum(err, axis=1)
    plt.figure()
    plt.plot(range_lbda, mean_err)
    plt.show()
    return err


def find_lambda(data, times, range_lbda, lower_bounds, upper_bounds, func, deriv_func, threshold=1e-5, merging_threshold=1e-5, rank="full", size_grids = None,  normalized=True, epsilon = 1e-2, max_iter=100, step_mesh=1e-1):
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
                        epsilon=epsilon, max_iter=max_iter, step_mesh=step_mesh)
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
