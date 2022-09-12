import os, sys
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
from sliding_frank_wolfe.dictionary import expo
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.tools import trunc_rank
from sliding_frank_wolfe.tools import merge_function_Phi
from scipy.spatial.distance import cdist
from numpy import triu_indices
import matplotlib.pyplot as plt
import pytest


def test_build_phi():

    x = np.linspace(-20, 20, 100)
    parameters = np.array([[1, 2], [2, 3]])
    y = np.sum(build_Phi(x, parameters, 2, 0, False, expo), axis=1)
    z = np.exp(-1 * ((x - 1) ** 2) / 2 ** 2) + np.exp(-1 * ((x - 2) ** 2) / 3 ** 2)
    assert (len(y) == len(x))
    assert np.linalg.norm(y - z, ord=2) < 0.001
    yy = build_Phi(x, parameters, 2, 0, True, expo)
    assert (np.linalg.norm(yy[:, 1], ord=2) == 1)
    assert (np.linalg.norm(yy[:, 0], ord=2) == 1)


    parameters = np.array([[1, 2]])
    y = build_Phi(x, parameters, 1, 0, False, expo)
    z = np.exp(-1 * ((x - 1) ** 2) / 2 ** 2)
    assert (len(y) == len(x))
    assert np.linalg.norm(y.T - z, ord=2) < 0.001

test_build_phi()


def test_trunc_rank():
    A = np.random.normal(0, 1, (5, 5))
    rank = np.linalg.matrix_rank(A)
    assert (np.linalg.matrix_rank(trunc_rank(A, rank='full')) == rank)
    assert (np.linalg.matrix_rank(trunc_rank(A, rank=1)) == 1)


test_trunc_rank()


def test_merge_function_phi():
    A = np.array([[1,1,3,4,5]])
    parameters  = np.array([[2,1],[-5,1],[2,1],[-5,1],[2,1]])
    times =  np.linspace(-20, 20, 100)
    merging_threshold = 1.2
    lower_bounds = [-20,-20]
    upper_bounds = [20,20]
    B, new_parameters = merge_function_Phi(merging_threshold, A, parameters, times, lower_bounds, upper_bounds, True, expo)
    assert( B[0,2] == 4)
    #print(B)
    #print(new_parameters)
    F = build_Phi(times, new_parameters, 5, 0, True, expo)
    #print(cdist(np.transpose(F), np.transpose(F), 'minkowski', p=2))


test_merge_function_phi()