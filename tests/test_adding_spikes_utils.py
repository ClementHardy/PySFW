from sliding_frank_wolfe.adding_spikes_utils import jac_eta
from sliding_frank_wolfe.adding_spikes_utils import objective_eta
from sliding_frank_wolfe.adding_spikes_utils import locate_new_spike
from sliding_frank_wolfe.dictionary import expo
from sliding_frank_wolfe.dictionary import derivExpo
from sliding_frank_wolfe.tools import build_Phi
import numpy as np
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt


def test_jac_eta():
    p = 100
    n = 10
    times = np.linspace(-100, 100, p)
    number_parameters = 2
    A = np.ones((n, 2))
    parameters = np.array([[3, 9], [5, 8]])
    F = build_Phi(times, parameters, 2, 0, True, expo)
    residual = np.dot(F, A.T)
    tolerance = 1e-3
    X0 = np.array([np.random.uniform(-5, 5), np.random.uniform(10, 30)])
    assert (check_grad(lambda x: objective_eta(x, residual, times, True, expo),
                       lambda x: jac_eta(x, number_parameters, residual, times, True, expo, derivExpo),
                       X0) < tolerance)



def test_locate_new_spike():
    p = 1000
    n = 10
    tolerance = 1e-3
    times = np.linspace(-20, 20, p)
    parameters = np.array([[2,10]])
    A = np.random.normal(0,20,(n,1))
    F = build_Phi(times,parameters,1,0,True,expo)
    data = np.dot(F,A.T)
    size_grids = np.array([10,10])
    lower_bounds = np.array([-20,0.1])
    upper_bounds = np.array([20, 100])
    solution = locate_new_spike(data, times, True, size_grids, lower_bounds, upper_bounds, expo, derivExpo)
    assert np.linalg.norm(solution - parameters[0]) < tolerance
