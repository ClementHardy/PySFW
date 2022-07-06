
from sliding_frank_wolfe.optimizer_utils import objectiveFunc
from sliding_frank_wolfe.optimizer_utils import  JacObjectiveFunc
from sliding_frank_wolfe.dictionary import expo
from sliding_frank_wolfe.dictionary import derivExpo
from sliding_frank_wolfe.tools import build_Phi
import numpy as np
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt


def test_jac_objective_func():
    p = 1000
    n = 10
    k = 3
    parameters = np.hstack(
        (np.random.uniform(-10, 10, k).reshape(k, 1), np.random.uniform(10,15,k).reshape(k, 1)))
    times = np.linspace(-100,100,p)
    F = build_Phi(times,parameters,k,0,False,expo)
    A = np.ones((n, k))
    data = np.dot(F,A.T)
    tolerance = 1e-2
    print(JacObjectiveFunc(np.array([20,10,5]), 0, A, parameters, data, times, 0, True, expo,
                           derivExpo))
    print(approx_fprime(np.array([20,10,5]), lambda x: objectiveFunc(x, 0, A, parameters, data, times, 0, True, expo), 1e-5))
    for i in range(100):
        X0 = np.random.uniform(-20,20,k)
        # print(JacObjectiveFunc(X0, 0, A, parameters, data, times, 0, True, expo,
        #                        derivExpo))
        # print(approx_fprime(X0, lambda x: objectiveFunc(x, 0, A, parameters, data, times, 0, True, expo), 1e-5))
        assert (check_grad(lambda x: objectiveFunc(x, 0, A, parameters, data, times, 0, True, expo),
                lambda x: JacObjectiveFunc(x, 0, A, parameters, data, times, 0, True, expo, derivExpo),
                           X0) < tolerance)

        Y0 = np.random.uniform(5, 20,k)
        # print(JacObjectiveFunc(Y0, 1, A, parameters, data, times, 0, True, expo,
        #                        derivExpo))
        # print(approx_fprime(Y0, lambda x: objectiveFunc(x, 1, A, parameters, data, times, 0, True, expo), 1e-5))
        assert (check_grad(lambda x: objectiveFunc(x,1, A, parameters, data, times, 0, True, expo), lambda x: JacObjectiveFunc(x, 1, A, parameters, data, times, 0, True, expo,
                     derivExpo),
                       Y0) < tolerance)