from sliding_frank_wolfe.dictionary import expo
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.group_Lasso_utils import objectiveRegGroupLasso
from sliding_frank_wolfe.group_Lasso_utils import JacRegGroupLasso
from sliding_frank_wolfe.group_Lasso_utils import group_lasso_step
from scipy.optimize import check_grad
import numpy as np


def test_objective_reg_group_lasso():
    times = np.linspace(-20, 20, 100)
    parameters = np.array([[3, 4], [5, 8]])
    A = np.array([[2, 3]])
    F = build_Phi(times, parameters, 2, 0, True, expo)
    assert (objectiveRegGroupLasso(A.flatten(), np.dot(F, A.T), F, 1, 0) == np.sum(np.linalg.norm(A, ord=2, axis=0)))


test_objective_reg_group_lasso()


def test_jac_reg_group_lasso():
    lbda = 1
    times = np.linspace(-20, 20, 100)
    parameters = np.array([[3, 4], [5, 10]])
    A = np.random.uniform(-2, 2, (2, 2))
    F = build_Phi(times, parameters, 2, 0, True, expo)
    Y = np.dot(F, A.T) + np.random.normal(0, 1, (100, 2))
    tolerance = 10e-5
    for i in range(10000):
        X0 = np.random.uniform(-10, 10, 4)
        assert (check_grad(lambda x: objectiveRegGroupLasso(x, Y, F, lbda, 0), lambda x: JacRegGroupLasso(x, Y, F, lbda, 0),
                       X0) < tolerance)



def test_group_lasso_step():
    #test 1
    lbda = 0
    times = np.linspace(-20, 20, 100)
    parameters = np.array([[3, 4], [5, 3]])
    F = build_Phi(times, parameters, 2, 0, True, expo)
    for i in range(10):
        A = np.random.uniform(10, 20, (2, 2))
        data = np.dot(F, A.T)
        X0= np.random.uniform(-2, 2, (2, 2))
        assert( np.allclose(group_lasso_step(data, times, X0, parameters, lbda, 0, True, expo), A))
    #test 2
    parameters = np.array([[3, 4]])
    F = build_Phi(times, parameters, 1, 0, True, expo)
    tolerance = 1e-2
    for i in range(10):
        lbda = np.random.uniform(1e-5, 1e-4)
        a = np.array([[np.random.uniform(3, 10)]])
        data = np.dot(F, a.T)
        p, n  = data.shape
        X0 = np.array([[np.random.uniform(-2, 2)]])
        solution = group_lasso_step(data, times, X0, parameters, lbda, 0, True, expo)[0,0]
        assert(np.abs(np.abs(2 * (1. / (p * n*lbda)) * np.inner(data.T- solution * F.T,F.T) ) -1 ) <  tolerance)

