import os, sys
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from sliding_frank_wolfe.dictionary import expo
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.SFW_algorithm import stop_condition
import numpy as np



def test_stop_condition():
    n = 10
    p = 100
    reg = 2 /(p*n)
    epsilon = 1e-4
    size_mesh = 1000
    times  = np.linspace(-20,20,p)
    lower_bounds = np.array([-20,1])
    upper_bounds = np.array([20,10])
    residual = (1./p) * np.ones((p,n))
    assert(stop_condition(residual, times, epsilon, reg, lower_bounds, upper_bounds, size_mesh, True, expo) == False )
    residual =  np.ones((p, n))
    assert (stop_condition(residual, times, epsilon, reg, lower_bounds, upper_bounds, size_mesh, True, expo) == True)
