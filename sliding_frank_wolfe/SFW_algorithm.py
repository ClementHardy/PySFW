

from sliding_frank_wolfe.tools import trunc_rank
from sliding_frank_wolfe.tools import merge_function_Phi
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.adding_spikes_utils import locate_new_spike
from sliding_frank_wolfe.group_Lasso_utils import group_lasso_step, objectiveRegGroupLasso
from sliding_frank_wolfe.optimizer_utils import nlls_step_jac_decomp
import numpy as np

class optimSFW:
    def __init__(self,linear_parameters, dictionary_parameters, iterations, history_norms_linear_parameters):
        self.linear_coefficients = linear_parameters
        self.dictionary_parameters = dictionary_parameters
        self.iterations = iterations
        self.history_norms_linear_parameters = history_norms_linear_parameters
        self.is_full_rank = np.linalg.matrix_rank(linear_parameters) == (linear_parameters.shape[1])
        self.sparsity = linear_parameters.shape[1]
        self.rank = np.linalg.matrix_rank(linear_parameters)

def SFW(data, times, reg, lower_bounds, upper_bounds, func, deriv_func, threshold=1e-4,
                 merging_threshold=1e-4, rank="full", size_grids = None, normalized= True,
                 epsilon = 1e-4, max_iter = 100, step_mesh = 1e-1, positive = False):

    """
    Run the sliding Frank-Wolfe algorithm on a set of n signals disctretized on p points. The signal are approximated by
    linear combinations of parametric functions "func". The parametric functions are parametrized by d parameters.

    Parameters
    ----------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signal are discretized.

    reg : float
        regularization parameter of the optimization problem.

    lower_bounds : array, shape(d,)
        lower_bounds on the parameters of the parametric functions. The k-th coordinate of the array "lower_bounds"
        corresponds to the lower_bound on the k-th parameter.

    lower_bounds : array, shape(d,)
        upper_bounds on the parameters of the parametric functions. The k-th coordinate of the array "upper_bounds"
        corresponds to the upper_bound on the k-th parameter.

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

    normalized : bool
    if normalized == True, the parametric functions discretized on p points used to approximate the signals
    are normalized with respect to the 2-norm.

    epsilon : float, must be non negative
    "epsilon" is the tolerance for the stopping criteria in the Frank-Wolfe algorithm.

    max_iter : int,
    maximal number of Frank_Wolfe iterations allowed.

    step_mesh : float,
    step for the mesh over which the stopping criteria is checked.



    Returns
    -------
    optimSFW object,
    optimSFW contains all the parameters and linear coefficients to approximate the n signal distretized on p points.
    At the end of the optimization k parametric functions are used to approximate the data.

    optimSFW.linear_coefficients : array, shape (n,k)
    It corresponds to the linear coefficents in the linear combination of parametric functions used
    to approximate the signals.

    optimSFW.parameters : array shape (k,d)
    array containing the d-dimensional parameters of the k parametric functions used to approximate the signals.

    optimSFW.iterations : array, shape (nb_ite,), where nb_ite corresponds to the number of Frank-Wolfe iterations used.
    It contains the history of the value of the objective function after each iteration of the
    Frank-Wolfe algorithm.

    optimSFW.history_norms_linear_parameters  : array shape (n, max_iter)
    It contains the history of the 2-norm of the linear coefficients associated to each parametric function
    after each Frank-Wolfe iteration.

    optimSFW.sparsity, int
    number of parametric functions used to approximate the n signals.

    positive, Bool,
    if True the coefficients in the linear combination are required to be non negative.

    """

    #INITIALIZATION OF THE PARAMETERS
    p, n = data.shape
    assert(threshold >= 0)
    assert (merging_threshold >= 0)
    assert(epsilon >= 0)
    assert(step_mesh >=0)
    number_parameters = len(lower_bounds)
    bounds = np.zeros((number_parameters,2,max_iter))
    for i in range(number_parameters):
        bounds[i,0,:] = np.ones(max_iter) * lower_bounds[i]
        bounds[i, 1, :] = np.ones(max_iter) * upper_bounds[i]
    if size_grids == None:
        size_grids = 10 * np.ones(number_parameters,dtype = int)
    A = np.zeros((n, max_iter))
    ite = []
    coeffA = []
    x0 = locate_new_spike(data, times, normalized, size_grids, lower_bounds, upper_bounds, func,
                          deriv_func)
    parameters_temp = np.array([x0])
    Residual = data
    i = 0
    k_temp = 1
    A_temp = np.reshape(A[:, i], (n, k_temp))
    #FRANK-WOLFE ITERATIONS
    while stop_condition(Residual,times,epsilon,reg, lower_bounds, upper_bounds,step_mesh, normalized, func) and i < max_iter:
        if i > 0:
            k_temp = parameters_temp.shape[0]
            F = build_Phi(times, parameters_temp, k_temp, 0, normalized, func)
            Residual = data - np.dot(F, np.transpose(A_temp))
            x0 = locate_new_spike(Residual, times, normalized, size_grids, lower_bounds, upper_bounds, func,
                                  deriv_func)
            parameters_temp = np.vstack((parameters_temp, x0))
            k_temp = parameters_temp.shape[0]
            A_temp = np.hstack((A_temp, np.reshape(A[:, i], (n, 1))))

        # LINEAR REGRESSION STEP
        #print("linear descent: ", i)
        A_temp = group_lasso_step(data, times, A_temp, parameters_temp, reg, 0, normalized, func, positive = positive)
        # SUPPRESS USELESS VARIABLES
        cut = []
        for j in range(k_temp):
            if np.linalg.norm(A_temp[:, j], 2) < threshold:
                cut.append(j)
        if len(cut) < k_temp:
            A_temp = np.delete(A_temp, cut, 1)
            parameters_temp = np.delete(parameters_temp, cut,0)
        k_temp = parameters_temp.shape[0]
        # NON LINEAR STEP
        #print('non linear descent :', i)
        parameters_temp  = nlls_step_jac_decomp(data, times, A_temp, parameters_temp, 0, bounds[:,:,:k_temp] , normalized, func, deriv_func)
        # MERGE CLOSE ATOMIC FUNCTIONS
        A_temp, parameters_temp = merge_function_Phi(merging_threshold, A_temp, parameters_temp,times, lower_bounds, upper_bounds,
                                                    normalized, func)

        # LINEAR REGRESSION STEP BIS
        #print("linear descent bis: ", i)
        A_temp = group_lasso_step(data, times, A_temp, parameters_temp, reg, 0, normalized, func, positive = positive)
        cut = []
        # SUPPRESS USELESS VARIABLES
        for j in range(k_temp):
            if np.linalg.norm(A_temp[:, j], 2) < threshold:
                cut.append(j)
        if len(cut) < k_temp:
            A_temp = np.delete(A_temp, cut, 1)
            parameters_temp = np.delete(parameters_temp, cut,0)
        k_temp = parameters_temp.shape[0]

        # COMPUTE OBJECTIVE FUNCTION
        F = build_Phi(times, parameters_temp, k_temp, 0, normalized, func)
        ite.append(objectiveRegGroupLasso(A_temp, data, F, reg, 0))
        A_temp = trunc_rank(A_temp, rank)
        i += 1
        coeffA.append(1. / len(A_temp) * np.linalg.norm(A_temp[:, :k_temp], ord=2, axis=0))

    #coeffA = np.array(coeffA)
    return optimSFW(A_temp, parameters_temp, ite, coeffA)

def stop_condition(residual,times,epsilon,reg, lower_bounds, upper_bounds,step_mesh, normalized, func):
    p,n = residual.shape
    var_bool = False
    #CREATE A MESHGRID
    number_parameters = len(lower_bounds)
    grids = []
    for i in range(number_parameters):
        grids.append(np.arange(lower_bounds[i], upper_bounds[i], step_mesh))
    meshgrid = np.meshgrid(*grids)
    flatten_meshgrid = []
    for j in range(len(meshgrid)):
        flatten_meshgrid.append(meshgrid[j].flatten())
    flatten_meshgrid = np.array(flatten_meshgrid)
    jj = 0
    #CHECK IF THE CERTFICATE FUNCTION IS UNDER n * ((reg * p * n / 2)**2) + epsilon ON THE MESHGRID
    while not var_bool and jj < len(flatten_meshgrid[0]):
        values = func(flatten_meshgrid[:,jj], times)
        if normalized:
            values = values / np.linalg.norm(values, ord=2)
        if np.linalg.norm(np.dot(values,residual), ord=2) ** 2 > n * ((reg * p * n / 2)**2) + epsilon:
            var_bool = True
        jj +=1

    return var_bool

