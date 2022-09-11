
######################################################################
from sliding_frank_wolfe.tools import trunc_rank
from sliding_frank_wolfe.tools import merge_function_Phi
from sliding_frank_wolfe.tools import build_Phi
from sliding_frank_wolfe.adding_spikes_utils import locate_new_spike
from sliding_frank_wolfe.group_Lasso_utils import group_lasso_step, objectiveRegGroupLasso
from sliding_frank_wolfe.optimizer_utils import nlls_step_jac_decomp
import numpy as np
#######################################################################



class optimSFW:
    """
    object returned by the sliding Franck-Wolfe algorithm.

    Attributes:
    --------
    linear_coefficients : array, shape(n,k)
        contains the linear coefficients found by the sliding Frank-wolfe algorithm to approximate the n signals.
        linear_coefficients[i,j] corresponds to the linear coefficients associated to the i-th signal and the j-th
        parametric function.

    dictionary_parameters : array, shape(k,d)
        array that contains the d-dimensional parameters of the k parametric functions found by the sliding Frank-Wolfe
        to approximate the data.

    iterations : array, shape(number_of_iterations,)
        array that contains the successive values of the objective function  minimized at each step of the sliding
        Frank-Wolfe algorithm.

    history_norms_linear_parameters : array, shape(number_of_iterations,k)
        array that contains (when the parameters "threshold" and "merging_threshold" of the SFW algorithm are taken equal to zero)
        the history of the 2-norm of the columns of the array "linear_coefficients"
        after each Frank-Wolfe iteration.

    is_full_rank, bool
        if is_full_rank = True, the matrix linear_coefficients of size (n,k) is required to be of full rank.

    sparsity, int
        sparsity = k, the number of parametric functions found by the SFW algorithm to approximate the data.

    rank, int
        rank of the matrix  linear_coefficients of size (n,k).
    """
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
    Run the sliding Frank-Wolfe algorithm on a set of n signals disctretized on p points. The signals are approximated by
    linear combinations of parametric functions "func". The parametric functions are parametrized by a parameter of dimension d.

    Parameters
    ----------
    data : array, shape (p,n)
        array of n signals distretized on p points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    reg : float
        regularization parameter of the optimization problem.

    lower_bounds : array, shape(d,)
        lower_bounds on the parameters of the parametric functions. The k-th coordinate of the array "lower_bounds"
        corresponds to the lower_bound on the k-th dimension of the parameter.

    upper_bounds : array, shape(d,)
        upper_bounds on the parameters of the parametric functions. The k-th coordinate of the array "upper_bounds"
        corresponds to the upper_bound on the k-th dimension of the parameter.

    func : callable,
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    deriv_func : callable,
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

    step_mesh : float,
        step for the mesh on the parameter space over which the stopping criteria is checked.

    positive : Bool,
        if True the coefficients in the linear combination of parametric functions used to approximate the signals
        are required to be non negative.

    Returns
    -------
    optimSFW object,
        optimSFW contains all the parameters and linear coefficients to approximate the n signals distretized on p points.
        At the end of the optimization mixtures of  k parametric functions are used to approximate the data.

    optimSFW.linear_coefficients : array, shape (n,k)
        It corresponds to the linear coefficents in the linear combinations of parametric functions used
        to approximate the n signals. linear_coefficients[i,j] corresponds to the linear coefficients associated to the
        i-th signal and the j-th parametric function.

    optimSFW.parameters : array shape (k,d)
        array containing the d-dimensional parameters of the k parametric functions used to approximate the signals.

    optimSFW.iterations : array, shape (nb_ite,), where nb_ite corresponds to the number of Frank-Wolfe iterations used.
        It contains the history of the values of the objective function after each iteration of the
        Frank-Wolfe algorithm.

    optimSFW.history_norms_linear_parameters  : array shape (max_iter,k)
        It contains the history of the 2-norm of the columns of the array "linear_coefficients"
        after each Frank-Wolfe iteration.

    optimSFW.sparsity, int
        number of parametric functions used to approximate the n signals.


    """

    #INITIALIZATION OF THE VARIABLES
    p, n = data.shape # n is the number of signals, p is the number of points for each signal
    assert(threshold >= 0)
    assert (merging_threshold >= 0)
    assert(epsilon >= 0)
    assert(step_mesh >=0)
    number_parameters = len(lower_bounds) #this variable corresponds to the dimension of the parameter space : d
    bounds = np.zeros((number_parameters,2,max_iter)) #this table gathers the upper bounds and the lower bounds in one table
    for i in range(number_parameters):
        bounds[i,0,:] = np.ones(max_iter) * lower_bounds[i]
        bounds[i, 1, :] = np.ones(max_iter) * upper_bounds[i]
    if size_grids == None:
        size_grids = 10 * np.ones(number_parameters,dtype = int) #default grid size is 10 * d
    A = np.zeros((n, max_iter)) #table that will contain the linear coefficients of the mixture
    ite = [] #table that will contain the value of the objective function after each sliding Frank-Wolfe step
    coeffA = []
    x0 = locate_new_spike(data, times, normalized, size_grids, lower_bounds, upper_bounds, func,
                          deriv_func) #find the parameter of the first parametric function used to approximate the data
    parameters_temp = np.array([x0]) #update the table of parameters
    Residual = data #initialization of the residuals
    i = 0
    k_temp = 1 #initialization of the current number of parametric used to approximate the data
    A_temp = np.reshape(A[:, i], (n, k_temp))
    #FRANK-WOLFE ITERATIONS
    while stop_condition(Residual,times,epsilon,reg, lower_bounds, upper_bounds,step_mesh, normalized, func) and i < max_iter:
        if i > 0:
            k_temp = parameters_temp.shape[0]
            F = build_Phi(times, parameters_temp, k_temp, 0, normalized, func) #build the current dictionary of parametric functions
            Residual = data - np.dot(F, np.transpose(A_temp)) #update the residuals
            x0 = locate_new_spike(Residual, times, normalized, size_grids, lower_bounds, upper_bounds, func,
                                  deriv_func) #find the parameter of the k_temp-th parametric function used to approximate the data
            parameters_temp = np.vstack((parameters_temp, x0)) #update the table containing the parameters of the functions in the dictionary
            k_temp = parameters_temp.shape[0] #current number of parametric functions used to approximate the data
            A_temp = np.hstack((A_temp, np.reshape(A[:, i], (n, 1))))

        # LINEAR REGRESSION STEP
        #print("linear descent: ", i)
        A_temp = group_lasso_step(data, times, A_temp, parameters_temp, reg, 0, normalized, func, positive = positive) #current estimation of the linear coefficients
        # SUPPRESS USELESS VARIABLES
        #suppress the parametric functions in the current dictionary associated to null linear coeffficients
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
        parameters_temp  = nlls_step_jac_decomp(data, times, A_temp, parameters_temp, 0, bounds[:,:,:k_temp] , normalized, func, deriv_func) #non linear least square step to update the non linear parameters
        # MERGE CLOSE PARAMETRIC FUNCTIONS
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
    """
    Compute the stopping criteria used in the sliding Frank-Wolfe algorithm

    Parameters
    ----------
    residual : array, shape(p,n)
        array containing the current value of the residuals in a SFW step. n is the number of signals, p is the number
        of discretization points.

    times : array, shape(p,)
        array of size p corresponding to the points over which the signals are discretized.

    epsilon : float, must be non negative
        tolerance used to check the stopping criteria

    reg : float
        regularization parameter of the optimization problem.

    lower_bounds : array, shape(d,)
        lower_bounds on the parameters of the parametric functions. The k-th coordinate of the array "lower_bounds"
        corresponds to the lower_bound on the k-th dimension of parameter.

    upper_bounds : array, shape(d,)
        upper_bounds on the parameters of the parametric functions. The k-th coordinate of the array "upper_bounds"
        corresponds to the upper_bound on the k-th dimension of the parameter.

    step_mesh : float,
        step for the mesh on the parameter space over which the stopping criteria is checked.

    normalized : bool
        if normalized == True, the parametric functions discretized on p points used to approximate the signals
        are normalized with respect to the 2-norm.

    func : callable
        parametric function giving the continuous dictionary over which the signals are decomposed
        func(parameters, x) -> float`, where "x" is either a float or an array of float, "parameters" is an array of
        shape (d,).

    Returns
    -------
    var_bool : bool
        False if the SFW algorithm must stop.
    """

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

