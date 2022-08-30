
from numba import jit
import numpy as np





class dictionary:
    def __init__(self,function,derivatives, jacobian = None):
        func = func
        derivatives = derivatives
        if jacobian != None:
            jacobian = jacobian


@jit(nopython=True)
def cauchy(parameters, x):
    '''
    This function returns the values of the Cauchy function parametrized by the coordinates of the array
    "parameters" on the discretization points "x".

    Parameters
    ----------
    parameters : array, shape(d=2,)
        parameters of the Cauchy function. The first coordinate corresponds to the location, the second coordinate
            corresponds to the scale.
    x : float or array of shape(p,)
        discretization point or array of discretization points.

    Returns
    -------
    array, shape (p,)
        values of the Cauchy function parametrized by the the coordinates of the array
        "parameters" on the discretization points "x".
    '''
    b = parameters[0]
    c = parameters[1]
    return 1. / (1 + ((x - b) ** 2 / c ** 2))

@jit(nopython=True)
def derivCauchyB(parameters, x):
    '''
    This function returns the values of the derivative of the Cauchy function parametrized by the coordinates of the array
    "parameters" with respect to the location parameter (i.e parameters[0]) on the discretization points "x".

    Parameters
    ----------
    parameters : array, shape(d=2,)
        parameters of the Cauchy function. The first coordinate corresponds to the location, the second coordinate
            corresponds to the scale.

    x : float or array of shape(p,)
        discretization point or table of discretization points.

    Returns
    -------
    array, shape (p,)
        values of the derivative of the Cauchy function parametrized by the coordinates of the array
        "parameters" with respect to the location parameter (i.e parameters[0]) on the discretization points "x".
    '''
    b = parameters[0]
    c = parameters[1]
    return -2 * (b - x) / (c ** 2) * np.power(1 + ((b - x) / c) ** 2, -2)

@jit(nopython=True)
def derivCauchyC(parameters, x):
    '''
        This function returns the values of the derivative of the Cauchy function parametrized by the coordinates of the array
        "parameters" with respect to the scale parameter (i.e parameters[1]) on the discretization points "x".

        Parameters
        ----------
        parameters : array, shape(d=2,)
            parameters of the Cauchy function. The first coordinate corresponds to the location, the second coordinate
                corresponds to the scale.

        x : float or array of shape(p,)
            discretization point or table of discretization points.

        Returns
        -------
        array, shape (p,)
            values of the derivative of the Cauchy function parametrized by the coordinates of the array
            "parameters" with respect to the scale parameter (i.e parameters[1]) on the discretization points "x".
        '''
    b = parameters[0]
    c = parameters[1]
    return 2 * (((x - b) ** 2) / (c ** 3)) * np.power(1 + ((b - x) / c) ** 2, -2)

@jit(nopython=True)
def expo(parameters, x):
    '''
    This function returns the values of the Gaussian function parametrized by the the coordinates of the array
    "parameters" on the discretization points "x".

    Parameters
    ----------
    parameters : array, shape(d=2,)
        parameters of the parametric function. The first coordinate corresponds to the mean, the second coordinate
            corresponds to the standard deviation.
    x : float or array of shape(p,)
        discretization point or table of discretization points.

    Returns
    -------
    array, shape (p,)
        values of the Gaussian function parametrized by the the coordinates of the array
        "parameters" on the discretization points "x".
    '''
    b = parameters[0]
    c = parameters[1]
    return np.exp(-1 * ((x - b) ** 2) / c ** 2)

@jit(nopython=True)
def derivExpoB(parameters, x):
    '''
        This function returns the values of the derivative of the Gaussian function parametrized by the the coordinates of the array
        "parameters" with respect to the mean (i.e parameters[0]) on the discretization points "x".

        Parameters
        ----------
        parameters : array, shape(d=2,)
            parameters of the parametric function. The first coordinate corresponds to the mean, the second coordinate
                corresponds to the standard deviation.

        x : float or array of shape(p,)
            discretization point or table of discretization points.

        Returns
        -------
        array, shape (p,)
            values of the derivative of the Gaussian function parametrized by the the coordinates of the array
            "parameters" with respect to the mean  (i.e parameters[0]) on the discretization points "x".
        '''
    b = parameters[0]
    c = parameters[1]
    return 2 * (x - b) / (c ** 2) * np.exp(-1 * ((x - b) ** 2) / (c**2))


@jit(nopython=True)
def derivExpoC(parameters, x):
    '''
            This function returns the values of the derivative of the Gaussian function parametrized by the table
            "parameters" with respect to the standard deviation (i.e parameters[1]) on the discretization points "x".

            Parameters
            ----------
            parameters : array, shape(d=2,)
                parameters of the parametric function. The first coordinate corresponds to the mean, the second coordinate
                    corresponds to the standard deviation.

            x : float or array of shape(p,)
                discretization point or table of discretization points.

            Returns
            -------
            array, shape (p,)
                values of the derivative of the Gaussian function parametrized by the the coordinates of the array
                "parameters" with respect to the mean  (i.e parameters[1]) on the discretization points "x".
            '''
    b = parameters[0]
    c = parameters[1]
    return 2 *(((x - b)**2) / (c**3)) * np.exp(-1 * ((x - b)**2) / (c**2))

@jit(nopython=True)
def derivExpo(index_parameter, parameters, x):
    '''
                This function returns the values of the derivative of the Gaussian function parametrized by the table
                "parameters" with respect  to "parameters[index_parameter]" on the discretization points "x".

                Parameters
                ----------
                parameters : array, shape(d=2,)
                    parameters of the parametric function. The first coordinate corresponds to the mean, the second coordinate
                        corresponds to the standard deviation.

                x : float or array of shape(p,)
                    discretization point or table of discretization points.

                Returns
                -------
                array, shape (p,)
                    values of the derivative of the Gaussian function parametrized by the the coordinates of the array
                    "parameters" with respect to "parameters[index_parameter]" on the discretization points "x".
                '''
    if index_parameter == 0:
        res = derivExpoB(parameters, x)
    if index_parameter == 1:
        res = derivExpoC(parameters, x)
    return res

@jit(nopython=True)
def derivCauchy(index_parameter, parameters, x):
    '''
                    This function returns the values of the derivative of the Cauchy function parametrized by the the coordinates of the array
                    "parameters" with respect  to "parameters[index_parameter]" on the discretization points "x".

                    Parameters
                    ----------
                    parameters : array, shape(d=2,)
                        parameters of the parametric function. The first coordinate corresponds to the location, the second coordinate
                            corresponds to the scale.

                    x : float or array of shape(p,)
                        discretization point or table of discretization points.

                    Returns
                    -------
                    array, shape (p,)
                        values of the derivative of the Cauchy function parametrized by the the coordinates of the array
                        "parameters" with respect to "parameters[index_parameter]" on the discretization points "x".
                    '''
    if index_parameter == 0:
        res = derivCauchyB(parameters, x)
    if index_parameter == 1:
        res = derivCauchyC(parameters, x)
    return res



