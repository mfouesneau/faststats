"""
Rbf - Radial basis functions for interpolation/smoothing scattered Nd data.  A
radial basis function (RBF) is a real-valued function whose value depends only
on the distance from the origin.


Rbf are typically used to build up function approximations of the form:
    $ y(x) = \sum_{i=1}^N w_i \, \phi(||x - x_i||) $,

where the approximating function $y(x)$ is represented as a sum of $N$ Rbfs,
each associated with a different center $x_i$, and weighted by an appropriate
coefficient $w_i$. The weights $w_i$ can be estimated using the matrix methods
of Weighted (linear) least squares, given that the approximating function is
''linear'' in the weights.

It can be shown that any continuous function on a compact interval can in
principle be interpolated with arbitrary accuracy by a sum of this form, if a
sufficiently large number N of radial basis functions is used.

You can consider this approximation function $y(x)$ as a Nd-barycentric
interpolation


A rather simple neural-network
------------------------------

This approximation process can also be interpreted as a rather simple
single-layer type of artificial neural network called a "radial basis function
network", with the radial basis functions taking on the role of the activation
functions of the network. [Park]

The approximation $y(x)$ is differentiable with respect to the weights $wi$. The
weights could thus be learned using any of the standard iterative methods for
neural networks.  Using radial basis functions in this manner yields a
reasonable interpolation approach provided that the fitting set has been chosen
such that it covers the entire range systematically (equidistant data points
are ideal).

However, estimates outside the fitting set tend to perform poorly.



references
----------

wikipedia: http://en.wikipedia.org/wiki/Radial_basis_function
Buhmann: Buhmann, Martin D. (2003), "Radial Basis Functions: Theory and
         Implementations", Cambridge University Press, ISBN 978-0-521-63338-3.
Park: Park, J., Sandberg, I. W., "Universal Approximation Using Radial-Basis-Function Networks"
         http://www.ise.ncsu.edu/fangroup/ie789.dir/Park.pdf

Fornberg: Bengt Fornberg, Julia Zuev, "The runge phenomenon and spatially variable shape
        parameters in rbf interpolation"
        http://amath.colorado.edu/faculty/fornberg/Docs/fz_var_eps.pdf
"""
from scipy.lib.six import get_function_code
import numpy as np


__all__ = [ 'RbfInterpolator', 'cubic', 'euclidean_norm', 'gaussian',
            'inverse_multiquadric', 'linear', 'multiquadric', 'polyharmonic',
            'thin_plate']


#distance functions
def euclidean_norm(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum(axis=0))


#Radial basic functions
def multiquadric(r, epsilon=1.):
    return np.sqrt((1.0 / epsilon * r) ** 2 + 1)


def inverse_multiquadric(r, epsilon=1.):
    return 1.0 / multiquadric(r, epsilon)


def gaussian(r, epsilon=1.):
    return np.exp(-(1.0 / epsilon * r) ** 2)


def linear(r):
    return r


def cubic(r):
    return r ** 3


def polyharmonic(r, k=5):
    if (k % 2) == 0:
        return r ** k * np.log(r)
    else:
        return r ** k


def thin_plate(r):
    result = r ** 2 * np.log(r)
    result[r == 0] = 0  # the spline is zero at zero
    return result


class RbfInterpolator(object):
    """
    Rbf(*args)

    A class for radial basis function approximation/interpolation of
    n-dimensional scattered data.

    Parameters
    ----------
    *args : sequence of ndarrays
        x1, x2, ... xn, y, where xi are the coordinates of the sampling points
        and y is the array of values at the nodes

    function : callable, optional
        The radial basis function (Rbf). (default: 'multiquadric')

        Using any callable as radial function is possible.
        The function must take 1 argument (radii) and the epsilon
        parameter will be given if a keyword of the same name is defined.

        Other keyword arguments (**kwargs) will also be forwarded.

    epsilon : float, optional
        Adjustable constant for gaussian or multiquadrics functions
        defaults: mean(radii)

    smooth : float, optional
        Values greater than zero increase the smoothness of the approximation.
        (default: 0, function will go through all nodal points)

    norm : callable, optional
        A distance function between 2 vectors (xi, xj)
        (default: euclidean_norm)

    Example Usage
    -------------
    fitting a Gaussian in 3D

    >>> x = np.arange(-1, 1, 0.1)
    >>> y = np.arange(-1, 1, 0.1)
    >>> X = np.asarray([ k for k in np.nditer(np.ix_(x, y))])
    >>> Y = Y = np.exp(-0.5 * (X ** 2).sum(axis=1))
    >>> rb = RbfInterpolator(X[:, 0], X[:, 1], Y)  # radial basis function interpolator instance
    >>> Yn = rb(X[:, 0], X[:, 1])      # interpolated values
    >>> ((Yn - Y) / Y).ptp() < 1e-5
    """
    def __init__(self, *args, **kwargs):
        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten() for a in args[:-1]])
        self.N = self.xi.shape[-1]
        self.di = np.asarray(args[-1]).flatten()

        if self.xi.shape[1] != self.di.size:
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', euclidean_norm)
        self.epsilon = kwargs.pop('epsilon', None)
        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', multiquadric)
        self._fn_eps = 'epsilon' in get_function_code(self.function).co_varnames

        r = self.distance_matrix(self.xi, self.xi)

        if self._fn_eps & (self.epsilon is None):
            self.epsilon = r.mean()

        self._extra_kwargs = kwargs   # in case it as any use when calling function

        self.A = self.phi(r)
        if self.smooth > 0:
            self.A -= np.eye(self.N) * self.smooth

        self.nodes = np.linalg.solve(self.A, self.di)

    def phi(self, r, **kwargs):
        if self._fn_eps:
            kwargs['epsilon'] = self.epsilon
        if len(self._extra_kwargs) > 0:
            kwargs.update(kwargs, **self._extra_kwargs)
        return self.function(r, **kwargs)

    def update_yi(self, yi):
        """
        allow a quick update of the node values without re-generating a full
        object
        """
        di = np.asarray(yi).flatten()
        if self.xi.shape[1] != di.size:
            raise ValueError("All arrays must be equal length.")
        self.di = di
        self.nodes = np.linalg.solve(self.A, self.di)

    def distance_matrix(self, x1, x2):
        """ array of distance
        parameters
        ----------
        x1: ndarray
            array of points

        x2: ndarray
            array of points

        returns
        -------
        d: ndarray, shape(len(x1), len(x2)
            matrix of the pairwise-distances from each point in x1 to each
            point in x2.
        """
        if len(x1.shape) == 1:
            x1 = x1[np.newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[np.newaxis, :]
        x1 = x1[..., :, np.newaxis]
        x2 = x2[..., np.newaxis, :]
        return self.norm(x1, x2)

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]

        if any([x.shape != y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")

        shp = args[0].shape

        self.xa = np.asarray([a.flatten() for a in args], dtype=np.float_)

        r = self.distance_matrix(self.xa, self.xi)
        return np.dot(self.phi(r), self.nodes).reshape(shp)
