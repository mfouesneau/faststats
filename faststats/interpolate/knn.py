import numpy as np
from scipy.spatial import cKDTree
from .bary import RbfInterpolator


class KDTreeInterpolator(object):
    """
    KDTreeInterpolator(points, values)

    Nearest-neighbours (barycentric) interpolation in N dimensions.

    This interpolator uses a KDTree to find the closest neighbours of a ND point
    and returns a barycentric interpolation (uses .bary.RbfInterpolator)

    if the number of neighbours is 1 or if the distance to the closest match is
    smaller than `eps`, the value of this point is returned instead.

    Parameters
    ----------
    points : (Npoints, Ndims) ndarray of floats
        Data point coordinates.

    values : (Npoints,) ndarray of float or complex
        Data values.

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``
    """

    def __init__(self, x, y):
        self.points = np.asarray(x)
        npoints, ndim = x.shape
        self.npoints = npoints
        self.ndim = ndim
        self.values = np.asarray(y)
        if npoints != len(self.values):
            raise ValueError('different number of points in x and y')
        self.tree = cKDTree(x)

    def __call__(self, *args, **kwargs):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        k : integer
            The number of nearest neighbors to use.

        eps : non-negative float

            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.

        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance

        """
        xi = np.squeeze(np.asarray(args))
        s = xi.shape
        if s[1] != self.ndim:
            raise AttributeError('Points must have {0:d} dimensions, found {1:d}.'.format(self.ndim, s[1]))

        k = kwargs.get('k', 1)
        eps = kwargs.get('eps', 0)
        p = kwargs.get('p', 2)

        dist, i = self.tree.query(xi, k=k, eps=eps)

        if k <= 1:
            return self.values[i]
        else:
            pts = self.points
            val = self.values
            p = []
            for xik, ik in zip(xi, i):
                try:
                    r = self._NDInterp(pts[ik], val[ik], xik)
                    p.append(r)
                except:
                    p.append(np.asarray(self.values[ik[0]]))
            return np.squeeze(np.asarray(p))

    def _NDInterp(self, X, Y, x):
        rb = RbfInterpolator(*( (X.T).tolist() + [Y]))
        self._rb = rb
        return rb(*x)
