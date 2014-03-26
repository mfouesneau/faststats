""" a bit faster math operations when knowing what you're doing"""
import numpy as np
from scipy import linalg


def dot(A,B):
    """
    Dot product of two arrays that directly calls blas libraries

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    arrays to inner product of vectors (without complex conjugation). For N
    dimensions it is a sum product over the last axis of `a` and the
    second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])


    Parameters
    ----------
    A : array_like
        First argument.

    B : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.


    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    """
    def _force_forder(x):
        """
        Converts arrays x to fortran order. Returns
        a tuple in the form (x, is_transposed).
        """
        if x.flags.c_contiguous:
            return (x.T, True)
        else:
            return (x, False)

    A, trans_a = _force_forder(A)
    B, trans_b = _force_forder(B)
    gemm_dot = linalg.get_blas_funcs("gemm", arrays=(A,B))

    # gemm is implemented to compute: C = alpha*AB  + beta*C
    return gemm_dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)


def percentile(data, percentiles, weights=None):
    """Compute weighted percentiles.

    If the weights are equal, this is the same as normal percentiles.
    Elements of the data and wt arrays correspond to each other and must have
    equal length.
    If wt is None, this function calls numpy's percentile instead (faster)

    TODO: re-implementing the normal percentile could be faster
          because it would avoid more variable checks and overheads

    Parameters
    ----------
    data: ndarray[float, ndim=1]
        data points

    percentiles: ndarray[float, ndim=1]
        percentiles to use. (between 0 and 100)

    weights: ndarray[float, ndim=1] or None
        Weights of each point in data
        All the weights must be non-negative and the sum must be
        greater than zero.

    Returns
    -------
    p: ndarray[float, ndim=1]
        the weighted percentiles of the data.


    percentile
    ----------

    A percentile is the value of a variable below which a certain percent of
    observations fall.
    The term percentile and the related term percentile rank are often used in
    the reporting of scores from *normal-referenced tests*, 16th and 84th
    percentiles corresponding to the 1-sigma interval of a Normal distribution.

    Note that there are very common percentiles values:
        * 0th   = minimum value
        * 50th  = median value
        * 100th = maximum value


    Weighted percentile
    -------------------

    A weighted percentile where the percentage in the total weight is counted
    instead of the total number. *There is no standard function* for a weighted
    percentile.

    Implementation
    --------------

    The method implemented here extends the commom percentile estimation method
    (linear interpolation beteeen closest ranks) approach in a natural way.
    Suppose we have positive weights, W= [W_i], associated, respectively, with
    our N sorted sample values, D=[d_i]. Let S_n = Sum_i=0..n {w_i} the
    the n-th partial sum of the weights. Then the n-th percentile value is
    given by the interpolation between its closest values v_k, v_{k+1}:

        v = v_k + (p - p_k) / (p_{k+1} - p_k) * (v_{k+1} - v_k)

    where
        p_n = 100/S_n * (S_n - w_n/2)
    """
    # check if actually weighted percentiles is needed
    if weights is None:
        return np.percentile(data, list(percentiles))
    if np.equal(weights, 1.).all():
        return np.percentile(data, list(percentiles))

    # make sure percentiles are fractions between 0 and 1
    if not np.greater_equal(percentiles, 0.0).all():
        raise ValueError("Percentiles less than 0")
    if not np.less_equal(percentiles, 100.0).all():
        raise ValueError("Percentiles greater than 100")

    #Make sure data is in correct shape
    shape = np.shape(data)
    n = len(data)
    if (len(shape) != 1):
        raise ValueError("wrong data shape, expecting 1d")

    if len(weights) != n:
        raise ValueError("weights must be the same shape as data")
    if not np.greater_equal(weights, 0.0).all():
        raise ValueError("Not all weights are non-negative.")

    _data = np.asarray(data, dtype=float)

    if hasattr(percentiles, '__iter__'):
        _p = np.asarray(percentiles, dtype=float) * 0.01
    else:
        _p = np.asarray([percentiles * 0.01], dtype=float)

    _wt = np.asarray(weights, dtype=float)

    len_p = len(_p)
    sd = np.empty(n, dtype=float)
    sw = np.empty(n, dtype=float)
    aw = np.empty(n, dtype=float)
    o = np.empty(len_p, dtype=float)

    i = np.argsort(_data)
    np.take(_data, i, axis=0, out=sd)
    np.take(_wt, i, axis=0, out=sw)
    np.add.accumulate(sw, out=aw)

    if not aw[-1] > 0:
        raise ValueError("Nonpositive weight sum")

    w = (aw - 0.5 * sw) / aw[-1]

    spots = np.searchsorted(w, _p)
    for (pk, s, p) in zip(range(len_p), spots, _p):
        if s == 0:
            o[pk] = sd[0]
        elif s == n:
            o[pk] = sd[n - 1]
        else:
            f1 = (w[s] - p) / (w[s] - w[s - 1])
            f2 = (p - w[s - 1]) / (w[s] - w[s - 1])
            assert (f1 >= 0) and (f2 >= 0) and (f1 <= 1 ) and (f2 <= 1)
            assert abs(f1 + f2 - 1.0) < 1e-6
            o[pk] = sd[s - 1] * f1 + sd[s] * f2
    return o
