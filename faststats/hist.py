import numpy as np
from scipy import sparse
from matplotlib.mlab import griddata


def fast_histogram2d(x, y, bins=10, weights=None, reduce_w=None, NULL=None, reinterp=None):
    """
    Compute the sparse bi-dimensional histogram of two data samples where *x*,
    and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).

    If *weights* is specified, it specifies values at the coordinate (x[i],
    y[i]). These values are accumulated for each bin and then reduced according
    to *reduce_w* function, which defaults to numpy's sum function (np.sum).
    (If *weights* is specified, it must also be a 1-D sequence of the same
    length as *x* and *y*.)

    INPUTS
    ------
    x: ndarray[ndim=1]
        first data sample coordinates

    y: ndarray[ndim=1]
        second data sample coordinates

    KEYWORDS
    --------
    bins: int or [int, int]
        int, the number of bins for the two dimensions (nx=ny=bins)
        or [int, int], the number of bins in each dimension (nx, ny = bins)

    weights: ndarray[ndim=1]
        values *w_i* weighing each sample *(x_i, y_i)*
                                        accumulated and reduced (using reduced_w) per bin
    reduce_w: callable
        function that will reduce the *weights* values accumulated per bin
        defaults to numpy's sum function (np.sum)

    NULL: value type
        filling missing data value

    reinterp: str
        values are [None, 'nn', linear']
        if set, reinterpolation is made using mlab.griddata to fill missing
        data within the convex polygone that encloses the data

    OUTPUTS
    -------
    B: ndarray[ndim=2]
        bi-dimensional histogram

    extent: tuple(4)
        (xmin, xmax, ymin, ymax) extension of the histogram

    steps: tuple(2)
        (dx, dy) bin size in x and y direction
    """
    # define the bins (do anything you want here but needs edges and sizes of the 2d bins)
    try:
        nx, ny = bins
    except TypeError:
        nx = ny = bins

    #values you want to be reported
    if weights is None:
        weights = np.ones(x.size)

    if reduce_w is None:
        reduce_w = np.sum
    else:
        if not hasattr(reduce_w, '__call__'):
            raise TypeError('reduce function is not callable')

    # culling nans
    finite_inds = (np.isfinite(x) & np.isfinite(y) & np.isfinite(weights))
    _x = np.asarray(x)[finite_inds]
    _y = np.asarray(y)[finite_inds]
    _w = np.asarray(weights)[finite_inds]

    if not (len(_x) == len(_y)) & (len(_y) == len(_w)):
        raise ValueError('Shape mismatch between x, y, and weights: {}, {}, {}'.format(_x.shape, _y.shape, _w.shape))

    xmin, xmax = _x.min(), _x.max()
    ymin, ymax = _y.min(), _y.max()
    dx = (xmax - xmin) / (nx - 1.0)
    dy = (ymax - ymin) / (ny - 1.0)

    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((_x, _y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    #xyi contains the bins of each point as a 2d array [(xi,yi)]

    d = {}
    for e, k in enumerate(xyi.T):
        key = (k[0], k[1])

        if key in d:
            d[key].append(_w[e])
        else:
            d[key] = [_w[e]]

    _xyi = np.array(d.keys()).T
    _w   = np.array([ reduce_w(v) for v in d.values() ])

    # exploit a sparse coo_matrix to build the 2D histogram...
    _grid = sparse.coo_matrix((_w, _xyi), shape=(nx, ny))

    if reinterp is None:
        #convert sparse to array with filled value
        ## grid.toarray() does not account for filled value
        ## sparse.coo.coo_todense() does actually add the values to the existing ones, i.e. not what we want -> brute force
        if NULL is None:
            B = _grid.toarray()
        else:  # Brute force only went needed
            B = np.zeros(_grid.shape, dtype=_grid.dtype)
            B.fill(NULL)
            for (x, y, v) in zip(_grid.col, _grid.row, _grid.data):
                B[y, x] = v
    else:  # reinterp
        xi = np.arange(nx, dtype=float)
        yi = np.arange(ny, dtype=float)
        B = griddata(_grid.col.astype(float), _grid.row.astype(float), _grid.data, xi, yi, interp=reinterp)

    return B, (xmin, xmax, ymin, ymax), (dx, dy)


def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


def optbins(data, method='freedman', ret='N'):
    """ Determine the optimal binning of the data based on common estimators
    and returns either the number of bins of the width to use.

    inputs
    ------
        data    1d dataset to estimate from

    keywords
    --------
        method  the method to use: str in {sturge, scott, freedman}
        ret set to N will return the number of bins / edges
            set to W will return the width

    refs
    ----
    * Sturges, H. A. (1926)."The choice of a class interval". J. American Statistical Association, 65-66
    * Scott, David W. (1979), "On optimal and data-based histograms". Biometrika, 66, 605-610
    * Freedman, D.; Diaconis, P. (1981). "On the histogram as a density estimator: L2 theory".
            Zeitschrift fur Wahrscheinlichkeitstheorie und verwandte Gebiete, 57, 453-476
    *Scargle, J.D. et al (2012) "Studies in Astronomical Time Series Analysis. VI. Bayesian
        Block Representations."
    """
    x = np.asarray(data)
    n = x.size
    r = x.max() - x.min()

    def sturge():
        if (n <= 30):
            print "Warning: Sturge estimator can perform poorly for small samples"
        k = int(np.log(n) + 1)
        h = r / k
        return h, k

    def scott():
        h = 3.5 * np.std(x) * float(n) ** (-1. / 3.)
        k = int(r / h)
        return h, k

    def freedman():
        q = quantiles(x, [25, 75])
        h = 2 * (q[75] - q[25]) * float(n) ** (-1. / 3.)
        k = int(r / h)
        return h, k

    def bayesian():
        r = bayesian_blocks(x)
        return np.diff(r), r

    m = {'sturge':sturge, 'scott':scott, 'freedman': freedman, 'bayesian':bayesian}

    if method.lower() in m:
        s = m[method.lower()]()
        if ret.lower() == 'n':
            return s[1]
        elif ret.lower() == 'w':
            return s[0]
    else:
        return None


def quantiles(x, qlist=[2.5, 25, 50, 75, 97.5]):
    """computes quantiles from an array

    Quantiles :=  points taken at regular intervals from the cumulative
    distribution function (CDF) of a random variable. Dividing ordered data
    into q essentially equal-sized data subsets is the motivation for
    q-quantiles; the quantiles are the data values marking the boundaries
    between consecutive subsets.

    The quantile with a fraction 50 is called the median
    (50% of the distribution)

    Inputs:
        x     - variable to evaluate from
        qlist - quantiles fraction to estimate (in %)

    Outputs:
        Returns a dictionary of requested quantiles from array
    """
    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort, then transpose back
        sx = np.transpose(np.sort(np.transpose(x)))
    else:
        # Sort univariate node
        sx = np.sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx) * q / 100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        print "Too few elements for quantile calculation"
