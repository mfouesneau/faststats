
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getenv('HOME') + '/bin/python/libs')
# just in case notebook was not launched with the option
#%pylab inline

import pylab as plt
import numpy as np
from scipy import sparse
from matplotlib.mlab import griddata
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

try:
    import faststats
except ImportError:
    faststats = None


#===============================================================================
#============== FIGURE SETUP FUNCTIONS =========================================
#===============================================================================
def theme(ax=None, minorticks=False):
    """ update plot to make it nice and uniform """
    from matplotlib.ticker import AutoMinorLocator
    from pylab import rcParams, gca, tick_params
    if minorticks:
        if ax is None:
            ax = gca()
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    tick_params(which='both', width=rcParams['lines.linewidth'])


def steppify(x, y):
    """ Steppify a curve (x,y). Useful for manually filling histograms """
    dx = 0.5 * (x[1:] + x[:-1])
    xx = np.zeros( 2 * len(dx), dtype=float)
    yy = np.zeros( 2 * len(y), dtype=float)
    xx[0::2], xx[1::2] = dx, dx
    yy[0::2], yy[1::2] = y, y
    xx = np.concatenate(([x[0] - (dx[0] - x[0])], xx, [x[-1] + (x[-1] - dx[-1])]))
    return xx, yy


def colorify(data, vmin=None, vmax=None, cmap=plt.cm.Spectral):
    """ Associate a color map to a quantity vector """
    import matplotlib.colors as colors

    _vmin = vmin or min(data)
    _vmax = vmax or max(data)
    cNorm = colors.normalize(vmin=_vmin, vmax=_vmax)

    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = map(scalarMap.to_rgba, data)
    return colors, scalarMap


def hist_with_err(x, xerr, bins=None, normed=False, step=False, *kwargs):
    from scipy import integrate

    #check inputs
    assert( len(x) == len(xerr) ), 'data size mismatch'
    _x = np.asarray(x).astype(float)
    _xerr = np.asarray(xerr).astype(float)

    #def the evaluation points
    if (bins is None) | (not hasattr(bins, '__iter__')):
        m = (_x - _xerr).min()
        M = (_x + _xerr).max()
        dx = M - m
        m -= 0.2 * dx
        M += 0.2 * dx
        if bins is not None:
            N = int(bins)
        else:
            N = 10
        _xp = np.linspace(m, M, N)
    else:
        _xp = 0.5 * (bins[1:] + bins[:-1])

    def normal(v, mu, sig):
        norm_pdf = 1. / (np.sqrt(2. * np.pi) * sig ) * np.exp( - ( (v - mu ) / (2. * sig) ) ** 2 )
        return norm_pdf / integrate.simps(norm_pdf, v)

    _yp = np.array([normal(_xp, xk, xerrk) for xk, xerrk in zip(_x, _xerr) ]).sum(axis=0)

    if normed:
        _yp /= integrate.simps(_yp, _xp)

    if step:
        return steppify(_xp, _yp)
    else:
        return _xp, _yp


def hist_with_err_bootstrap(x, xerr, bins=None, normed=False, nsample=50, step=False, **kwargs):
    x0, y0 = hist_with_err(x, xerr, bins=bins, normed=normed, step=step, **kwargs)

    yn = np.empty( (nsample, len(y0)), dtype=float)
    yn[0, :] = y0
    for k in range(nsample - 1):
        idx = np.random.randint(0, len(x), len(x))
        yn[k, :] = hist_with_err(x[idx], xerr[idx], bins=bins, normed=normed, step=step, **kwargs)[1]

    return x0, yn


def __get_hesse_bins__(_x, _xerr=0., bins=None, margin=0.2):
    if (bins is None) | (not hasattr(bins, '__iter__')):
        m = (_x - _xerr).min()
        M = (_x + _xerr).max()
        dx = M - m
        m -= margin * dx
        M += margin * dx
        if bins is not None:
            N = int(bins)
        else:
            N = 10
        _xp = np.linspace(m, M, N)
    else:
        _xp = 0.5 * (bins[1:] + bins[:-1])
    return _xp


def scatter_contour(x, y,
                    levels=10,
                    bins=40,
                    threshold=50,
                    log_counts=False,
                    histogram2d_args={},
                    plot_args={},
                    contour_args={},
                    ax=None):
    """Scatter plot with contour over dense regions

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    plot_args : dict
        keyword arguments passed to pylab.scatter
        see doc string of pylab.scatter for more information
    contourf_args : dict
        keyword arguments passed to pylab.contourf
        see doc string of pylab.contourf for more information
    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used
    """
    if ax is None:
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)

    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the filled contour below.
    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent)
    try:
        outer_poly = outline.allsegs[0][0]

        ax.contourf(H.T, levels, extent=extent, **contour_args)
        X = np.hstack([x[:, None], y[:, None]])

        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]

        ax.plot(Xplot[:, 0], Xplot[:, 1], zorder=1, **plot_args)
    except IndexError:
        ax.plot(x, y, zorder=1, **plot_args)


def latex_float(f, precision=0.2, delimiter=r'\times'):
    float_str = ("{0:" + str(precision) + "g}").format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return (r"{0}" + delimiter + "10^{{{1}}}").format(base, int(exponent))
    else:
        return float_str


#===============================================================================
#===============================================================================
#===============================================================================

def ezrc(fontSize=22., lineWidth=2., labelSize=None, tickmajorsize=10, tickminorsize=5):
    """
    slides - Define params to make pretty fig for slides
    """
    from pylab import rc, rcParams
    if labelSize is None:
        labelSize = fontSize + 5
    rc('figure', figsize=(8, 6))
    rc('lines', linewidth=lineWidth)
    rcParams['grid.linewidth'] = lineWidth
    rc('font', size=fontSize, family='serif', weight='normal')
    rc('axes', linewidth=lineWidth, labelsize=labelSize)
    #rc('xtick', width=2.)
    #rc('ytick', width=2.)
    #rc('legend', fontsize='x-small', borderpad=0.1, markerscale=1.,
    rc('legend', borderpad=0.1, markerscale=1., fancybox=False)
    rc('text', usetex=True)
    rc('image', aspect='auto')
    rc('ps', useafm=True, fonttype=3)
    rcParams['xtick.major.size'] = tickmajorsize
    rcParams['xtick.minor.size'] = tickminorsize
    rcParams['ytick.major.size'] = tickmajorsize
    rcParams['ytick.minor.size'] = tickminorsize
    rcParams['font.sans-serif'] = 'Helvetica'
    rcParams['font.serif'] = 'Helvetica'
    #rcParams['text.latex.preamble'] = '\usepackage{pslatex}'


def hide_axis(where, ax=None):
    ax = ax or plt.gca()
    if type(where) == str:
        _w = [where]
    else:
        _w = where
    [sk.set_color('None') for k, sk in ax.spines.items() if k in _w ]

    if 'top' in _w and 'bottom' in _w:
        ax.xaxis.set_ticks_position('none')
    elif 'top' in _w:
        ax.xaxis.set_ticks_position('bottom')
    elif 'bottom' in _w:
        ax.xaxis.set_ticks_position('top')

    if 'left' in _w and 'right' in _w:
        ax.yaxis.set_ticks_position('none')
    elif 'left' in _w:
        ax.yaxis.set_ticks_position('right')
    elif 'right' in _w:
        ax.yaxis.set_ticks_position('left')

    plt.draw_if_interactive()


def despine(fig=None, ax=None, top=True, right=True,
            left=False, bottom=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure
        figure to despine all axes of, default uses current figure
    ax : matplotlib axes
        specific axes object to despine
    top, right, left, bottom : boolean
        if True, remove that spine

    """
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            ax_i.spines[side].set_visible(not locals()[side])


def shift_axis(which, delta, where='outward', ax=None):
    ax = ax or plt.gca()
    if type(which) == str:
        _w = [which]
    else:
        _w = which

    scales = (ax.xaxis.get_scale(), ax.yaxis.get_scale())
    lbls = (ax.xaxis.get_label(), ax.yaxis.get_label())

    for wk in _w:
        ax.spines[wk].set_position((where, delta))

    ax.xaxis.set_scale(scales[0])
    ax.yaxis.set_scale(scales[1])
    ax.xaxis.set_label(lbls[0])
    ax.yaxis.set_label(lbls[1])
    plt.draw_if_interactive()


class AutoLocator(MaxNLocator):
    def __init__(self, nbins=9, steps=[1, 2, 5, 10], **kwargs):
        MaxNLocator.__init__(self, nbins=nbins, steps=steps, **kwargs )


def setMargins(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        """
        Tune the subplot layout via the meanings (and suggested defaults) are::

            left  = 0.125  # the left side of the subplots of the figure
            right = 0.9    # the right side of the subplots of the figure
            bottom = 0.1   # the bottom of the subplots of the figure
            top = 0.9      # the top of the subplots of the figure
            wspace = 0.2   # the amount of width reserved for blank space between subplots
            hspace = 0.2   # the amount of height reserved for white space between subplots

        The actual defaults are controlled by the rc file

        """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.draw_if_interactive()


def setNmajors(xval=None, yval=None, ax=None, mode='auto', **kwargs):
        """
        setNmajors - set major tick number
        see figure.MaxNLocator for kwargs
        """
        if ax is None:
                ax = plt.gca()
        if (mode == 'fixed'):
                if xval is not None:
                        ax.xaxis.set_major_locator(MaxNLocator(xval, **kwargs))
                if yval is not None:
                        ax.yaxis.set_major_locator(MaxNLocator(yval, **kwargs))
        elif (mode == 'auto'):
                if xval is not None:
                        ax.xaxis.set_major_locator(AutoLocator(xval, **kwargs))
                if yval is not None:
                        ax.yaxis.set_major_locator(AutoLocator(yval, **kwargs))

        plt.draw_if_interactive()


def crazy_histogram2d(x, y, bins=10, weights=None, reduce_w=None, NULL=None, reinterp=None):
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

    INPUTS:
        x       ndarray[ndim=1]         first data sample coordinates
        y       ndarray[ndim=1]         second data sample coordinates

    KEYWORDS:
        bins                            the bin specification
                   int                     the number of bins for the two dimensions (nx=ny=bins)
                or [int, int]              the number of bins in each dimension (nx, ny = bins)
        weights     ndarray[ndim=1]     values *w_i* weighing each sample *(x_i, y_i)*
                                        accumulated and reduced (using reduced_w) per bin
        reduce_w    callable            function that will reduce the *weights* values accumulated per bin
                                        defaults to numpy's sum function (np.sum)
        NULL        value type          filling missing data value
        reinterp    str                 values are [None, 'nn', linear']
                                        if set, reinterpolation is made using mlab.griddata to fill missing data
                                        within the convex polygone that encloses the data

    OUTPUTS:
        B           ndarray[ndim=2]     bi-dimensional histogram
        extent      tuple(4)            (xmin, xmax, ymin, ymax) entension of the histogram
        steps       tuple(2)            (dx, dy) bin size in x and y direction

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


def histplot(data, bins=10, range=None, normed=False, weights=None, density=None, ax=None, **kwargs):
    """ plot an histogram of data `a la R`: only bottom and left axis, with
    dots at the bottom to represent the sample

    Example
    -------
        import numpy as np
        x = np.random.normal(0, 1, 1e3)
        histplot(x, bins=50, density=True, ls='steps-mid')
    """
    h, b = np.histogram(data, bins, range, normed, weights, density)
    if ax is None:
        ax = plt.gca()
    x = 0.5 * (b[:-1] + b[1:])
    l = ax.plot(x, h, **kwargs)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    _w = ['top', 'right']
    [ ax.spines[side].set_visible(False) for side in _w ]

    for wk in ['bottom', 'left']:
        ax.spines[wk].set_position(('outward', 10))

    ylim = ax.get_ylim()
    ax.plot(data, -0.02 * max(ylim) * np.ones(len(data)), '|', color=l[0].get_color())
    ax.set_ylim(-0.02 * max(ylim), max(ylim))


def scatter_plot(x, y, ellipse=False, levels=[0.99, 0.95, 0.68], color='w', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if faststats is not None:
        im, e = faststats.fastkde.fastkde(x, y, (50, 50), adjust=2.)
        V = im.max() * np.asarray(levels)

        plt.contour(im.T, levels=V, origin='lower', extent=e, linewidths=[1, 2, 3], colors=color)

    ax.plot(x, y, 'b,', alpha=0.3, zorder=-1, rasterized=True)

    if ellipse is True:
        data = np.vstack([x, y])
        mu = np.mean(data, axis=1)
        cov = np.cov(data)
        error_ellipse(mu, cov, ax=plt.gca(), edgecolor="g", ls="dashed", lw=4, zorder=2)


def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot


def plotCorr(l, pars, plotfunc=None, lbls=None, triangle='lower', *args, **kwargs):
        """ Plot correlation matrix between variables
        inputs
        -------
        l: dict
            dictionary of variables (could be a Table)

        pars: sequence of str
            parameters to use

        plotfunc: callable
            function to be called when doing the scatter plots

        lbls: sequence of str
            sequence of string to use instead of dictionary keys

        triangle: str in ['upper', 'lower']
            Which side of the triangle to use.

        *args, **kwargs are forwarded to the plot function

        Example
        -------
            import numpy as np
            figrc.ezrc(16, 1, 16, 5)

            d = {}

            for k in range(4):
                d[k] = np.random.normal(0, k+1, 1e4)

            plt.figure(figsize=(8 * 1.5, 7 * 1.5))
            plotCorr(d, d.keys(), plotfunc=figrc.scatter_plot)
            #plotCorr(d, d.keys(), alpha=0.2)
        """

        if lbls is None:
                lbls = pars

        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}
        if not len(pars) - 1 in fontmap:
                fontmap[len(pars) - 1] = 3

        k = 1
        axes = np.empty((len(pars) + 1, len(pars)), dtype=object)
        for j in range(len(pars)):
                for i in range(len(pars)):
                        if j > i:
                                sharex = axes[j - 1, i]
                        else:
                                sharex = None

                        if i == j:
                            # Plot the histograms.
                            ax = plt.subplot(len(pars), len(pars), k)
                            axes[j, i] = ax
                            n, b, p = ax.hist(l[pars[i]], bins=50, histtype="step", color=kwargs.get("color", "b"))
                            if triangle == 'upper':
                                ax.set_xlabel(lbls[i])
                                ax.set_ylabel(lbls[i])
                                ax.xaxis.set_ticks_position('bottom')
                                ax.yaxis.set_ticks_position('left')
                            else:
                                ax.yaxis.set_ticks_position('right')
                                ax.xaxis.set_ticks_position('bottom')

                        if triangle == 'upper':
                            if i > j:

                                if i > j + 1:
                                        sharey = axes[j, i - 1]
                                else:
                                        sharey = None

                                ax = plt.subplot(len(pars), len(pars), k, sharey=sharey, sharex=sharex)
                                axes[j, i] = ax
                                if plotfunc is None:
                                        plt.plot(l[pars[i]], l[pars[j]], ',', **kwargs)
                                else:
                                        plotfunc(l[pars[i]], l[pars[j]], ax=ax, *args, **kwargs)

                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)

                        if triangle == 'lower':
                            if i < j:

                                if i < j:
                                        sharey = axes[j, i - 1]
                                else:
                                        sharey = None

                                ax = plt.subplot(len(pars), len(pars), k, sharey=sharey, sharex=sharex)
                                axes[j, i] = ax
                                if plotfunc is None:
                                        plt.plot(l[pars[i]], l[pars[j]], ',', **kwargs)
                                else:
                                        plotfunc(l[pars[i]], l[pars[j]], ax=ax, *args, **kwargs)

                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)

                            if i == 0:
                                ax.set_ylabel(lbls[j])
                                plt.setp(ax.get_yticklabels(), visible=True)

                            if j == len(pars) - 1:
                                ax.set_xlabel(lbls[i])
                                plt.setp(ax.get_xticklabels(), visible=True)

                        N = int(0.5 * fontmap[len(pars) - 1])
                        if N <= 4:
                            N = 5
                        setNmajors(N, N, ax=ax, prune='both')

                        k += 1

        setMargins(hspace=0.0, wspace=0.0)


def hinton(W, bg='grey', facecolors=('w', 'k')):
    """Draw a hinton diagram of the matrix W on the current pylab axis

    Hinton diagrams are a way of visualizing numerical values in a matrix/vector,
    popular in the neural networks and machine learning literature. The area
    occupied by a square is proportional to a value's magnitude, and the colour
    indicates its sign (positive/negative).

    Example usage:

        R = np.random.normal(0, 1, (2,1000))
        h, ex, ey = np.histogram2d(R[0], R[1], bins=15)
        hh = h - h.T
        hinton.hinton(hh)
    """
    M, N = W.shape
    square_x = np.array([-.5, .5, .5, -.5])
    square_y = np.array([-.5, -.5, .5, .5])

    ioff = False
    if plt.isinteractive():
        plt.ioff()
        ioff = True

    plt.fill([-.5, N - .5, N - .5, - .5], [-.5, -.5, M - .5, M - .5], bg)
    Wmax = np.abs(W).max()
    for m, Wrow in enumerate(W):
        for n, w in enumerate(Wrow):
            c = plt.signbit(w) and facecolors[1] or facecolors[0]
            plt.fill(square_x * w / Wmax + n, square_y * w / Wmax + m, c, edgecolor=c)

    plt.ylim(-0.5, M - 0.5)
    plt.xlim(-0.5, M - 0.5)

    if ioff is True:
        plt.ion()

    plt.draw_if_interactive()
