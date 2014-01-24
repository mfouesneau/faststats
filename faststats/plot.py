"""Variations on boxplots."""
# Based on code by Ralf Gommers, Flavio Coelho and Teemu Ikonen.

import numpy as np
import pylab as plt
import fastkde


__all__ = ['violinplot', 'beanplot', 'kde_plot']


def violinplot(data, ax=None, labels=None, positions=None, side='both',
               show_boxplot=True, plot_opts={}):
    """Make a violin plot of each dataset in the `data` sequence.

    A violin plot is a boxplot combined with a kernel density estimate of the
    probability density function per point.

    Parameters
    ----------
    data : sequence of ndarrays
        Data arrays, one array per value in `positions`.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    labels : list of str, optional
        Tick labels for the horizontal axis.  If not given, integers
        ``1..len(data)`` are used.
    positions : array_like, optional
        Position array, used as the horizontal axis of the plot.  If not given,
        spacing of the violins will be equidistant.
    side : {'both', 'left', 'right'}, optional
        How to plot the violin.  Default is 'both'.  The 'left', 'right'
        options can be used to create asymmetric violin plots.
    show_boxplot : bool, optional
        Whether or not to show normal box plots on top of the violins.
        Default is True.
    plot_opts : dict, optional
        A dictionary with plotting options.  Any of the following can be
        provided, if not present in `plot_opts` the defaults will be used::

          - 'violin_fc', MPL color.  Fill color for violins.  Default is 'y'.
          - 'violin_ec', MPL color.  Edge color for violins.  Default is 'k'.
          - 'violin_lw', scalar.  Edge linewidth for violins.  Default is 1.
          - 'violin_alpha', float.  Transparancy of violins.  Default is 0.5.
          - 'cutoff', bool.  If True, limit violin range to data range.
                Default is False.
          - 'cutoff_val', scalar.  Where to cut off violins if `cutoff` is
                True.  Default is 1.5 standard deviations.
          - 'cutoff_type', {'std', 'abs'}.  Whether cutoff value is absolute,
                or in standard deviations.  Default is 'std'.
          - 'violin_width' : float.  Relative width of violins.  Max available
                space is 1, default is 0.8.
          - 'label_fontsize', MPL fontsize.  Adjusts fontsize only if given.
          - 'label_rotation', scalar.  Adjusts label rotation only if given.
                Specify in degrees.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    beanplot : Bean plot, builds on `violinplot`.
    matplotlib.pyplot.boxplot : Standard boxplot.

    Notes
    -----
    The appearance of violins can be customized with `plot_opts`.  If
    customization of boxplot elements is required, set `show_boxplot` to False
    and plot it on top of the violins by calling the Matplotlib `boxplot`
    function directly.  For example::

        violinplot(data, ax=ax, show_boxplot=False)
        ax.boxplot(data, sym='cv', whis=2.5)

    It can happen that the axis labels or tick labels fall outside the plot
    area, especially with rotated labels on the horizontal axis.  With
    Matplotlib 1.1 or higher, this can easily be fixed by calling
    ``ax.tight_layout()``.  With older Matplotlib one has to use ``plt.rc`` or
    ``plt.rcParams`` to fix this, for example::

        plt.rc('figure.subplot', bottom=0.25)
        violinplot(data, ax=ax)

    References
    ----------
    J.L. Hintze and R.D. Nelson, "Violin Plots: A Box Plot-Density Trace
    Synergism", The American Statistician, Vol. 52, pp.181-84, 1998.

    """
    if ax is None:
        ax = plt.gca()

    if positions is None:
        positions = np.arange(len(data)) + 1

    # Determine available horizontal space for each individual violin.
    pos_span = np.max(positions) - np.min(positions)
    width = np.min([0.15 * np.max([pos_span, 1.]),
                    plot_opts.get('violin_width', 0.8) / 2.])

    # Plot violins.
    for pos_data, pos in zip(data, positions):
        xvals, violin = _single_violin(ax, pos, pos_data, width, side,
                                       plot_opts)

    if show_boxplot:
        ax.boxplot(data, notch=1, positions=positions, vert=1)

    # Set ticks and tick labels of horizontal axis.
    _set_ticks_labels(ax, data, labels, positions, plot_opts)

    plt.draw_if_interactive()
    return ax


def _single_violin(ax, pos, pos_data, width, side, plot_opts):
    """"""

    def _violin_range(pos_data, plot_opts):
        """Return array with correct range, with which violins can be plotted."""
        cutoff = plot_opts.get('cutoff', False)
        cutoff_type = plot_opts.get('cutoff_type', 'std')
        cutoff_val = plot_opts.get('cutoff_val', 1.5)

        s = 0.0
        if cutoff:
            if cutoff_type == 'std':
                s = cutoff_val * np.std(pos_data)
            else:
                s = cutoff_val

        x_lower = pos_data.min() - s
        x_upper = pos_data.max() + s
        return np.linspace(x_lower, x_upper, 100)

    pos_data = np.asarray(pos_data)

    # Create violin for pos, scaled to the available space.
    xvals = _violin_range(pos_data, plot_opts)
    # Kernel density estimate for data at this position.
    violin, e = fastkde.fastkde1D(pos_data, len(xvals), extents=[min(xvals), max(xvals)])
    xvals = np.linspace(e[0], e[1], len(violin))
    violin = width * violin / violin.max()

    if side == 'both':
        envelope_l, envelope_r = (-violin + pos, violin + pos)
    elif side == 'right':
        envelope_l, envelope_r = (pos, violin + pos)
    elif side == 'left':
        envelope_l, envelope_r = (-violin + pos, pos)
    else:
        msg = "`side` parameter should be one of {'left', 'right', 'both'}."
        raise ValueError(msg)

    # Draw the violin.
    ax.fill_betweenx(xvals, envelope_l, envelope_r,
                     facecolor=plot_opts.get('violin_fc', 'y'),
                     edgecolor=plot_opts.get('violin_ec', 'k'),
                     lw=plot_opts.get('violin_lw', 1),
                     alpha=plot_opts.get('violin_alpha', 0.5))

    return xvals, violin


def _set_ticks_labels(ax, data, labels, positions, plot_opts):
    """Set ticks and labels on horizontal axis."""

    # Set xticks and limits.
    ax.set_xlim([np.min(positions) - 0.5, np.max(positions) + 0.5])
    ax.set_xticks(positions)

    label_fontsize = plot_opts.get('label_fontsize')
    label_rotation = plot_opts.get('label_rotation')
    if label_fontsize or label_rotation:
        from matplotlib.artist import setp

    if labels is not None:
        if not len(labels) == len(data):
            msg = "Length of `labels` should equal length of `data`."
            raise(ValueError, msg)

        xticknames = ax.set_xticklabels(labels)
        if label_fontsize:
            setp(xticknames, fontsize=label_fontsize)

        if label_rotation:
            setp(xticknames, rotation=label_rotation)

    return


def beanplot(data, ax=None, labels=None, positions=None, side='both',
             jitter=False, plot_opts={}):
    """Make a bean plot of each dataset in the `data` sequence.

    A bean plot is a combination of a `violinplot` (kernel density estimate of
    the probability density function per point) with a line-scatter plot of all
    individual data points.

    Parameters
    ----------
    data : sequence of ndarrays
        Data arrays, one array per value in `positions`.
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    labels : list of str, optional
        Tick labels for the horizontal axis.  If not given, integers
        ``1..len(data)`` are used.
    positions : array_like, optional
        Position array, used as the horizontal axis of the plot.  If not given,
        spacing of the violins will be equidistant.
    side : {'both', 'left', 'right'}, optional
        How to plot the violin.  Default is 'both'.  The 'left', 'right'
        options can be used to create asymmetric violin plots.
    jitter : bool, optional
        If True, jitter markers within violin instead of plotting regular lines
        around the center.  This can be useful if the data is very dense.
    plot_opts : dict, optional
        A dictionary with plotting options.  All the options for `violinplot`
        can be specified, they will simply be passed to `violinplot`.  Options
        specific to `beanplot` are:

          - 'bean_color', MPL color.  Color of bean plot lines.  Default is 'k'.
                Also used for jitter marker edge color if `jitter` is True.
          - 'bean_size', scalar.  Line length as a fraction of maximum length.
                Default is 0.5.
          - 'bean_lw', scalar.  Linewidth, default is 0.5.
          - 'bean_show_mean', bool.  If True (default), show mean as a line.
          - 'bean_show_median', bool.  If True (default), show median as a
                marker.
          - 'bean_mean_color', MPL color.  Color of mean line.  Default is 'b'.
          - 'bean_mean_lw', scalar.  Linewidth of mean line, default is 2.
          - 'bean_median_color', MPL color.  Color of median marker.  Default
                is 'r'.
          - 'bean_median_marker', MPL marker.  Marker type, default is '+'.
          - 'jitter_marker', MPL marker.  Marker type for ``jitter=True``.
                Default is 'o'.
          - 'jitter_marker_size', int.  Marker size.  Default is 4.
          - 'jitter_fc', MPL color.  Jitter marker face color.  Default is None.
          - 'bean_legend_text', str.  If given, add a legend with given text.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    violinplot : Violin plot, also used internally in `beanplot`.
    matplotlib.pyplot.boxplot : Standard boxplot.

    References
    ----------
    P. Kampstra, "Beanplot: A Boxplot Alternative for Visual Comparison of
    Distributions", J. Stat. Soft., Vol. 28, pp. 1-9, 2008.
    """
    if ax is None:
        ax = plt.gca()

    if positions is None:
        positions = np.arange(len(data)) + 1

    # Determine available horizontal space for each individual violin.
    pos_span = np.max(positions) - np.min(positions)
    width = np.min([0.15 * np.max([pos_span, 1.]),
                    plot_opts.get('bean_size', 0.5) / 2.])

    legend_txt = plot_opts.get('bean_legend_text', None)
    for pos_data, pos in zip(data, positions):
        # Draw violins.
        xvals, violin = _single_violin(ax, pos, pos_data, width, side, plot_opts)

        if jitter:
            # Draw data points at random coordinates within violin envelope.
            jitter_coord = pos + _jitter_envelope(pos_data, xvals, violin, side)
            ax.plot(jitter_coord, pos_data, ls='',
                    marker=plot_opts.get('jitter_marker', 'o'),
                    ms=plot_opts.get('jitter_marker_size', 4),
                    mec=plot_opts.get('bean_color', 'k'),
                    mew=1, mfc=plot_opts.get('jitter_fc', 'none'),
                    label=legend_txt)
        else:
            # Draw bean lines.
            ax.hlines(pos_data, pos - width, pos + width,
                      lw=plot_opts.get('bean_lw', 0.5),
                      color=plot_opts.get('bean_color', 'k'),
                      label=legend_txt)

        # Show legend if required.
        if legend_txt is not None:
            _show_legend(ax)
            legend_txt = None  # ensure we get one entry per call to beanplot

        # Draw mean line.
        if plot_opts.get('bean_show_mean', True):
            ax.hlines(np.mean(pos_data), pos - width, pos + width,
                      lw=plot_opts.get('bean_mean_lw', 2.),
                      color=plot_opts.get('bean_mean_color', 'b'))

        # Draw median marker.
        if plot_opts.get('bean_show_median', True):
            ax.plot(pos, np.median(pos_data),
                    marker=plot_opts.get('bean_median_marker', '+'),
                    color=plot_opts.get('bean_median_color', 'r'))

    # Set ticks and tick labels of horizontal axis.
    _set_ticks_labels(ax, data, labels, positions, plot_opts)

    plt.draw_if_interactive()
    return ax


def _jitter_envelope(pos_data, xvals, violin, side):
    """Determine envelope for jitter markers."""
    if side == 'both':
        low, high = (-1., 1.)
    elif side == 'right':
        low, high = (0, 1.)
    elif side == 'left':
        low, high = (-1., 0)
    else:
        raise ValueError("`side` input incorrect: %s" % side)

    jitter_envelope = np.interp(pos_data, xvals, violin)
    jitter_coord = jitter_envelope * np.random.uniform(low=low, high=high,
                                                       size=pos_data.size)

    return jitter_coord


def _show_legend(ax):
    """Utility function to show legend."""
    leg = ax.legend(loc=1, shadow=True, fancybox=True, labelspacing=0.2,
                    borderpad=0.15)
    ltext  = leg.get_texts()
    llines = leg.get_lines()

    from matplotlib.artist import setp
    setp(ltext, fontsize='small')
    setp(llines, linewidth=1)


def kde_plot(x, ax=None, orientation='horizontal', cutoff=False, log=False, cutoff_type='std', cutoff_val=1.5, pos=100,
             pos_marker='line', pos_width=0.05, pos_kwargs={}, **kwargs):
    """"""

    if ax is None:
        ax = plt.gca()

    # Massage '_data' for processing.
    _data = np.asarray(x)

    # Create violin for pos, scaled to the available space.
    s = 0.0
    if cutoff:
        if cutoff_type == 'std':
            s = cutoff_val * np.std(_data)
        else:
            s = cutoff_val

    x_lower = x.min() - s
    x_upper = x.max() + s

    # Kernel density estimate for data at this position.
    violin, e = fastkde.fastkde1D(_data, pos, extents=[x_lower, x_upper])
    xvals = np.linspace(e[0], e[1], len(violin))
    #violin /= violin.max()

    # Draw the violin.
    if ('facecolor' not in kwargs.keys()) | ('fc' not in kwargs.keys()):
        kwargs['facecolor'] = 'y'
    if ('edgecolor' not in kwargs.keys()) | ('ec' not in kwargs.keys()):
        kwargs['edgecolor'] = 'k'
    if ('alpha' not in kwargs.keys()):
        kwargs['alpha'] = 0.5
    #draw the positions
    if not 'marker' in pos_kwargs:
        if pos_marker != 'line':
            pos_kwargs['marker'] = pos_marker
        else:
            pos_kwargs['marker'] = 's'
    else:
        pos_marker = pos_kwargs['marker']
    if ('facecolor' not in pos_kwargs.keys()) | ('fc' not in pos_kwargs.keys()) | \
       ('markerfacecolor' not in pos_kwargs.keys()) | ('mfc' not in pos_kwargs.keys()):
        pos_kwargs['markerfacecolor'] = 'None'
    if ('edgecolor' not in kwargs.keys()) | ('ec' not in pos_kwargs.keys()) | \
       ('markeredgecolor' not in kwargs.keys()) | ('mec' not in pos_kwargs.keys()):
        pos_kwargs['markeredgecolor'] = 'k'
    if ('linestyle' not in pos_kwargs.keys()) | ('ls' not in pos_kwargs.keys()):
        pos_kwargs['linestyle'] = 'None'
    if ('size' not in kwargs.keys()) | ('markersize' not in pos_kwargs.keys()):
        pos_kwargs['markersize'] = 3

    if orientation == 'horizontal':
        ax.fill(xvals, violin, **kwargs)

        mv = np.max(violin)
        #Draw the lines
        if pos_marker is not None:
            if (pos_marker == 'line') | (pos_marker == 'lines'):
                pos_kwargs.pop('marker')
                ax.plot(x, - 0.5 * pos_width * mv * np.ones(len(x)), marker='|', **pos_kwargs)
            else:
                ax.plot(x, np.random.uniform(low=-pos_width * mv, high=0., size=len(x)), **pos_kwargs)

            ax.set_ylim(-pos_width * mv, ax.get_ylim()[1])
            plt.draw_if_interactive()

    elif orientation == 'vertical':
        ax.fill_betweenx(xvals, 0, violin, **kwargs)

        #Draw the lines
        if pos_marker is not None:
            if (pos_marker == 'line') | (pos_marker == 'lines'):
                pos_kwargs.pop('marker')
                ax.plot(-0.5 * pos_width * mv * np.ones(len(x)), x, marker='_', **pos_kwargs)
            else:
                ax.plot(np.random.uniform(low=-pos_width * mv, high=0., size=len(x)), x, **pos_kwargs)

            ax.set_xlim(-pos_width * mv, ax.get_xlim()[1])
    plt.draw_if_interactive()

    return xvals, violin


def symmatplot(mat, p_mat=None, names=None, cmap="Greys", cmap_range=None,
               cbar=True, annot=True, diag_names=True, ax=None, **kwargs):
    """Plot a symmetric matrix with colormap and statistic values.

    Parameters
    ----------

    mat : square array
         square array of statistics.

    p_mat: nobs x nvars array
        Rectangular matrix of statistic uncertainties

    names : sequence of strings
        Names to associate with variables if `data` is not a named array.

    cmap : colormap
        Colormap name as string or colormap object.

    cmap_range : None, "full", (low, high)
        Either truncate colormap at (-max(abs(r)), max(abs(r))), use the
        full range (-1, 1), or specify (min, max) values for the colormap.

    cbar : bool
        If true, plot the colorbar legend.

    annot : bool
        Whether to annotate the upper triangle with correlation coefficients.

    diag_names: bool
        if set, display variable names in the diagonal cells

    ax : matplotlib axis
        Axis to draw plot in.

    kwargs : other keyword arguments
        Passed to ax.matshow()

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot.

    """
    if ax is None:
        ax = plt.gca()

    nvars = len(mat)
    plotmat = mat.copy()
    plotmat[np.triu_indices(nvars)] = np.nan

    if cmap_range is None:
        vmax = np.nanmax(plotmat) * 1.15
        vmin = np.nanmin(plotmat) * 1.15
    elif len(cmap_range) == 2:
        vmin, vmax = cmap_range
    else:
        raise ValueError("cmap_range argument not understood")

    mat_img = ax.matshow(plotmat, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if cbar:
        plt.colorbar(mat_img, pad=0.01, shrink=0.90)

    if annot:
        for i, j in zip(*np.triu_indices(nvars, 1)):
            val = mat[i, j]
            ax.text(j, i, "%.2g" % (val), fontdict=dict(ha="center", va="center", size='small'))
            if p_mat is not None:
                ax.text(j, i + 0.2, "+/-%.2g" % (p_mat[i, j]), fontdict=dict(ha="center", va="center", size='xx-small'))
    else:
        fill = np.ones_like(plotmat)
        fill[np.tril_indices_from(fill, -1)] = np.nan
        ax.matshow(fill, cmap="Greys", vmin=0, vmax=0, zorder=2)

    if names is None:
        names = ["x %d" % i for i in range(nvars)]

    if diag_names:
        for i, name in enumerate(names):
            ax.text(i, i, name, fontdict=dict(ha="center", va="center", weight="bold", rotation=45))
        ax.set_xticklabels(())
        ax.set_yticklabels(())
    else:
        ax.xaxis.set_ticks_position("bottom")
        xnames = names if annot else names[:-1]
        ax.set_xticklabels(xnames, rotation=90)
        ynames = names if annot else names[1:]
        ax.set_yticklabels(ynames)

    minor_ticks = np.linspace(-.5, nvars - 1.5, nvars)
    ax.set_xticks(minor_ticks, True)
    ax.set_yticks(minor_ticks, True)
    major_ticks = np.linspace(0, nvars - 1, nvars)
    xticks = major_ticks if annot else major_ticks[:-1]
    ax.set_xticks(xticks)
    yticks = major_ticks if annot else major_ticks[1:]
    ax.set_yticks(yticks)
    ax.grid(False, which="major")
    ax.grid(True, which="minor", linestyle="-")

    return ax


def corrplot(data, names=None, annot=True, sig_corr=True,
             cmap=None, cmap_range=None, cbar=True,
             diag_names=True, ax=None, **kwargs):
    """Plot a correlation matrix with colormap and r values.

    Parameters
    ----------
    data : nobs x nvars array
        Rectangular input data with variabes in the columns.

    names : sequence of strings
        Names to associate with variables if `data` is not a named array.

    annot : bool
        Whether to annotate the upper triangle with correlation coefficients.

    sig_corr : bool
        If True, use FWE-corrected p values for the sig stars.

    cmap : colormap
        Colormap name as string or colormap object.

    cmap_range : None, "full", (low, high)
        Either truncate colormap at (-max(abs(r)), max(abs(r))), use the
        full range (-1, 1), or specify (min, max) values for the colormap.

    cbar : bool
        If true, plot the colorbar legend.

    ax : matplotlib axis
        Axis to draw plot in.

    kwargs : other keyword arguments
        Passed to ax.matshow()

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot.
    """
    if not hasattr(data, 'keys'):
        names = names or ["x %d" % i for i in range(len(data))]
    else:
        names = names or data.keys()

    # Calculate the correlation matrix of the dataframe
    corrmat = np.corrcoef(data)

    # Get p values with a permutation test
    def _get_sig_cor(data, n=100):
        _data = np.asarray(data)
        N = len(data)
        dk = np.asarray([ np.corrcoef(_data[np.random.randint(0, N, size=N), :]) for k in range(10) ])
        #p16, p50, p84 = np.percentile(dk, [0.16, 0.5, 0.84], axis=0)
        #return p16, p50, p84
        return np.std(dk, axis=0)

    if annot and sig_corr:
        p_mat = _get_sig_cor(data)
    else:
        p_mat = None

    # Sort out the color range
    if cmap_range is None:
        triu = np.triu_indices(len(corrmat), 1)
        vmax = min(1, np.max(np.abs(corrmat[triu])) * 1.15)
        vmin = -vmax
        cmap_range = vmin, vmax
    elif cmap_range == "full":
        cmap_range = (-1, 1)

    # Find a colormapping, somewhat intelligently
    if cmap is None:
        if min(cmap_range) >= 0:
            cmap = "OrRd"
        elif max(cmap_range) <= 0:
            cmap = "PuBu_r"
        else:
            cmap = "coolwarm"

    # Plot using the more general symmatplot function
    ax = symmatplot(corrmat, p_mat, names, cmap, cmap_range, cbar, annot, diag_names, ax, **kwargs)

    return ax
