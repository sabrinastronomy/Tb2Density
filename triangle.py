"""

triangle.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm

try:
    from scipy.ndimage.filters import gaussian_filter
except ImportError:
    pass

def bin_e2c(bins):
    """
    Convert bin edges to bin centers.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return 0.5 * (bins[1:] + bins[:-1])

def bin_c2e(bins):
    """
    Convert bin centers to bin edges.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return np.concatenate(([bins[0] - 0.5 * dx], bins + 0.5 * dx))

def get_error_2d(L, nu=[0.95, 0.68]):
    """
    Integrate outward at "constant water level" to determine proper
    2-D marginalized confidence regions.

    ..note:: This is fairly crude -- the "coarse-ness" of the resulting
        PDFs will depend a lot on the binning, hence the option to smooth
        the 2-D posteriors in `triangle` method below.

    Parameters
    ----------
    L : np.ndarray
        2-D histogram of posterior samples
    nu : float, list
        Confidence intervals of interest.

    Returns
    -------
    List of contour values (relative to maximum likelihood) corresponding
    to the confidence region bounds specified in the "nu" parameter,
    in order of decreasing nu.
    """

    if type(nu) in [int, float]:
        nu = np.array([nu])

    # Put nu-values in ascending order
    if not np.all(np.diff(nu) > 0):
        nu = nu[-1::-1]

    peak = float(L.max())
    tot = float(L.sum())

    # Counts per bin in descending order
    Ldesc = np.sort(L.ravel())[-1::-1]

    Lencl_prev = 0.0

    # Will correspond to whatever contour we're on
    j = 0

    # Some preliminaries
    contours = [1.0]
    Lencl_running = []

    # Iterate from high likelihood to low
    for i in range(1, Ldesc.size):

        # How much area (fractional) is contained in bins at or above the current level?
        Lencl_now = L[L >= Ldesc[i]].sum() / tot

        # Keep running list of enclosed (integrated) likelihoods
        Lencl_running.append(Lencl_now)

        # What contour are we on?
        Lnow = Ldesc[i]

        # Haven't hit next contour yet
        if Lencl_now < nu[j]:
            pass
        # Just passed a contour
        else:

            # Interpolate to find contour more precisely
            Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                [Ldesc[i-1], Ldesc[i]])

            # Save relative to peak
            contours.append(Linterp / peak)

            j += 1

            if j == len(nu):
                break

        Lencl_prev = Lencl_now

    # Return values that match up to inputs
    return nu[-1::-1], np.array(contours[-1::-1])

def plot_triangle(flatchain, fig=1, axes=None, elements=None,
    bins=20, burn=0, fig_kwargs={}, contours=True,
    fill=False, conflevels=[0.95, 0.68], take_log=False, is_log=False,
    labels=None, smooth=None, **kwargs):
    """
    Minimalist routine for making triangle plots.

    Parameters
    ----------
    flatchain : np.ndarray
        MCMC samples, should have shape (num samples, num parameters)
    fig : int, object
        Can provide integer figure ID number if calling this method the first
        time, or provide previously returned matplotlib.figure.Figure
        instance if over-plotting
    axes : list
        If calling this routine to over-plot more contours, this should be
        the second returned value of the previous call, i.e., a nested list
        of matplotlib.axes._subplots.AxesSubplot instances. It is sorted by
        row, such that axes[0][0] is the top-left corner, axes[0][1] is the
        top row, second column, etc., with axes[-1][-1] being the bottom right
        corner.
    elements : list of integers
        Subset of parameters to include in plot, given by their corresponding
        index in the flatchain (second axis), e.g., elements=[0,1] would include
        only the first and second parameters.
    bins : list, int
        Either number of bins to use in each dimension, or a list specifying
        the bins explicitly for each parameter.
    burn : int
        Number of samples to discard at beginning of chain.
    contours : bool
        If True, plot open contours, otherwise plot posterior with pcolormesh.
    fill : bool
        If True, plot filled contours, otherwise, use open contours.
    conflevels : list
        List of confidence levels, i.e., default of [0.95, 0.68] indicates that
        we want contours enclosing 95% and 68% of the total samples.
    take_log : bool, list
        Take log10 of samples before histogramming?
    is_log : bool, list
        If parameter is actually log10(parameter), setting this to True
        will perform a 10**parameter before histogramming.
    labels : list
        Can provide a set of custom labels if you'd like.
    smooth : int, float
        If not None, this will smooth the 2-D histogram before finding
        contour levels. This is the width of the Gaussian smoothing kernel
        in number of pixels.

    kwargs : dict, optional
        Any additional keyword arguments passed will be sent along to
        contour, contourf, or pcolormesh. For example, colors, linestyles, etc.
    fig_kwargs : dict, optional
        Additional kwargs to pass to matplotlib.pyplot.figure when initializing
        axes. For example, left, right, top, bottom.

    Returns
    -------
    A tuple containing the figure object and nested list of axes objects,
    in that order, i.e., something like

        fig, axes = triangle(flatchain)

    will yield an matplotlib.figure.Figure for `fig` and nested list of
    matplotlib.axes._subplots.AxesSubplot objects in `axes`. Can pass these
    back in to subsequent `triangle` calls to over-plot on same axes.

    """

    has_ax = axes is not None

    if not has_ax:
        fig = pl.figure(constrained_layout=True, num=fig, **fig_kwargs)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
    else:
        axes_by_row = axes

    if elements is None:
        elements = range(0, flatchain.shape[1])
    else:
        Np = len(elements)
        print(Np)

    if type(bins) not in [list, tuple, np.ndarray]:
        bins = [bins] * Np
    if type(is_log) not in [list, tuple, np.ndarray]:
        is_log = [is_log] * Np
    if type(take_log) not in [list, tuple, np.ndarray]:
        take_log = [take_log] * Np

    if labels is None:
        labels = ['par {}'.format(i) for i in range(Np)]
    else:
        assert len(labels) == Np, "Must supply label for each parameter!"

    # Remember, for gridspec, rows are numbered frop top-down.
    if not has_ax:
        gs = fig.add_gridspec(Np, Np)
        axes_by_row = [[] for i in range(Np)]

    for i, row in enumerate(range(Np)):
        for j, col in enumerate(range(Np)):
            # Skip elements in upper triangle
            if j > i:
                continue

            # Create axis
            if not has_ax:
                _ax = fig.add_subplot(gs[i,j])
                axes_by_row[i].append(_ax)
            else:
                _ax = axes_by_row[i][j]

            # Retrieve data to be used in plot
            if not is_log[i]:
                p1 = flatchain[burn:,elements[i]]
            else:
                p1 = 10**flatchain[burn:,elements[i]]

            if take_log[i]:
                p1 = np.log10(p1)

            # 2-D PDFs from here on
            if not is_log[j]:
                p2 = flatchain[burn:,elements[j]]
            else:
                p2 = 10**flatchain[burn:,elements[j]]

            if take_log[j]:
                p2 = np.log10(p2)

            # 1-D PDFs
            if i == j:
                kw = kwargs.copy()
                if 'colors' in kw:
                    del kw['colors']
                _ax.hist(p2, density=True, bins=bins[j], histtype='step', **kw)

                if j > 0:
                    _ax.set_yticklabels([])
                    if j == Np - 1:
                        _ax.set_xlabel(labels[j])
                    else:
                        _ax.set_xticklabels([])
                else:
                    _ax.set_ylabel(r'PDF')

                ok = np.isfinite(p2)
                _ax.set_xlim(p2[ok==1].min(), p2[ok==1].max())
                continue

            # Histogram samples in this 2-D plane, optionally smooth
            hist, be2, be1 = np.histogram2d(p2, p1, [bins[j], bins[i]])
            bc1 = bin_e2c(be1)
            bc2 = bin_e2c(be2)

            if smooth:
                hist = gaussian_filter(hist, smooth)

            # Draw contours or plot image
            if contours:

                # Convert desired confidence levels to iso-likelihood contours
                # that we can use to help plot.
                nu, levels = get_error_2d(hist, nu=conflevels)

                # (columns, rows, histogram)
                if fill:
                    _ax.contourf(bc2, bc1, hist.T / hist.max(),
                        levels, **kwargs)
                else:
                    _ax.contour(bc2, bc1, hist.T / hist.max(),
                        levels, **kwargs)
            else:
                _ax.pcolormesh(bc2, bc1, hist.T / hist.max(),
                    norm=LogNorm(), **kwargs)

            # Get rid of labels/ticks on interior panels.
            if i < Np - 1:
                _ax.set_xticklabels([])
            else:
                _ax.set_xlabel(labels[j])

            if j > 0:
                _ax.set_yticklabels([])
            else:
                _ax.set_ylabel(labels[i])

            # Set axis limits
            ok1 = np.isfinite(p1)
            ok2 = np.isfinite(p2)
            _ax.set_ylim(p1[ok1==1].min(), p1[ok1==1].max())
            _ax.set_xlim(p2[ok2==1].min(), p2[ok2==1].max())

    # Done
    return fig, axes_by_row


if __name__ == '__main__':
    redshifts = [10, 8, 6, 4]
    direc = "/Users/sabrinaberger/new_data"
    plot_direc = "/Users/sabrinaberger/RESULTS_PLOTS"
    for redshift in redshifts:
        corr_samples = np.load("{}/plots_num_samples_10000_eps_0_dvals_128_sigmaT_1_z_{}_CORRELATED_DENSITIES_{}_SAMPLES.npy".format(direc, redshift, redshift))
        uncorr_samples = np.load("{}/plots_num_samples_10000_eps_0_dvals_128_sigmaT_1_z_{}_UNCORR_DENSITIES_{}_SAMPLES.npy".format(direc, redshift, redshift))

        corr_samples = corr_samples[-1000:]
        uncorr_samples = uncorr_samples[-1000:]
        size = 5
        xvals = np.linspace(0, size-1, size, dtype=int)
        print(xvals)

        fig_corr, axes_corr = plot_triangle(corr_samples, contours=True, color='k', colors='k', elements=xvals, labels=xvals)
        fig_uncorr, axes_uncorr = plot_triangle(uncorr_samples, contours=True, color='b', colors='b', fig=fig_corr, axes=axes_corr, elements=xvals, labels=xvals)
        fig_corr.savefig("{}/{}_corr_vs_uncorr_corner_first5.pdf".format(plot_direc, redshift))
    # flatchain1 = np.random.normal(scale=0.5, size=40000).reshape(10000, 4)
    # flatchain2 = np.random.normal(scale=1, size=40000).reshape(10000, 4)
    #
    # fig, axes = plot_triangle(flatchain1, contours=True, color='k', colors='k',
    #     smooth=1.5)
    # fig, axes = plot_triangle(flatchain2, contours=True, color='b', colors='b',
    #     smooth=1.5, fig=fig, axes=axes)
