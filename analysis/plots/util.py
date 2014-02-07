from ConfigParser import SafeConfigParser
import matplotlib.pyplot as plt
import os
import sys
from path import path

from mental_rotation.analysis import load_human, load_all, zscore


def load_config(pth):
    config = SafeConfigParser()
    config.read(pth)
    return config


def clear_right(ax=None):
    """Remove the right edge of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')


def clear_top(ax=None):
    """Remove the top edge of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')


def clear_top_bottom(ax=None):
    """Remove the top and bottom edges of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks([])


def outward_ticks(ax=None, axis='both'):
    """Make axis ticks face outwards rather than inwards (which is the
    default).

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    axis : string (default='both')
        The axis (either 'x', 'y', or 'both') for which to set the tick
        direction.

    """

    if ax is None:
        ax = plt.gca()
    if axis == 'both':
        ax.tick_params(direction='out')
    else:
        ax.tick_params(axis=axis, direction='out')


def save(path, fignum=None, close=True, width=None, height=None,
         ext=None, verbose=False):
    """Save a figure from pyplot.

    Parameters:

    path [string] : The path (and filename, without the extension) to
    save the figure to.

    fignum [integer] : The id of the figure to save. If None, saves the
    current figure.

    close [boolean] : (default=True) Whether to close the figure after
    saving.  If you want to save the figure multiple times (e.g., to
    multiple formats), you should NOT close it in between saves or you
    will have to re-plot it.

    width [number] : The width that the figure should be saved with. If
    None, the current width is used.

    height [number] : The height that the figure should be saved with. If
    None, the current height is used.

    ext [string or list of strings] : (default='png') The file
    extension. This must be supported by the active matplotlib backend
    (see matplotlib.backends module).  Most backends support 'png',
    'pdf', 'ps', 'eps', and 'svg'.

    verbose [boolean] : (default=True) Whether to print information
    about when and where the image has been saved.

    """

    # get the figure
    if fignum is not None:
        fig = plt.figure(fignum)
    else:
        fig = plt.gcf()

    # set its dimenions
    if width:
        fig.set_figwidth(width)
    if height:
        fig.set_figheight(height)

    # make sure we have a list of extensions
    if ext is not None and not hasattr(ext, '__iter__'):
        ext = [ext]

    # Extract the directory and filename from the given path
    directory, basename = os.path.split(path)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # infer the extension if ext is None
    if ext is None:
        basename, ex = os.path.splitext(basename)
        ext = [ex[1:]]

    for ex in ext:
        # The final path to save to
        filename = "%s.%s" % (basename, ex)
        savepath = os.path.join(directory, filename)

        if verbose:
            sys.stdout.write("Saving figure to '%s'..." % savepath)

        # Actually save the figure
        plt.savefig(savepath)

    # Close it
    if close:
        plt.close()

    if verbose:
        sys.stdout.write("Done\n")


def sync_ylims(*axes):
    """Synchronize the y-axis data limits for multiple axes. Uses the maximum
    upper limit and minimum lower limit across all given axes.

    Parameters
    ----------
    *axes : axis objects
        List of matplotlib axis objects to format

    Returns
    -------
    out : ymin, ymax
        The computed bounds

    """
    ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes])
    ymin = min(ymins)
    ymax = max(ymaxs)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    return ymin, ymax


def sync_xlims(*axes):
    """Synchronize the x-axis data limits for multiple axes. Uses the maximum
    upper limit and minimum lower limit across all given axes.

    Parameters
    ----------
    *axes : axis objects
        List of matplotlib axis objects to format

    Returns
    -------
    out : xmin, xmax
        The computed bounds

    """
    xmins, xmaxs = zip(*[ax.get_xlim() for ax in axes])
    xmin = min(xmins)
    xmax = max(xmaxs)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
    return xmin, xmax


def sync_xlabel_coords(axes, y, x=0.5):
    """Set the y-coordinate (and optionally the x-coordinate) of the x-axis
    labels.

    Parameters
    ----------
    axes : list
        list of axis objects
    y : float
        y-coordinate for the label
    x : float (default=0.5)
        x-coordinate for the label
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots

    """
    for ax in axes:
        ax.xaxis.set_label_coords(x, y)


def sync_ylabel_coords(axes, x, y=0.5):
    """Set the x-coordinate (and optionally the y-coordinate) of the y-axis
    labels.

    Parameters
    ----------
    axes : list
        list of axis objects
    x : float
        x-coordinate for the label
    y : float (default=0.5)
        y-coordinate for the label
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots

    """
    for ax in axes:
        ax.yaxis.set_label_coords(x, y)


report_pearson = r"$r$ = {median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"


def make_plot(func):
    config = load_config("config.ini")
    version = config.get("global", "version")
    results_path = path(config.get("paths", "results")).joinpath(version)
    fig_path = path(config.get("paths", "figures")).joinpath(version)
    print func(results_path, fig_path)
