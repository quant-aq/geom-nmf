import numpy as np


def plot_components(model, feature_names=None, ax=None):
    """Plot the learned NMF components (basis vectors).

    Parameters
    ----------
    model : GeomNMF
        A fitted GeomNMF instance.
    feature_names : list of str, optional
        Names for each feature/column.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if not provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError


def plot_reconstruction_error(model, ax=None):
    """Plot reconstruction error over iterations.

    Parameters
    ----------
    model : GeomNMF
        A fitted GeomNMF instance.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if not provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError


def plot_spatial_map(model, coords, component=0, ax=None):
    """Plot the spatial distribution of a single NMF component.

    Parameters
    ----------
    model : GeomNMF
        A fitted GeomNMF instance.
    coords : array-like of shape (n_samples, 2)
        Longitude/latitude coordinates for each sample.
    component : int, default=0
        Index of the component to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if not provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError


def plot_predicted_vs_actual(y_true, y_pred, ax=None):
    """Scatter plot of predicted vs. actual target values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if not provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    raise NotImplementedError
