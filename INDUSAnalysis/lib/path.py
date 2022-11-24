"""
Functions to compute and plot path collective variables
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp


def path_s(x, y, x_i, y_i, lam, sqrt=False):
    r"""
    Computes progress (tangential) path collective variable.

    $$s = \frac{1}{N} \frac{\sum_{i=0}^{N-1} (i + 1)\ e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}{\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}$$
    
    Args:
        x: x-values to compute path CV at.
        y: y-values to compute path CV at.
        x_i: x-coordinates of images defining path.
        y_i: y-coordinates of images defining path.
        lam: Value of $\lambda$ for constructing path CV.
        sqrt: If True, computes distance instead of squared distance (default=False).
    """
    assert(len(x_i) == len(y_i))
    ivals = np.arange(len(x_i))
    npath = len(ivals)

    if sqrt:
        s = 1 / (npath - 1) * np.exp(logsumexp(-lam * (((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2) ** 0.5) + np.log(ivals)[:, np.newaxis], axis=0) 
                                - logsumexp(-lam * (((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2) ** 0.5), axis=0))
    else:
        s = 1 / (npath - 1) * np.exp(logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2) + np.log(ivals)[:, np.newaxis], axis=0) 
                                - logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2), axis=0))

    return s


def path_s_scaled(x, y, x_i, y_i, lam, sqrt=False, x_min=None, x_max=None, y_min=None, y_max=None):
    r"""
    Computes progress (tangential) path collective variable using a scaled distance function.

    $$s = \frac{1}{N} \frac{\sum_{i=0}^{N-1} (i + 1)\ e^{-\lambda [(x' - x_i') ^ 2 + (y' - y_i') ^ 2]}}{\sum_{i=0}^{N-1} e^{-\lambda [(x' - x_i') ^ 2 + (y' - y_i') ^ 2]}}$$

    where

    $$x' = (x - x_{min}) / (x_{max} - x_{min})$$
    $$x_i' = (x_i - x_{min}) / (x_{max} - x_{min})$$
    $$y' = (y - y_{min}) / (y_{max} - y_{min})$$
    $$y_i' = (y - y_{min}) / (y_{max} - y_{min})$$
    
    Args:
        x: x-values to compute path CV at.
        y: y-values to compute path CV at.
        x_i: x-coordinates of images defining path.
        y_i: y-coordinates of images defining path.
        lam: Value of $\lambda$ for constructing path CV.
        sqrt: If True, computes distance instead of squared distance (default=False).
        x_min, x_max, y_min, y_max: Scaling parameters (optional. 
            If None, the values are calculated from the max and min values of the path points.)
    """
    assert(len(x_i) == len(y_i))
    ivals = np.arange(len(x_i))
    npath = len(ivals)

    if x_min is None:
        x_min = x_i.min()
    if x_max is None:
        x_max = x_i.max()
    if y_min is None:
        y_min = y_i.min()
    if y_max is None:
        y_max = y_i.max()

    x_s = (x - x_min) / (x_max - x_min)
    x_i_s = (x_i - x_min) / (x_max - x_min)

    y_s = (y - y_min) / (y_max - y_min)
    y_i_s = (y_i - y_min) / (y_max - y_min)

    if sqrt:
        s = 1 / (npath - 1) * np.exp(logsumexp(-lam * (((x_s[np.newaxis, :] - x_i_s[:, np.newaxis]) ** 2 + (y_s[np.newaxis, :] - y_i_s[:, np.newaxis]) ** 2) ** 0.5) + np.log(ivals)[:, np.newaxis], axis=0) 
                                - logsumexp(-lam * (((x_s[np.newaxis, :] - x_i_s[:, np.newaxis]) ** 2 + (y_s[np.newaxis, :] - y_i_s[:, np.newaxis]) ** 2) ** 0.5), axis=0))
    else:
        s = 1 / (npath - 1) * np.exp(logsumexp(-lam * ((x_s[np.newaxis, :] - x_i_s[:, np.newaxis]) ** 2 + (y_s[np.newaxis, :] - y_i_s[:, np.newaxis]) ** 2) + np.log(ivals)[:, np.newaxis], axis=0) 
                                - logsumexp(-lam * ((x_s[np.newaxis, :] - x_i_s[:, np.newaxis]) ** 2 + (y_s[np.newaxis, :] - y_i_s[:, np.newaxis]) ** 2), axis=0))

    return s


def path_z(x, y, x_i, y_i, lam, sqrt=False):
    r"""
    Computes distance (parallel) path collective variable.

    Args:
        x: x-values to compute path CV at.
        y: y-values to compute path CV at.
        x_i: x-coordinates of images defining path.
        y_i: y-coordinates of images defining path.
        lam: Value of $\lambda$ for constructing path CV.
        sqrt: If True, computes distance instead of squared distance (default=False).
    """
    assert(len(x_i) == len(y_i))
    
    if sqrt:
        z = -1 / lam * logsumexp(-lam * (((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2) ** 0.5), axis=0)
    else:
        z = -1 / lam * logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2), axis=0)

    return z


def path_z_scaled(x, y, x_i, y_i, lam, sqrt=False, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Computes distance (parallel) path collective variable using a scaled distance function.

    $$z = -\frac{1}{\lambda} \ln (\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]})$$

    where

    $$x' = (x - x_{min}) / (x_{max} - x_{min})$$
    $$x_i' = (x_i - x_{min}) / (x_{max} - x_{min})$$
    $$y' = (y - y_{min}) / (y_{max} - y_{min})$$
    $$y_i' = (y - y_{min}) / (y_{max} - y_{min})$$

    Args:
        x: x-values to compute path CV at.
        y: y-values to compute path CV at.
        x_i: x-coordinates of images defining path.
        y_i: y-coordinates of images defining path.
        lam: Value of $\lambda$ for constructing path CV.
        sqrt: If True, computes distance instead of squared distance (default=False).
        x_min, x_max, y_min, y_max: Scaling parameters (optional. 
            If None, the values are calculated from the max and min values of the path points.)
    """
    assert(len(x_i) == len(y_i))

    if x_min is None:
        x_min = x_i.min()
    if x_max is None:
        x_max = x_i.max()
    if y_min is None:
        y_min = y_i.min()
    if y_max is None:
        y_max = y_i.max()

    x_s = (x - x_min) / (x_max - x_min)
    x_i_s = (x_i - x_min) / (x_max - x_min)

    y_s = (y - y_min) / (y_max - y_min)
    y_i_s = (y_i - y_min) / (y_max - y_min)
    
    z = -1 / lam * logsumexp(-lam * (((x_s[np.newaxis, :] - x_i_s[:, np.newaxis]) ** 2 + (y_s[np.newaxis, :] - y_i_s[:, np.newaxis]) ** 2) ** 0.5), axis=0)

    return z


def plot_path_s(xcoord, ycoord, x_i, y_i, lam, contourvals, sqrt=False, scaled=False, x_min=None, x_max=None, y_min=None, y_max=None, 
    cmap='jet', dpi=150):
    r"""
    Plots progress (tangential) path collective variable on a 2-dimensional grid.

    $$s = \frac{1}{N} \frac{\sum_{i=0}^{N-1} (i + 1)\ e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}{\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}$$

    Args:
        xcoord: Array specifying x-axis coordinates of grid.
        ycoord: Array specifying y-axis coordinates of grid.
        x_i: x-coordinates of images defining a path.
        y_i: y-coordinates of images defining a path.
        lam: Value of $\lambda$ for constructing path CV.
        contourvals (int or array-like): Determines the number and positions of the contour lines / regions. Refer to the `matplotlib documentation`_ for details.
        scaled: If true, uses scaled distance functions for computing s.
        x_min, x_max, y_min, y_max: Scaling parameters (optional. 
            If None, the values are calculated from the max and min values of the path points.)
        cmap: Matplotlib colormap (default=jet).
        dpi: Output DPI (default=150).

    Returns:
        tuple(fig, ax, cbar): Matplotlib figure, axis, and colorbar
    """
    xx, yy = np.meshgrid(xcoord, ycoord)
    x = xx.ravel()
    y = yy.ravel()

    if scaled:
        s = path_s_scaled(x, y, x_i, y_i, lam, sqrt, x_min, x_max, y_min, y_max)
    else:
        s = path_s(x, y, x_i, y_i, lam, sqrt)

    # Plot s
    fig, ax = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs = ax.contourf(xx, yy, s.reshape(len(ycoord), len(xcoord)).clip(min=0, max=1), contourvals, cmap=cmap)
    else:
        cs = ax.contourf(xx, yy, s.reshape(len(ycoord), len(xcoord)).clip(min=0, max=1), cmap=cmap)
    cbar = fig.colorbar(cs)

    return fig, ax, cbar


def plot_path_z(xcoord, ycoord, x_i, y_i, lam, contourvals, sqrt=False, scaled=False, x_min=None, x_max=None, y_min=None, y_max=None,
    cmap='jet', dpi=150):
    r"""
    Plots distance (parallel) path collective variable on a 2-dimensional grid.

    $$z = -\frac{1}{\lambda} \ln (\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]})$$

    Args:
        xcoord: Array specifying x-axis coordinates of grid.
        ycoord: Array specifying y-axis coordinates of grid.
        x_i: x-coordinates of images defining a path.
        y_i: y-coordinates of images defining a path.
        lam: Value of $\lambda$ for constructing path CVs.
        contourvals (int or array-like): Determines the number and positions of the contour lines / regions. Refer to the `matplotlib documentation`_ for details.
        scaled: If true, uses scaled distance functions for computing z.
        x_min, x_max, y_min, y_max: Scaling parameters (optional. 
            If None, the values are calculated from the max and min values of the path points.)
        cmap: Matplotlib colormap (default=jet).
        dpi: Output DPI (default=150).
    """
    xx, yy = np.meshgrid(xcoord, ycoord)
    x = xx.ravel()
    y = yy.ravel()

    if scaled:
        z = path_z_scaled(x, y, x_i, y_i, lam, sqrt, x_min, x_max, y_min, y_max)
    else:
        z = path_z(x, y, x_i, y_i, lam, sqrt)

    # Plot z
    fig, ax = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs = ax.contourf(xx, yy, z.reshape(len(ycoord), len(xcoord)), contourvals, cmap=cmap)
    else:
        cs = ax.contourf(xx, yy, z.reshape(len(ycoord), len(xcoord)), cmap=cmap)
    cbar = fig.colorbar(cs)

    return fig, ax, cbar




