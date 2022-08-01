"""
Functions to compute and plot path collective variables
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


def path_s(x, y, x_i, y_i, lam):
    r"""
    Computes progress (tangential) path collective variable.

    $$s = \frac{1}{N} \frac{\sum_{i=0}^{N-1} (i + 1)\ e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}{\sum_{i=0}^{N-1} e^{-\lambda [(x - x_i) ^ 2 + (y - y_i) ^ 2]}}$$
    
    Args:
        x: x-values to compute path CV at.
        y: y-values to compute path CV at.
        x_i: x-coordinates of images defining path.
        y_i: y-coordinates of images defining path.
        lam: Value of $\lambda$ for constructing path CV.
    """
    assert(len(x_i) == len(y_i))
    ivals = np.arange(len(x_i))
    npath = len(ivals)

    s = 1 / (npath - 1) * np.exp(logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2) + np.log(ivals)[:, np.newaxis], axis=0) 
                               - logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2), axis=0))

    return s


def path_z(x, y, x_i, y_i, lam):
    r"""
    Computes distance (parallel) path collective variable.

    Args:
        x: x-values to compute path CV at.
        y: y-values to compute path CV at.
        x_i: x-coordinates of images defining path.
        y_i: y-coordinates of images defining path.
        lam: Value of $\lambda$ for constructing path CV.
    """
    assert(len(x_i) == len(y_i))
    
    z = -1 / lam * logsumexp(-lam * ((x[np.newaxis, :] - x_i[:, np.newaxis]) ** 2 + (y[np.newaxis, :] - y_i[:, np.newaxis]) ** 2), axis=0)

    return z


def plot_path_s(xcoord, ycoord, x_i, y_i, lam, contourvals, cmap='jet', dpi=150):
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
        cmap: Matplotlib colormap (default=jet).
        dpi: Output DPI (default=150).

    Returns:
        tuple(fig, ax, cbar): Matplotlib figure, axis, and colorbar
    """
    xx, yy = np.meshgrid(xcoord, ycoord)
    x = xx.ravel()
    y = yy.ravel()

    s = path_s(x, y, x_i, y_i, lam)

    # Plot s
    fig, ax = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs = ax.contourf(xx, yy, s.reshape(len(xcoord), len(ycoord)).clip(min=0, max=1), contourvals, cmap=cmap)
    else:
        cs = ax.contourf(xx, yy, s.reshape(len(xcoord), len(ycoord)).clip(min=0, max=1), cmap=cmap)
    cbar = fig.colorbar(cs)

    return fig, ax, cbar


def plot_path_z(xcoord, ycoord, x_i, y_i, lam, contourvals, cmap, dpi):
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
        cmap: Matplotlib colormap (default=jet).
        dpi: Output DPI (default=150).
    """
    xx, yy = np.meshgrid(xcoord, ycoord)
    x = xx.ravel()
    y = yy.ravel()

    z = path_z(x, y, x_i, y_i, lam)

    # Plot s
    fig, ax = plt.subplots(dpi=dpi)
    if contourvals is not None:
        cs = ax.contourf(xx, yy, z.reshape(len(xcoord), len(ycoord)).clip(min=0, max=1), contourvals, cmap=cmap)
    else:
        cs = ax.contourf(xx, yy, z.reshape(len(xcoord), len(ycoord)).clip(min=0, max=1), cmap=cmap)
    cbar = fig.colorbar(cs)

    return fig, ax, cbar




