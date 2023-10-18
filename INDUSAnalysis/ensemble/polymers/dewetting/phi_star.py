"""
Plots phi-ensemble and calculates phi*.
"""

import argparse
import logging
import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import MDAnalysis as mda
import numpy as np
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

from INDUSAnalysis import timeseries


def phi_star(
    phivals: list,
    start_time: int,
    datformat: str,
    skip: int,
    imgfile: str,
):
    ############################################################################
    # Load data
    ############################################################################

    meanwaters = np.zeros(len(phivals))

    for phiidx, phi in enumerate(tqdm(phivals, desc="Looping over phis")):
        ts_waters = timeseries.loadTimeSeriesFromDAT(
            datformat.format(phi=phi), tcol=0, datacols=[2], labels=["c1", "c2"]
        )

        # Calculate mean
        meanwaters[phiidx] = ts_waters[start_time:].data_array.mean()

    print(meanwaters)

    phivals = np.array([float(phi) for phi in phivals])
    order = np.argsort(phivals)

    ############################################################################
    # Actual fits
    ############################################################################

    # Use text-only Matplotlib backend
    matplotlib.use("Agg")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    order = np.argsort(phivals)
    xdata = np.array(phivals)[order]
    ydata = meanwaters[order]

    # Plot original data
    ax.plot(
        xdata,
        ydata,
        linestyle=":",
        marker="s",
        markersize=3,
        fillstyle="none",
    )

    # Fit univariate spline
    spline = UnivariateSpline(xdata, ydata, k=3, s=0.5 * len(ydata))  # Cubic spline

    x_spline_data = np.linspace(min(phivals), max(phivals), 100)
    y_spline_data = spline(x_spline_data)

    # Find phi_i_star
    phi_i_star_yval = ydata[0] / 2

    phi_i_star_spline_idx = np.abs(y_spline_data - phi_i_star_yval).argmin()
    phi_i_star = x_spline_data[phi_i_star_spline_idx]
    print(phi_i_star)

    # Plot spline and phi_i_star
    ax.plot(x_spline_data, y_spline_data)
    ax.plot(x_spline_data[phi_i_star_spline_idx], y_spline_data[phi_i_star_spline_idx], "x")

    ax.grid()

    ax.set_xlabel(r"$\phi$ (kJ/mol)")
    ax.set_ylabel(r"$\langle \tilde{N} \rangle$")

    plt.savefig(imgfile)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs="+", help="phi values to read (phi=0 must be first)")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument(
        "-datformat",
        help="format of INDUS waters (.dat) file, with {phi} placeholders for phi value. Missing placeholders are ignored.",
    )
    parser.add_argument(
        "-skip", type=int, default=1, help="Interval between frames when reading trajectory (default=1)"
    )
    parser.add_argument(
        "-imgfile",
        default="phi_star.png",
        help="filename of phi* output image (default=phi_star.png)",
    )

    a = parser.parse_args()

    phi_star(
        phivals=a.phi,
        start_time=a.start,
        datformat=a.datformat,
        skip=a.skip,
        imgfile=a.imgfile,
    )
