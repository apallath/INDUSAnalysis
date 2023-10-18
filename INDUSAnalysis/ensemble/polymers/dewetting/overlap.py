"""
Calculates overlap between windows
"""

import argparse
import logging
import os
import pickle
import warnings

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import MDAnalysis as mda
import numpy as np
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

from INDUSAnalysis import timeseries


def overlap(phivals: list, start_time: int, datformat: str, skip: int, imgfile: str, Nmin=0, Nmax=3000, Nbins=200):
    Ntw_range = [Nmin, Nmax]
    Ntw_bins = Nbins

    ############################################################################
    # Plot histogram
    ############################################################################

    # Use text-only Matplotlib backend
    matplotlib.use("Agg")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Set up normalization and colormap
    normalize = mcolors.Normalize(vmin=min(phivals), vmax=max(phivals))
    colormap = cm.rainbow

    for phi_idx, phi in enumerate(tqdm(phivals, desc="Looping over phis")):
        ts_waters = timeseries.loadTimeSeriesFromDAT(
            datformat.format(phi=phi), tcol=0, datacols=[2], labels=["c1", "c2"]
        )
        ts_waters = ts_waters[start_time:]

        hist, edges = np.histogram(ts_waters.data_array, bins=Ntw_bins, range=Ntw_range, density=True)
        x = 0.5 * (edges[1:] + edges[:-1])
        y = hist
        ax.plot(x, y, color=colormap(normalize(float(phi))))
        ax.fill_between(x, 0, y, color=colormap(normalize(float(phi))), alpha=0.4)

    ax.set_xlabel(r"$\tilde{N}$")

    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which="major", linestyle="-")
    ax.grid(which="minor", linestyle=":")

    # Display colorbar
    phivals = np.array([float(phi) for phi in phivals])
    scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappable.set_array(np.array(phivals))
    fig.colorbar(scalarmappable, label=r"$\phi$ (kJ/mol)")

    plt.savefig(imgfile, format="png")
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
        default="overlap.png",
        help="filename of overlap output image (default=phi_star.png)",
    )
    parser.add_argument("-Nmin", type=int, default=0, help="min value of N~ (default=0)")
    parser.add_argument("-Nmax", type=int, default=150, help="max value of N~ (default=3000)")
    parser.add_argument("-Nbins", type=int, default=200, help="number of N~ bins for histogram (default=200)")

    a = parser.parse_args()

    overlap(
        phivals=a.phi,
        start_time=a.start,
        datformat=a.datformat,
        skip=a.skip,
        imgfile=a.imgfile,
        Nmin=a.Nmin,
        Nmax=a.Nmax,
        Nbins=a.Nbins,
    )
