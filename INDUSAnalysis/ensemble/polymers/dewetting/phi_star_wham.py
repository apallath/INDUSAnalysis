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

import WHAM.binless
from WHAM.lib import potentials
from WHAM import statistics

from INDUSAnalysis import timeseries


def phi_star_wham(
    phivals: list,
    start_time: int,
    datformat: str,
    skip: int,
    imgfile: str,
    Nmin=0,
    Nmax=150,
    Nbins=200,
    temp=294,
    phibins=200,
    reload=False,
):
    # Binning
    x_bin = np.linspace(Nmin, Nmax, Nbins)

    # Beta
    beta = 1000 / (8.3145 * temp)

    # Use text-only Matplotlib backend
    matplotlib.use("Agg")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # WHAM variables
    x_it = []
    u_i = []

    for phi_idx, phi in enumerate(tqdm(phivals, desc="Looping over phis")):
        ts_waters = timeseries.loadTimeSeriesFromDAT(
            datformat.format(phi=phi), tcol=0, datacols=[2], labels=["c1", "c2"]
        )
        x_it.append(ts_waters[start_time:].data_array.flatten())
        u_i.append(potentials.linear(np.float(phi)))

    if not reload:
        # WHAM
        calc = WHAM.binless.Calc1D()
        bF, bF_bin_counts, _ = calc.compute_betaF_profile(x_it, x_bin, u_i, beta=beta, bin_style="center", logevery=1)
        bF = bF - np.min(bF)

        # Save
        with open("calc_Ntw.pkl", "wb") as cf:
            pickle.dump(calc, cf)
    else:
        with open("calc_Ntw.pkl", "rb") as cf:
            calc = pickle.load(cf)
        bF, bF_bin_counts = calc.bin_betaF_profile(x_bin)
        bF = bF - np.min(bF)

    phivals = np.array([float(phi) for phi in phivals])
    phivalbins = np.linspace(phivals.min(), phivals.max(), phibins)

    N_avg, N_var = statistics.binless_reweight_phi_ensemble(calc, phivalbins, beta)
    max_idx = np.argmax(N_var)
    print(phivalbins[max_idx])

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8), dpi=600)

    ax[0].plot(phivalbins, N_avg)
    ax[0].plot(phivalbins[max_idx], N_avg[max_idx], "x")
    ax[0].set_ylabel(r"$\langle \tilde{N} \rangle$")

    ax[1].plot(phivalbins, N_var)
    ax[1].plot(phivalbins[max_idx], N_var[max_idx], "x")
    ax[1].set_ylabel(r"$\delta\tilde{N}^2$")

    ax[1].set_xlabel(r"$\phi$")

    minor_locator = AutoMinorLocator(2)
    ax[0].xaxis.set_minor_locator(minor_locator)
    ax[0].grid(which="major", linestyle="-")
    ax[0].grid(which="minor", linestyle=":")

    ax[1].xaxis.set_minor_locator(minor_locator)
    ax[1].grid(which="major", linestyle="-")
    ax[1].grid(which="minor", linestyle=":")

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
        default="phi_star_wham.png",
        help="filename of overlap output phi_star image (default=phi_star.png)",
    )
    parser.add_argument("-Nmin", type=int, default=0, help="min value of N~ for WHAM (default=0)")
    parser.add_argument("-Nmax", type=int, default=150, help="max value of N~ for WHAM (default=150)")
    parser.add_argument("-Nbins", type=int, default=200, help="number of N~ bins for WHAM (default=200)")
    parser.add_argument("-temp", type=float, default=294, help="temperature of simulation in K for WHAM (default=294)")
    parser.add_argument(
        "-phibins", type=int, default=200, help="number of phi bins for phi-ensemble plot (default=200)"
    )
    parser.add_argument("--reload", action="store_true", default=False)

    a = parser.parse_args()

    phi_star_wham(
        phivals=a.phi,
        start_time=a.start,
        datformat=a.datformat,
        skip=a.skip,
        imgfile=a.imgfile,
        Nmin=a.Nmin,
        Nmax=a.Nmax,
        Nbins=a.Nbins,
        temp=a.temp,
        phibins=a.phibins,
        reload=a.reload,
    )
