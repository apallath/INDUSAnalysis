"""
Plots ni v/s phi and phi_i* for a set of representative atoms, and also for each atom i.

Stores calculated phi_i* values.
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

from INDUSAnalysis.timeseries import create1DTimeSeries, TimeSeriesAnalysis


def phi_i_star(phivals: list,
               runs: list,
               start_time: int,
               structfile: str,
               calc_dir: str,
               ni_format: str,
               sample_imgfile: str,
               all_imgformat: str,
               pklfile: str,
               plot_probe_indices: list):

    nruns = len(runs)

    tsa = TimeSeriesAnalysis()

    ############################################################################
    # Load data
    ############################################################################

    u = mda.Universe(structfile)
    protein_heavy = u.select_atoms("protein and not name H*")
    protein_heavy_indices = protein_heavy.atoms.indices

    if nruns > 1:
        meanwaters = np.zeros((len(phivals), nruns, len(protein_heavy_indices)))

        for idx, phi in enumerate(phivals):
            for runidx, run in enumerate(runs):
                ts = tsa.load_TimeSeries(calc_dir + ni_format.format(phi=phi, run=run))
                ts = ts[start_time:]
                run_waters = ts.data_array[:, protein_heavy_indices]

                # Calculate per-atom mean waters for each run
                mean_run_waters = np.mean(run_waters, axis=0)

                # Append per-atom mean waters for each run
                meanwaters[idx, runidx, :] = mean_run_waters

        mean_meanwaters = np.mean(meanwaters, axis=1)
        std_meanwaters = np.std(meanwaters, axis=1)

    elif nruns == 1:
        mean_meanwaters = np.zeros((len(phivals), len(protein_heavy_indices)))
        std_meanwaters = np.zeros((len(phivals), len(protein_heavy_indices)))

        for idx, phi in enumerate(tqdm(phivals, desc="Computing standard errors across dataset")):
            for hidx, h in enumerate(protein_heavy_indices):
                ts = tsa.load_TimeSeries(calc_dir + ni_format.format(phi=phi))
                ts = ts[start_time:]
                run_waters = ts.data_array[:, h]

                # Calculate mean
                mean_meanwaters[idx, hidx] = run_waters.mean()

                # Calculate sem with bootstrapping
                std_meanwaters[idx, hidx] = create1DTimeSeries(run_waters).standard_error(nboot=25, use_pymbar=False)

    phivals = np.array([float(phi) for phi in phivals])
    order = np.argsort(phivals)

    ############################################################################
    # Sample fits
    ############################################################################
    plot_probe_indices = [int(x) for x in plot_probe_indices]

    fig, ax = plt.subplots(2, 1, figsize=(12, 16), dpi=300)

    for ixx, probe in enumerate(plot_probe_indices):
        order = np.argsort(phivals)
        xdata = np.array(phivals)[order]
        ydata = mean_meanwaters[:, probe][order]
        yerr = std_meanwaters[:, probe][order]

        # Disallow zero errors
        for idx, val in enumerate(yerr):
            if val < 1e-3:
                yerr[idx] = 1e-3

        # Plot original data
        ax[0].errorbar(xdata, ydata, yerr=yerr,
                       barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none",
                       label="Probe on h. atom " + str(probe), color="C{}".format(ixx))
        ax[0].fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.2)

        # Plot phi_i* y value
        phi_i_star_yval = ydata[0] / 2

        yweights = 1 / yerr
        spline = UnivariateSpline(xdata, ydata, w=yweights, k=3, s=0.5 * len(ydata))  # Cubic spline

        x_spline_data = np.linspace(min(phivals), max(phivals), 100)
        y_spline_data = spline(x_spline_data)
        dydx = spline.derivative()(x_spline_data)

        # Find phi_i_star
        phi_i_star_spline_idx = np.abs(y_spline_data - phi_i_star_yval).argmin()
        phi_i_star = x_spline_data[phi_i_star_spline_idx]

        ax[0].plot(x_spline_data, y_spline_data, label="Probe on h. atom " + str(probe), color="C{}".format(ixx))
        ax[0].plot(phi_i_star, y_spline_data[phi_i_star_spline_idx], 'x')

        ax[1].plot(x_spline_data, -dydx, label="Probe on h. atom " + str(probe), color="C{}".format(ixx))

    for i in range(2):
        x_minor_locator = AutoMinorLocator(10)
        y_minor_locator = AutoMinorLocator(10)
        ax[i].xaxis.set_minor_locator(x_minor_locator)
        ax[i].yaxis.set_minor_locator(y_minor_locator)
        ax[i].grid(which='major', linestyle='-')
        ax[i].grid(which='minor', linestyle=':')

        ax[i].legend()

    ax[0].set_xlabel(r"$\phi$")
    ax[0].set_ylabel(r"Mean probe waters $\langle n_i \rangle$")

    ax[1].set_xlabel(r"$\phi$")
    ax[1].set_ylabel(r"Susceptibility $-d\langle n_i \rangle / d\phi$")

    plt.savefig(sample_imgfile)

    ############################################################################
    # Actual fits
    ############################################################################

    # Use text-only Matplotlib backend
    matplotlib.use('Agg')

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Debug
    probes = range(len(protein_heavy_indices))

    phi_i_stars = np.infty * np.ones(len(protein_heavy_indices))

    for probe in tqdm(probes):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        order = np.argsort(phivals)
        xdata = np.array(phivals)[order]
        ydata = mean_meanwaters[:, probe][order]
        yerr = std_meanwaters[:, probe][order]

        # Disallow zero errors
        for idx, val in enumerate(yerr):
            if val < 1e-3:
                yerr[idx] = 1e-3

        # Plot original data
        ax.errorbar(xdata, ydata, yerr=yerr,
                    barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none")
        ax.fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.3)

        # Fit univariate spline
        yweights = 1 / yerr
        spline = UnivariateSpline(xdata, ydata, w=yweights, k=3, s=0.5 * len(ydata))  # Cubic spline

        x_spline_data = np.linspace(min(phivals), max(phivals), 100)
        y_spline_data = spline(x_spline_data)

        # Find phi_i_star
        phi_i_star_yval = ydata[0] / 2

        phi_i_star_spline_idx = np.abs(y_spline_data - phi_i_star_yval).argmin()
        phi_i_star = x_spline_data[phi_i_star_spline_idx]

        # Plot spline and phi_i_star
        ax.plot(x_spline_data, y_spline_data)
        ax.plot(x_spline_data[phi_i_star_spline_idx], y_spline_data[phi_i_star_spline_idx], 'x')

        ax.grid()

        ax.set_xlabel(r"$\phi$ (kJ/mol)")
        ax.set_ylabel(r"$\langle n_i \rangle$")

        atom = protein_heavy.atoms[probe]
        atom_name = "{}{}:{}".format(atom.resname, atom.resid, atom.name)
        title = r"H. atom {} ({})".format(probe, atom_name) + r", $\phi_i^*$ = {:.2f}".format(phi_i_star)

        ax.set_title(title)

        plt.savefig(all_imgformat.format(probe))
        plt.close()

        # Store data
        phi_i_stars[probe] = phi_i_star

    # Save phi_i_star data and errors to file
    phi_i_star_data = dict()
    phi_i_star_data['phi_i_stars'] = phi_i_stars

    with open(pklfile, "wb") as outfile:
        pickle.dump(phi_i_star_data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read (enter 0 in case of no runs)")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-structfile", help="path to structure file (.pdb, .gro, .tpr)")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("-ni_format", help="format of .pkl file containing Ntw, with {phi} placeholders for phi value and {run} placeholders for run value. Missing placeholders are ignored.")
    parser.add_argument("-sample_imgfile", help="sample phi_i* output image")
    parser.add_argument("-all_imgformat", help="format of phi_i* output images for all heavy atoms, with {} placeholder for heavy atom index")
    parser.add_argument("-pklfile", help="output file to dump phi_i* data to (.pkl)")
    parser.add_argument("-plot_probe_indices", type=int, nargs='+', help="probe indices to plot in the sample image file")

    a = parser.parse_args()

    phi_i_star(a.phi, a.runs, a.start, a.structfile, a.calc_dir, a.ni_format, a.sample_imgfile, a.all_imgformat,
               a.pklfile, a.plot_probe_indices)
