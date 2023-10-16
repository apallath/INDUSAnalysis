"""
Plots ni v/s phi and phi_i* for each atom i.

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

from INDUSAnalysis.timeseries import TimeSeries, TimeSeriesAnalysis


def phi_i_star(
    phivals: list,
    start_time: int,
    structfile: str,
    trajformat: str,
    skip: int,
    probe_selection: str,
    probe_radius: float,
    solvent_selection: str,
    all_imgformat: str,
    all_watersformat: str,
    pklfile: str,
    reload: bool,
):
    ############################################################################
    # Load data
    ############################################################################

    u = mda.Universe(structfile)
    probe_atoms = u.select_atoms(probe_selection)
    probe_indices = probe_atoms.atoms.indices

    meanwaters = np.zeros((len(phivals), len(probe_indices)))

    for phiidx, phi in enumerate(tqdm(phivals, desc="Looping over phis")):
        if reload:
            ts_waters = TimeSeriesAnalysis.load_TimeSeries(all_watersformat.format(phi=phi))

            print(ts_waters[start_time:].data_array)

        else:
            utraj = mda.Universe(structfile, trajformat.format(phi=phi))

            times = np.zeros(len(utraj.trajectory[::skip]))
            waters = np.zeros((len(utraj.trajectory[::skip]), len(probe_indices)))

            for tidx, ts in enumerate(tqdm(utraj.trajectory[::skip], desc="Processing trajectory")):
                times[tidx] = ts.time
                for pidx, pat in enumerate(probe_atoms):
                    wat = utraj.select_atoms(
                        "{} and (around {} (index {}))".format(solvent_selection, probe_radius, pat.index)
                    )
                    waters[tidx, pidx] = len(wat)

            ts_waters = TimeSeries(times, waters, labels=["ni", "at_idx"])

            TimeSeriesAnalysis.save_TimeSeries(ts_waters, all_watersformat.format(phi=phi))

        # Calculate mean
        meanwaters[phiidx, :] = ts_waters[start_time:].data_array.mean(axis=0)

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

    # Debug
    probes = range(len(probe_indices))

    phi_i_stars = np.infty * np.ones(len(probe_indices))

    for probe in tqdm(probes):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        order = np.argsort(phivals)
        xdata = np.array(phivals)[order]
        ydata = meanwaters[:, probe][order]

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

        # Plot spline and phi_i_star
        ax.plot(x_spline_data, y_spline_data)
        ax.plot(x_spline_data[phi_i_star_spline_idx], y_spline_data[phi_i_star_spline_idx], "x")

        ax.grid()

        ax.set_xlabel(r"$\phi$ (kJ/mol)")
        ax.set_ylabel(r"$\langle n_i \rangle$")

        atom = probe_atoms.atoms[probe]
        atom_name = "{}{}:{}".format(atom.resname, atom.resid, atom.name)
        title = r"P. atom {} ({})".format(probe, atom_name) + r", $\phi_i^*$ = {:.2f}".format(phi_i_star)

        ax.set_title(title)

        plt.savefig(all_imgformat.format(probe))
        plt.close()

        # Store data
        phi_i_stars[probe] = phi_i_star

    # Save phi_i_star data and errors to file
    phi_i_star_data = dict()
    phi_i_star_data["phi_i_stars"] = phi_i_stars

    with open(pklfile, "wb") as outfile:
        pickle.dump(phi_i_star_data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs="+", help="phi values to read (phi=0 must be first)")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-structfile", help="path to structure file (.pdb, .gro, .tpr)")
    parser.add_argument(
        "-trajformat",
        help="format of trajectory (.xtc, .pdb) file, with {phi} placeholders for phi value. Missing placeholders are ignored.",
    )
    parser.add_argument(
        "-skip", type=int, default=1, help="Interval between frames when reading trajectory (default=1)"
    )
    parser.add_argument("-probe_selection", help="MDAnalysis selection string for probe volume atoms")
    parser.add_argument("-probe_radius", type=float, default=6, help="Probe volume radius (in A, default=6)")
    parser.add_argument(
        "-solvent_selection",
        type=str,
        default="name OW",
        help="MDAnalysis selection string for solvent atoms (default='name OW')",
    )
    parser.add_argument(
        "-all_imgformat",
        default="phi_i_star_h{}.png",
        help="format of phi_i* output images for all probe atoms, with {} placeholder for probe atom index (default=phi_i_star_h{}.png)",
    )
    parser.add_argument(
        "-all_watersformat",
        default="ni_phi{phi}.pkl",
        help="format of pkl files containing TimeSeries probe waters, with {phi} placeholder for phi value (default=ni_phi{phi}.pkl).",
    )
    parser.add_argument(
        "-pklfile", default="phi_i_stars.pkl", help="output file to dump phi_i* data to (.pkl, default=phi_i_stars.pkl)"
    )
    parser.add_argument("--reload", action="store_true", default=False)

    a = parser.parse_args()

    phi_i_star(
        phivals=a.phi,
        start_time=a.start,
        structfile=a.structfile,
        trajformat=a.trajformat,
        skip=a.skip,
        probe_selection=a.probe_selection,
        probe_radius=a.probe_radius,
        solvent_selection=a.solvent_selection,
        all_imgformat=a.all_imgformat,
        all_watersformat=a.all_watersformat,
        pklfile=a.pklfile,
        reload=a.reload,
    )
