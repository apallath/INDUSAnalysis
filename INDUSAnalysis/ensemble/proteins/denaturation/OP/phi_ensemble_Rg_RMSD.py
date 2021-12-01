"""
Plots Rg and RMSD v/s phi for unfolding and/or folding simulation.
"""
import argparse
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from INDUSAnalysis.lib.collective import phi_to_P, P_to_phi
from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis


def phi_ensemble(phivals_str: list,
                 runs: list,
                 start_time: int,
                 calc_dir: str,
                 plot_fwd: bool,
                 fwd_OP_formats: list,
                 plot_rev: bool,
                 rev_OP_formats: list,
                 imgfiles: list,
                 P0=1):

    nruns = len(runs)

    tsa = TimeSeriesAnalysis()

    ############################################################################
    # Analysis
    ############################################################################
    OP_names = ["R_g", "RMSD"]

    for opidx in range(2):

        if plot_fwd:
            # Store mean OPs for each forward simulation
            meanOPs = np.zeros((len(phivals_str), nruns))
            varOPs = np.zeros((len(phivals_str), nruns))

        if plot_rev:
            # Store mean OPs for each forward simulation
            meanOPs_rev = np.zeros((len(phivals_str), nruns))
            varOPs_rev = np.zeros((len(phivals_str), nruns))

        # Read data
        for phi_idx, phi in enumerate(phivals_str):
            for run_idx, run in enumerate(runs):
                if plot_fwd:
                    ts = tsa.load_TimeSeries(calc_dir + fwd_OP_formats[opidx].format(phi=phi, run=run))
                    meanw = ts[start_time:].mean(axis=0)
                    stdw = ts[start_time:].std(axis=0)
                    varw = stdw ** 2

                    meanOPs[phi_idx, run_idx] = meanw
                    varOPs[phi_idx, run_idx] = varw

                if plot_rev:
                    tsr = tsa.load_TimeSeries(calc_dir + rev_OP_formats[opidx].format(phi=phi, run=run))
                    meanwr = tsr[start_time:].mean(axis=0)
                    stdwr = tsr[start_time:].std(axis=0)
                    varwr = stdwr ** 2

                    meanOPs_rev[phi_idx, run_idx] = meanwr
                    varOPs_rev[phi_idx, run_idx] = varwr

        # Plot <N_v> v/s phi
        if plot_fwd:
            mean_meanOPs = np.mean(meanOPs, axis=1)
            std_meanOPs = np.std(meanOPs, axis=1)

        if plot_rev:
            mean_meanOPs_rev = np.mean(meanOPs_rev, axis=1)
            std_meanOPs_rev = np.std(meanOPs_rev, axis=1)

        """Plot OPs"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        phivals = np.array([float(phi) for phi in phivals_str])

        order = np.argsort(phivals)

        if plot_fwd:
            ax.errorbar(np.array(phivals)[order], mean_meanOPs[order], yerr=std_meanOPs[order],
                        barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none",
                        label="Biased")
            ax.fill_between(np.array(phivals)[order], mean_meanOPs[order] + std_meanOPs[order], mean_meanOPs[order] - std_meanOPs[order],
                            alpha=0.5)

        if plot_rev:
            ax.errorbar(np.array(phivals)[order], mean_meanOPs_rev[order], yerr=std_meanOPs_rev[order],
                        barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none",
                        label="Biasing switched off")
            ax.fill_between(np.array(phivals)[order], mean_meanOPs_rev[order] + std_meanOPs_rev[order], mean_meanOPs_rev[order] - std_meanOPs_rev[order],
                            alpha=0.5)

        if plot_fwd:
            ax.axhline(mean_meanOPs[0], color="green", linestyle=":", label="Folded state")
            ax.axhspan(mean_meanOPs[0] - std_meanOPs[0], mean_meanOPs[0] + std_meanOPs[0], alpha=0.5,
                       color="green")
        elif not plot_fwd and plot_rev:
            ax.axhline(mean_meanOPs_rev[0], color="green", linestyle=":", label="Folded state")
            ax.axhspan(mean_meanOPs_rev[0] - std_meanOPs_rev[0], mean_meanOPs_rev[0] + std_meanOPs_rev[0], alpha=0.5,
                       color="green")
        else:
            pass

        ax.set_xlabel(r"$\phi$ (kJ/mol)")
        ax.set_ylabel(r"$\langle {} \rangle$".format(OP_names[opidx]))
        ax.legend()

        x_minor_locator = AutoMinorLocator(10)
        y_minor_locator = AutoMinorLocator(10)
        ax.xaxis.set_minor_locator(x_minor_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle=':')

        print(P0)
        phi_to_P_custom = partial(phi_to_P, P0=float(P0))
        P_to_phi_custom = partial(P_to_phi, P0=float(P0))

        secax = ax.secondary_xaxis('top', functions=(phi_to_P_custom, P_to_phi_custom))
        secax.set_xlabel(r"Effective hydration shell pressure, $P$ (kbar)")

        plt.savefig(imgfiles[opidx], format="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi for unfolding or folding simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dir", help="directory containing OPs extracted by INDUSAnalysis")
    parser.add_argument("--plot_fwd", action="store_true")
    parser.add_argument("-fwd_OP_formats", nargs=2, help="formats of .pkl file containing fwd Rg and RMSD (space separated), with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("--plot_rev", action="store_true")
    parser.add_argument("-rev_OP_formats", nargs=2, help="format of .pkl file containing rev Rg and RMSD (space separated), with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-imgfiles", nargs=2, help="output images for Rg and RMSD (space separated)")
    parser.add_argument("-P0", default=1, help="simulation pressure, in bar (default=1)")

    a = parser.parse_args()

    phi_ensemble(a.phi, a.runs, a.start, a.calc_dir, a.plot_fwd, a.fwd_OP_formats, a.plot_rev, a.rev_OP_formats, a.imgfiles, a.P0)
