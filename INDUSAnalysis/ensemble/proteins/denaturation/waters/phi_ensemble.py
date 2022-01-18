"""
Plots Nv v/s phi for unfolding and/or folding simulation.
"""
import argparse
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from INDUSAnalysis.lib.collective import phi_to_P, P_to_phi
from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis


def phi_ensemble(phivals: list,
                 runs: list,
                 start_time: int,
                 calc_dir: str = "./",
                 plot_fwd: bool = True,
                 fwd_Ntw_format: str = "",
                 plot_rev: bool = True,
                 rev_Ntw_format: str = "",
                 imgfile: str = "phi_ensemble.png",
                 P0=1,
                 no_pressure=False,
                 invert_signs=False):

    nruns = len(runs)

    tsa = TimeSeriesAnalysis()

    if plot_fwd:
        # Store mean waters for each forward simulation
        meanwaters = np.zeros((len(phivals), nruns))
        varwaters = np.zeros((len(phivals), nruns))

    if plot_rev:
        # Store mean waters for each forward simulation
        meanwaters_rev = np.zeros((len(phivals), nruns))
        varwaters_rev = np.zeros((len(phivals), nruns))

    # Read data
    for phi_idx, phi in enumerate(phivals):
        for run_idx, run in enumerate(runs):
            if plot_fwd:
                ts = tsa.load_TimeSeries(calc_dir + fwd_Ntw_format.format(phi=phi, run=run))
                meanw = ts[start_time:].mean(axis=0)
                stdw = ts[start_time:].std(axis=0)
                varw = stdw ** 2

                meanwaters[phi_idx, run_idx] = meanw
                varwaters[phi_idx, run_idx] = varw

            if plot_rev:
                tsr = tsa.load_TimeSeries(calc_dir + rev_Ntw_format.format(phi=phi, run=run))
                meanwr = tsr[start_time:].mean(axis=0)
                stdwr = tsr[start_time:].std(axis=0)
                varwr = stdwr ** 2

                meanwaters_rev[phi_idx, run_idx] = meanwr
                varwaters_rev[phi_idx, run_idx] = varwr

    # Plot <N_v> v/s phi
    if plot_fwd:
        mean_meanwaters = np.mean(meanwaters, axis=1)
        std_meanwaters = np.std(meanwaters, axis=1)

    if plot_rev:
        mean_meanwaters_rev = np.mean(meanwaters_rev, axis=1)
        std_meanwaters_rev = np.std(meanwaters_rev, axis=1)

    """Plot waters"""
    # CEMB grant figure edits
    # TODO: Revert
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    # fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    if invert_signs:
        phivals = -np.array([float(phi) for phi in phivals])
    else:
        phivals = np.array([float(phi) for phi in phivals])

    order = np.argsort(phivals)

    if plot_fwd:
        ax.errorbar(np.array(phivals)[order], mean_meanwaters[order], yerr=std_meanwaters[order],
                    barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none",
                    label="Biased")
        ax.fill_between(np.array(phivals)[order], mean_meanwaters[order] + std_meanwaters[order], mean_meanwaters[order] - std_meanwaters[order],
                        alpha=0.5)

    if plot_rev:
        ax.errorbar(np.array(phivals)[order], mean_meanwaters_rev[order], yerr=std_meanwaters_rev[order],
                    barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none",
                    label="Biasing switched off")
        ax.fill_between(np.array(phivals)[order], mean_meanwaters_rev[order] + std_meanwaters_rev[order], mean_meanwaters_rev[order] - std_meanwaters_rev[order],
                        alpha=0.5)

    if plot_fwd:
        ax.axhline(mean_meanwaters[0], color="green", linestyle=":", label="Native state")
        ax.axhspan(mean_meanwaters[0] - std_meanwaters[0], mean_meanwaters[0] + std_meanwaters[0], alpha=0.5,
                   color="green")
    elif not plot_fwd and plot_rev:
        ax.axhline(mean_meanwaters_rev[0], color="green", linestyle=":", label="Native state")
        ax.axhspan(mean_meanwaters_rev[0] - std_meanwaters_rev[0], mean_meanwaters_rev[0] + std_meanwaters_rev[0], alpha=0.5,
                   color="green")
    else:
        pass

    ax.set_xlabel(r"$\phi$ (kJ/mol)")
    ax.set_ylabel(r"$\langle \tilde{N}_v \rangle_\phi$")
    ax.legend()

    # CEMB grant edits
    # TODO: Revert
    x_minor_locator = AutoMinorLocator(10)
    y_minor_locator = AutoMinorLocator(10)
    #ax.xaxis.set_minor_locator(x_minor_locator)
    #ax.yaxis.set_minor_locator(y_minor_locator)
    #ax.grid(which='major', linestyle='-')
    #ax.grid(which='minor', linestyle=':')

    if not no_pressure:
        phi_to_P_custom = partial(phi_to_P, P0=float(P0))
        P_to_phi_custom = partial(P_to_phi, P0=float(P0))

        secax = ax.secondary_xaxis('top', functions=(phi_to_P_custom, P_to_phi_custom))
        secax.set_xlabel(r"Effective hydration shell pressure, $P$ (kbar)")

    plt.savefig(imgfile, format="png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi for unfolding or folding simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("--plot_fwd", action="store_true")
    parser.add_argument("-fwd_Ntw_format", help="format of .pkl file containing fwd Ntw, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("--plot_rev", action="store_true")
    parser.add_argument("-rev_Ntw_format", help="format of .pkl file containing rev Ntw, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-imgfile", help="output image (default=phi_ensemble.png)")
    parser.add_argument("-P0", default=1, help="simulation pressure, in bar (default=1)")
    parser.add_argument("--no_pressure", action="store_true", default=False)
    parser.add_argument("--invert_signs", action="store_true", default=False)

    a = parser.parse_args()

    phi_ensemble(a.phi, a.runs, a.start, a.calc_dir, a.plot_fwd, a.fwd_Ntw_format, a.plot_rev, a.rev_Ntw_format, a.imgfile, a.P0, a.no_pressure, a.invert_signs)
