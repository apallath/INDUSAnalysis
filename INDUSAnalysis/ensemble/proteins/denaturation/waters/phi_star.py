"""
Plots Nv v/s phi and phi* for unfolding or folding simulation.
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from INDUSAnalysis.lib.collective import fit_integrated_step_gaussian, \
    integrated_step_gaussian, phi_to_P, P_to_phi
from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis


def phi_ensemble(phivals: list,
                 runs: list,
                 start_time: int,
                 calc_dir: str = "./",
                 Ntw_format: str = "",
                 imgfile: str = "phi_star.png",
                 D_by_A_guess = 5,
                 E_guess = 0.05):

    nruns = len(runs)

    tsa = TimeSeriesAnalysis()

    meanwaters = np.zeros((len(phivals), nruns))
    varwaters = np.zeros((len(phivals), nruns))

    for phi_idx, phi in enumerate(phivals):
        for run_idx, run in enumerate(runs):
            ts = tsa.load_TimeSeries(calc_dir + Ntw_format.format(phi=phi, run=run))
            meanw = ts[start_time:].mean(axis=0)
            stdw = ts[start_time:].std(axis=0)
            varw = stdw ** 2

            meanwaters[phi_idx, run_idx] = meanw
            varwaters[phi_idx, run_idx] = varw

    mean_meanwaters = np.mean(meanwaters, axis=1)
    std_meanwaters = np.std(meanwaters, axis=1)

    """Plot waters and phi*"""
    phivals = np.array([float(phi) for phi in phivals])

    order = np.argsort(phivals)

    xdata = np.array(phivals)[order]
    ydata = np.array(mean_meanwaters)[order]
    yerr = np.array(std_meanwaters)[order]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    # Plot original data
    ax.errorbar(np.array(phivals)[order], mean_meanwaters[order], yerr=std_meanwaters[order],
                barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none")
    ax.fill_between(np.array(phivals)[order], mean_meanwaters[order] + std_meanwaters[order], mean_meanwaters[order] - std_meanwaters[order],
                    alpha=0.5)

    # Initial parameter guesses
    A_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    B_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])

    dy_data = ydata[1:] - ydata[0:-1]
    dx_data = xdata[1:] - xdata[0:-1]
    dydx_data = dy_data / dx_data
    C_guess_idx = np.argmin(dydx_data) + 1

    C_guess = xdata[C_guess_idx]    # This is phi*
    D_guess = D_by_A_guess * A_guess
    E_guess = E_guess                # Spread
    F_guess = ydata[C_guess_idx]

    p_guess = [A_guess, B_guess, C_guess, D_guess, E_guess, F_guess]

    popt, perr, chi_sq = fit_integrated_step_gaussian(xdata, ydata, yerr, p_guess)

    x_fit_data = np.linspace(min(phivals), max(phivals), 100)
    y_fit_data = integrated_step_gaussian(x_fit_data, *popt)
    ax.plot(x_fit_data, y_fit_data)
    ax.axvline(x=popt[2], linestyle=":", color="C3")
    ax.axvspan(popt[2] - perr[2], popt[2] + perr[2], alpha=0.3, color='C3')

    ax.set_xlabel(r"$\phi$ (kJ/mol)")
    ax.set_ylabel(r"$\langle \tilde{N_v} \rangle$")
    ax.text(popt[2], 0.8 * max(ydata),
            r"$\phi^*$ = {:.2f} $\pm$ {:.2f} kJ/mol".format(popt[2], perr[2]), bbox=dict(facecolor='C3', alpha=0.5))

    x_minor_locator = AutoMinorLocator(10)
    y_minor_locator = AutoMinorLocator(10)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    secax = ax.secondary_xaxis('top', functions=(phi_to_P, P_to_phi))
    secax.set_xlabel(r"Effective hydration shell pressure, $P$ (kbar)")

    plt.savefig(imgfile, format="png")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("-Ntw_format", help="format of .pkl file containing Ntw, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-imgfile", help="output image (default=phi_star.png)")
    parser.add_argument("-D_by_A_guess", default=5, help="initial guess for two-state model D/A parameter (default=5)")
    parser.add_argument("-E_guess", default=0.05, help="initial guess for two-state model E parameter (default=0.05)")

    a = parser.parse_args()

    phi_ensemble(a.phi, a.runs, a.start, a.calc_dir, a.Ntw_format, a.imgfile, a.D_by_A_guess, a.E_guess)
