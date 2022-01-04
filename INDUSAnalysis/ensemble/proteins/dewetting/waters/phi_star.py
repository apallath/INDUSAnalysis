"""
Plots Nv v/s phi and phi* for dewetting simulation.
"""
# TODO: Bootstrapping for error estimation in case of a single run.

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.interpolate import UnivariateSpline

from INDUSAnalysis.timeseries import TimeSeriesAnalysis


def phi_star(phivals: list,
             runs: list,
             start_time: int,
             calc_dir: str = "./",
             Ntw_format: str = "",
             imgfile: str = "phi_star.png",
             phi_star_method="absolute"):

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

    # Disallow zero errors
    for idx, val in enumerate(yerr):
        if val < 1e-3:
            yerr[idx] = 1e-3

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # Define phi* value in terms of ydata
    if phi_star_method == 'absolute':
        phi_star_yval = ydata[0] / 2
    elif phi_star_method == 'relative':
        phi_star_yval = (ydata[0] + ydata[-1]) / 2

    # Plot original data
    ax.errorbar(xdata, ydata, yerr=yerr,
                barsabove=True, capsize=3.0, linestyle=":", marker="s", markersize=3, fillstyle="none")
    ax.fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.3)

    ax.axhline(phi_star_yval)

    yweights = 1 / yerr
    spline = UnivariateSpline(xdata, ydata, w=yweights, k=3, s=0.5 * len(ydata))  # Cubic spline

    x_spline_data = np.linspace(min(phivals), max(phivals), 100)
    y_spline_data = spline(x_spline_data)

    ax.plot(x_spline_data, y_spline_data)

    # Find phi_i_star
    phi_star_spline_idx = np.abs(y_spline_data - phi_star_yval).argmin()
    ax.plot(x_spline_data[phi_star_spline_idx], y_spline_data[phi_star_spline_idx], 'x')
    phi_star = x_spline_data[phi_star_spline_idx]

    # Title
    title = r"$\phi^*$ = {:.2f} kJ/mol".format(phi_star)
    ax.set_title(title)

    x_minor_locator = AutoMinorLocator(10)
    y_minor_locator = AutoMinorLocator(10)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    plt.savefig(imgfile, format="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read (enter 0 in case of no runs)")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("-Ntw_format", help="format of .pkl file containing Ntw, with {phi} placeholders for phi value and {run} placeholders for run value. Missing placeholders are ignored.")
    parser.add_argument("-imgfile", help="output image (default=phi_star.png)")
    parser.add_argument("-phi_star_method", default="absolute", help="calculate phi* as when waters dip below 1/2 of native (absolute) or below midpoint of native and last dewetted simulation (relative)")

    a = parser.parse_args()

    phi_star(a.phi, a.runs, a.start, a.calc_dir, a.Ntw_format, a.imgfile, a.phi_star_method)
