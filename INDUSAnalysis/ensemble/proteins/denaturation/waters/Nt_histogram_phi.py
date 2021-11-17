"""
Plots histogram of Nv v/s phi for unfolding simulation
"""
import argparse

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import MDAnalysis as mda
import numpy as np

from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis

def phi_ensemble(phivals: list,
                 runs: list,
                 start_time: int,
                 calc_dir: str = "./",
                 Ntw_format: str = "",
                 imgfile: str = "Nt_hist_phi.png",
                 Nmin = 0,
                 Nmax = 3000,
                 Nbins = 200):

    Ntw_range = [Nmin, Nmax]
    Ntw_bins = Nbins

    phivals_numeric = [float(phi) for phi in phivals]
    nruns = len(runs)

    tsa = TimeSeriesAnalysis()

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Set up normalization and colormap
    normalize = mcolors.Normalize(vmin=min(phivals_numeric), vmax=max(phivals_numeric))
    colormap = cm.rainbow

    for phi_idx, phi in enumerate(phivals):

        waters = []

        for run_idx, run in enumerate(runs):
            ts = tsa.load_TimeSeries(calc_dir + Ntw_format.format(phi=phi, run=run))
            ts = ts[start_time:]
            waters.append(ts.data_array)

        waters = np.array(waters)

        hist, edges = np.histogram(waters, bins=Ntw_bins, range=Ntw_range, density=True)
        x = 0.5 * (edges[1:] + edges[:-1])
        y = hist
        ax.plot(x, y, color=colormap(normalize(float(phi))))
        ax.fill_between(x, 0, y, color=colormap(normalize(float(phi))), alpha=0.4)

    ax.set_xlabel(r"$\tilde{N}$")

    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    # Display colorbar
    scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappable.set_array(np.array(phivals_numeric))
    fig.colorbar(scalarmappable, label=r"$\phi$ (kJ/mol)")

    plt.savefig(imgfile, format="png")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi histograms.")
    parser.add_argument("-phi", type=str, nargs='+', help="phi values to read (phi=0 must be first)")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start binning histograms")
    parser.add_argument("-calc_dir", help="directory containing hydration OPs extracted by INDUSAnalysis")
    parser.add_argument("-Ntw_format", help="format of .pkl file containing Ntw, with {phi} placeholders for phi value and {run} placeholders for run value")
    parser.add_argument("-imgfile", help="output image (default=phi_star.png)")
    parser.add_argument("-Nmin", type=int, default=0, help="min value of N~ (default=0)")
    parser.add_argument("-Nmax", type=int, default=3000, help="max value of N~ (default=3000)")
    parser.add_argument("-Nbins", type=int, default=200, help="number of N~ bins for histogram (default=200)")

    a = parser.parse_args()

    phi_ensemble(a.phi, a.runs, a.start, a.calc_dir, a.Ntw_format, a.imgfile, a.Nmin, a.Nmax, a.Nbins)
