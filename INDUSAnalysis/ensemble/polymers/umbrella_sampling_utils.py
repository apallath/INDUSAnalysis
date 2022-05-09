"""
Functions to help perform umbrella sampling calculations.

This module can also be called as a stand-alone script, given command line options.
"""
import argparse

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from INDUSAnalysis import timeseries
from INDUSAnalysis.indus_waters import WatersAnalysis


def estimate_kappa(datf: str, temp: float, start_time: float = 0, end_time: float = None):
    """
    Estimates kappa and corresponding deltan based on var(N~) from an unbiased INDUS simulation.

    beta * kappa = alpha / var(Nt), alpha = 2 to 5

    deltan = 4 * sqrt(1 + alpha) / alpha * sqrt(var(Nt))

    Args:
        datf (str): Path to INDUS waters data file.
        temp (float): Simulation temperature (in K).
        start_time (float): Time to begin computation of variance at.
        end_time (float): Time to end computation of variance at.
    """
    ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(datf)
    var_Ntw = ts_Ntw[start_time:end_time].std() ** 2
    alphas = np.array([2, 3, 4, 5])
    kappas = 8.314 / 1000 * temp / var_Ntw * alphas
    deltas = 4 * np.sqrt((1 + alphas)) / alphas * np.sqrt(var_Ntw)

    print("alpha\tkappa\tdelta N*")
    for i in range(len(alphas)):
        if alphas[i] == 3:
            # recommended values
            print("{:.2f} {:.5f} kJ/mol {:.2f} ***".format(alphas[i], kappas[i], deltas[i]))
        else:
            print("{:.2f} {:.5f} kJ/mol {:.2f}".format(alphas[i], kappas[i], deltas[i]))


def estimate_min_max_Nstar(datf: str, start_time: float = 0, end_time: float = None, true_Nmin: float = 0, true_Nmax: float = 0):
    """
    Estimates the minimum and maximum values of N* to set to get sampling in the desired range (true_Nmin, true_Nmax)

    True biased distribution is centered around a value of N that is in-between the unbiased average <N>0 and chosen N* (Rego 2020, ProQuest).

    Args:
        datf (str): Path to INDUS waters data file.
        start_time (float): Time to begin computation of variance at.
        end_time (float): Time to end computation of variance at.
    """
    ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(datf)
    mean_Ntw = ts_Ntw[start_time:end_time].mean()
    alphas = np.array([2, 3, 4, 5])
    nstar_min = (true_Nmin * (1 + alphas) - mean_Ntw) / alphas
    nstar_max = (true_Nmax * (1 + alphas) - mean_Ntw) / alphas

    print("alpha\tmin N*\tmax N*")
    for i in range(len(alphas)):
        if alphas[i] == 3:
            # recommended values
            print("{:.2f} {:.2f} {:.2f} ***".format(alphas[i], nstar_min[i], nstar_max[i]))
        else:
            print("{:.2f} {:.2f} {:.2f}".format(alphas[i], nstar_min[i], nstar_max[i]))


def _read_energy_xvg(xvg_file):
    times = []
    pots = []
    with open(xvg_file, "r") as f:
        for line in f:
            if line.strip()[0] == '#' or line.strip()[0] == '@':
                pass
            else:
                time, pot = [float(n) for n in line.strip().split()]
                times.append(time)
                pots.append(pot)
    return timeseries.TimeSeries(times, pots, ["Potential Energy"])


def energy_overlap(temps: list,
                   xvgfs: list,
                   e_bin_min: float,
                   e_bin_max: float,
                   nbins: int,
                   start_time: float = 0,
                   end_time: float = None,
                   out_imgfile: str = 'hist.png'):
    """
    Plots histogram of energies, to check if overlap exists between them.

    Args:
        temps: List of temperatures.
        xvgfs: List of paths to .xvg files containing simulation energies.
        e_bin_min: Minimum energy for binning.
        e_bin_max: Maximum energy for binning.
        nbins: Number of bins.
        start_time: Time to start reading energies from (post-equilibration).
        end_time: Time to stop reading energies at.
        out_imgfile: Path of file to save output histogram to.
    """
    assert(len(temps) == len(xvgfs))

    # Prepare plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    # Setup normalization and colormap
    print(temps)
    normalize = mcolors.Normalize(vmin=temps[0], vmax=temps[-1])
    colormap = cm.rainbow

    bin_points = np.linspace(e_bin_min, e_bin_max, nbins)

    for t in range(len(temps)):
        ts = _read_energy_xvg(xvgfs[t])
        hist, edges = np.histogram(ts[start_time:end_time].data_array, bins=bin_points, density=True)
        x = 0.5 * (edges[1:] + edges[:-1])
        y = hist
        ax.plot(x, y, color=colormap(normalize(temps[t])), label="%d" % temps[t])
        ax.fill_between(x, 0, y, color=colormap(normalize(temps[t])), alpha=0.4)

    # Display colorbar
    # scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
    # scalarmappable.set_array(np.array(temps))
    # fig.colorbar(scalarmappable, label=r"$N*$")

    # Display legend
    ax.legend()

    # Save
    fig.savefig(out_imgfile, format="png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument("calc_type", help="Type of calculation (options = est_kappa, est_minmax, energy_overlap)")

    est_kappa_args = parser.add_argument_group("kappa estimation arguments")
    est_kappa_args.add_argument("-watersf", help="INDUS waters data file")
    est_kappa_args.add_argument("-temp", type=float, help="Simulation temperature")
    est_kappa_args.add_argument("-tstart", type=float, help="Time to begin computation of variance at", default=0)
    est_kappa_args.add_argument("-tend", type=float, help="Time to end computation of variance at", default=None)

    est_minmax_args = parser.add_argument_group("min-max N* estimation arguments")
    est_minmax_args.add_argument("-watersf", help="INDUS waters data file")
    est_minmax_args.add_argument("-tstart", type=float, help="Time to begin computation of variance at", default=0)
    est_minmax_args.add_argument("-tend", type=float, help="Time to end computation of variance at", default=None)
    est_minmax_args.add_argument("-true_Nmin", type=float, help="Desired (true) value of Nmin", default=0)
    est_minmax_args.add_argument("-true_Nmax", type=float, help="Desired (true) value of Nmax", default=0)

    energy_overlap_args = parser.add_argument_group("energy overlap histogram arguments")
    energy_overlap_args.add_argument("-temps", type=float, help="Simulation temperature", action='append', nargs='+')
    energy_overlap_args.add_argument("-xvgfs", type=str, help="XVG files containing potential energy data", action='append', nargs='+')
    energy_overlap_args.add_argument("-e_bin_min", type=float, help="Minimum energy for binning")
    energy_overlap_args.add_argument("-e_bin_max", type=float, help="Maximum energy for binning")
    energy_overlap_args.add_argument("-nbins", type=int, help="Number of energy bins")
    energy_overlap_args.add_argument("-tstart", type=float, help="Time to begin binning at", default=0)
    energy_overlap_args.add_argument("-tend", type=float, help="Time to end binning at", default=None)
    energy_overlap_args.add_argument("-outfile", type=str, help="Path of file to save output histogram to")


    args = parser.parse_args()

    if args.calc_type == "est_kappa":
        estimate_kappa(args.watersf, args.temp, args.tstart, args.tend)
    if args.calc_type == "est_minmax":
        estimate_min_max_Nstar(args.watersf, args.tstart, args.tend, args.true_Nmin, args.true_Nmax)
    elif args.calc_type == "energy_overlap":
        energy_overlap(args.temps[0], args.xvgfs[0], args.e_bin_min, args.e_bin_max, args.nbins, args.tstart, args.tend, args.outfile)
