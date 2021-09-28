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
    Estimates kappa based on var(N~) from an unbiased INDUS simulation.

    beta * kappa = 2 / var(Nt) to 5 / var(Nt).

    Args:
        datf (str): Path to INDUS waters data file.
        temp (float): Simulation temperature (in K).
        start_time (float): Time to begin computation of variance at.
        end_time (float): Time to end computation of variance at.

    Returns:
        kappa_range (tuple): Estimated range of kappa in kJ/mol (kappa_lower: float, kappa_upper: float)
    """
    ts_N, ts_Ntw, _ = WatersAnalysis.read_waters(datf)
    var_Ntw = ts_Ntw[start_time:end_time].std() ** 2
    return (8.314 / 1000 * temp * 2 / var_Ntw, 8.314 / 1000 * temp * 5 / var_Ntw)


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

    parser.add_argument("calc_type", help="Type of calculation (options = est_kappa)")

    est_kappa_args = parser.add_argument_group("kappa estimation arguments")
    est_kappa_args.add_argument("-watersf", help="INDUS waters data file")
    est_kappa_args.add_argument("-temp", type=float, help="Simulation temperature")
    est_kappa_args.add_argument("-tstart", type=float, help="Time to begin computation of variance at", default=0)
    est_kappa_args.add_argument("-tend", type=float, help="Time to end computation of variance at", default=None)

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
        print(estimate_kappa(args.watersf, args.temp, args.tstart, args.tend))
    elif args.calc_type == "energy_overlap":
        energy_overlap(args.temps[0], args.xvgfs[0], args.e_bin_min, args.e_bin_max, args.nbins, args.tstart, args.tend, args.outfile)
