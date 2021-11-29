"""
Plots histograms of <N_i~> (binning across i) for native (folded) state and unfolded state.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import MDAnalysis as mda

from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.timeseries import TimeSeriesAnalysis

def buried_surface_classifier(struct_file, traj_file, ni_native_format, ni_unfolded_format, runs, start, nirange, nbins, buried_cut, imgfile, classfile, classpdb):
    tsa = TimeSeriesAnalysis()

    ni_range = [nirange[0], nirange[1]]
    ni_bins = nbins

    """Prepare histogram"""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_xlabel(r"$n_i$")
    ax.set_ylabel("Number of h. atoms")

    u = mda.Universe(struct_file)
    protein_heavy = u.select_atoms("protein and not name H*")
    protein_heavy_indices = protein_heavy.atoms.indices

    for phiidx in range(2):
        if phiidx == 0:
            state_label = "Native"
            ni_format = ni_native_format
        elif phiidx == 1:
            state_label = "Unfolded"
            ni_format = ni_unfolded_format

        heavy_mean_waters = []

        for runidx, run in enumerate(runs):
            ts = tsa.load_TimeSeries(ni_format.format(run=run))
            ts = ts[start:]
            hwaters = ts.data_array[:, protein_heavy_indices]
            meanhwaters = hwaters.mean(axis=0)
            heavy_mean_waters.append(meanhwaters)

        heavy_mean_waters = np.array(heavy_mean_waters)

        if phiidx == 0:
            native_mean_waters = heavy_mean_waters

        hist_heavy_waters = np.zeros((len(runs), ni_bins))
        hist_edges = None

        for runidx, run in enumerate(runs):
            if ni_bins == ni_range[1] and ni_range[0] == 0:
                hist = np.bincount(heavy_mean_waters[runidx].astype(int))
                hist_edges = np.array(range(int(ni_range[1])))
                hist_heavy_waters[runidx, 0:len(hist)] = hist
            else:
                hist, edges = np.histogram(heavy_mean_waters[runidx], bins=ni_bins, range=ni_range)
                x = 0.5 * (edges[1:] + edges[:-1])
                y = hist
                hist_heavy_waters[runidx, :] = y
                hist_edges = x

        hist_heavy_waters_mean = hist_heavy_waters.mean(axis=0)
        hist_heavy_waters_std = hist_heavy_waters.std(axis=0)

        ax.errorbar(hist_edges, hist_heavy_waters_mean, yerr=hist_heavy_waters_std,
                    barsabove=True, capsize=3.0, linestyle="-", marker="s", markersize=3, fillstyle="none",
                    label="{} state".format(state_label))
        ax.fill_between(hist_edges, hist_heavy_waters_mean + hist_heavy_waters_std, hist_heavy_waters_mean - hist_heavy_waters_std, alpha=0.5)

    ax.axvline(x=buried_cut, label='Buried cutoff = {}'.format(buried_cut))

    minor_locator = AutoMinorLocator(10)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    ax.legend()
    plt.savefig(imgfile)

    """Assign buried/surface labels to each heavy atom based on its folded state number of waters"""
    # Compute and save to .npy file
    print(native_mean_waters.shape)
    native_waters = native_mean_waters.mean(axis=0)
    print(native_waters.shape)
    classes = native_waters <= buried_cut
    print("{} / {} = {:.2f} % buried".format(classes.sum(), len(classes), classes.sum() / len(classes) * 100))
    np.save(classfile, classes)

    # Write to PDB
    u = mda.Universe(struct_file, traj_file)
    u.add_TopologyAttr('tempfactors')
    protein_heavy = u.select_atoms("protein and not name H*")
    protein_heavy.atoms.tempfactors = classes
    with mda.Writer(classpdb, protein_heavy.n_atoms) as W:
        W.write(protein_heavy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Nv v/s phi and phi* for simulation.")
    parser.add_argument("-struct_file", help="path to structure file (.gro or .tpr)")
    parser.add_argument("-traj_file", help="path to trajectory file (.xtc) to extract first frame from")
    parser.add_argument("-ni_native_format", help="format of .pkl file containing native probe waters with {run} placeholders for run value")
    parser.add_argument("-ni_unfolded_format", help="format of .pkl file containing unfolded probe waters, relative to calc_dir, with {run} placeholders for run value")
    parser.add_argument("-runs", type=int, nargs='+', help="runs to read")
    parser.add_argument("-start", type=int, help="time (ps) to start computing averages")
    parser.add_argument("-nirange", type=float, nargs=2, help="min and max value of ni to bin over")
    parser.add_argument("-nbins", type=int, help="number of bins")
    parser.add_argument("-buried_cut", type=int, help="cutoff folded state <ni> at or below which (<=) atom i is classified as a buried atom")
    parser.add_argument("-imgfile", default="buried_surface.png", help="output image (default=buried_surface.png)")
    parser.add_argument("-classfile", default="buried_surface_indicator.npy", help="output numpy file buried v/s surface classes for heavy atoms(default=buried_surface_indicator.npy)")
    parser.add_argument("-classpdb", default="buried_surface_indicator.pdb", help="output PDB buried v/s surface classes as bfactors (default=buried_surface_indicator.pdb)")

    a = parser.parse_args()

    buried_surface_classifier(a.struct_file, a.traj_file, a.ni_native_format, a.ni_unfolded_format, a.runs, a.start, a.nirange, a.nbins, a.buried_cut, a.imgfile, a.classfile, a.classpdb)
