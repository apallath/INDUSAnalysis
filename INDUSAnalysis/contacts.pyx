"""
Defines class for analysing contacts along GROMACS simulation trajectory.

Extendable to add new types of contacts analyses.
"""

import copy
from itertools import combinations

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.align
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from tqdm import tqdm

from INDUSAnalysis import timeseries
from INDUSAnalysis.lib import profiling

"""Cython"""
cimport numpy as np


class ContactsAnalysis(timeseries.TimeSeriesAnalysis):
    def __init__(self):
        super().__init__()
        self.req_file_args.add_argument("structf", help="Topology or structure file (.tpr, .gro; .tpr required for atomic-sh)")
        self.req_file_args.add_argument("trajf", help="Compressed trajectory file (.xtc)")

        self.calc_args.add_argument("-method", help="Method for calculating contacts (atomic-h, atomic-sh; default=atomic-h)")
        self.calc_args.add_argument("-distcutoff", help="Distance cutoff for contacts, in A")
        self.calc_args.add_argument("-connthreshold", help="Connectivity threshold for contacts (definition varies by method)")
        self.calc_args.add_argument("-skip", help="Number of frames to skip between analyses (default = 1)")
        self.calc_args.add_argument("-bins", help="Number of bins for histogram (default = 20)")
        self.calc_args.add_argument("-refcontacts", help="Reference number of contacts for fraction (default = mean)")

        self.misc_args.add_argument("--verbose", action='store_true', help="Output progress of contacts calculation")

    def read_args(self):
        """
        Stores arguments from TimeSeries `args` parameter in class variables.
        """
        super().read_args()
        self.structf = self.args.structf
        self.trajf = self.args.trajf

        self.u = mda.Universe(self.structf, self.trajf)

        self.method = self.args.method
        if self.method is None:
            self.method = "atomic-h"

        self.opref = self.opref + "_" + self.method
        self.replotpref = self.replotpref + "_" + self.method

        self.distcutoff = self.args.distcutoff
        if self.distcutoff is not None:
            self.distcutoff = float(self.distcutoff)
        else:
            if self.method == "atomic-sh":
                self.distcutoff = 6.0

        self.connthreshold = self.args.connthreshold
        if self.connthreshold is not None:
            self.connthreshold = int(self.connthreshold)
        else:
            if self.method == "atomic-sh":
                self.connthreshold = 0  # TODO: Update

        self.refcontacts = self.args.refcontacts
        if self.refcontacts is not None:
            self.refcontacts = float(self.refcontacts)

        self.skip = self.args.skip
        if self.skip is not None:
            self.skip = int(self.skip)
        else:
            self.skip = 1

        self.bins = self.args.bins
        if self.bins is not None:
            self.bins = int(self.bins)
        else:
            self.bins = 20

        self.verbose = self.args.verbose

    def calc_trajcontacts(self, u, method, distcutoff, connthreshold, start_time, end_time, skip):
        """
        Calculates contacts between heavy atoms along a trajectory.

        Args:
            u (mda.Universe): Trajectory.
            method (str): Calculation method to use to compute contacts.
            distcutoff (float): Distance cutoff (in A).
            connthreshold (int): Connectivity threshold.
            start_time (float): Time to start averaging at.
            end_time (float): Time to end averaging at.
            skip (int): Resampling interval.

        Returns:
            {
                ts_contacts (timeseries.TimeSeries): TimeSeries objects containing total
                    number of contacts formed at each timestep.

                mean_contactmatrix(np.array): Array of shape (nheavy, nheavy) where
                    mean_contactmatrix[i,j] is ratio of number of timesteps where the contact
                    [i,j] is formed to the total number of timesteps.
            }

        Raises:
            ValueError if calculation method is not recognized.
        """
        if method == "atomic-h":
            return self.calc_trajcontacts_atomic_h(u, distcutoff, connthreshold, start_time, end_time, skip)
        elif method == "atomic-sh":
            return self.calc_trajcontacts_atomic_sh(u, distcutoff, connthreshold, start_time, end_time, skip)
        else:
            raise ValueError("Method not recognized")

    @profiling.timefunc
    def calc_trajcontacts_atomic_h(self, u, distcutoff, connthreshold, start_time, end_time, skip):
        """
        Calculates contacts between heavy atoms along trajectory.

        The connectivity threshold is the number of bonds heavy atoms
        have to be separated by on the shortest bond network path between them
        for the pair to be a candidate for contact formation. The distance cutoff
        is the distance between a candidate atomic pair within which it is
        considered to be a valid contact.

        Side chain heavy atoms i and j form a contact if
        N(i,j) > connthreshold and r(i,j) < distcutoff.
        """
        # MDAnalysis selection strings
        heavy_sel = "protein and not name H*"
        not_heavy_sel = "protein and name H*"

        # Select heavy atoms only
        protein_heavy = u.select_atoms(heavy_sel)
        nheavy = len(protein_heavy.atoms)

        start_index = None
        stop_index = None
        for tidx, ts in enumerate(u.trajectory):
            if start_index is None and ts.time >= start_time:
                start_index = tidx
            if ts.time == end_time:
                stop_index = tidx + 1

        # Select trajectory to average over
        utraj = u.trajectory[start_index:stop_index:skip]

        # Determine indices to exclude based on connectivity
        apsp, all_to_heavy = self.protein_heavy_APSP(u)

        if connthreshold < 0:
            raise ValueError("Connectivity threshold must be an integer value 0 or greater.")

        # Variables to store computed contacts to
        times = np.zeros(len(utraj))
        total_contacts = np.zeros(len(utraj))
        mean_contactmatrix = np.zeros((nheavy, nheavy))

        if self.verbose:
            pbar = tqdm(desc="Calculating contacts", total=len(utraj))

        for tidx, ts in enumerate(utraj):
            # Fast MDAnalysis distance matrix computation
            dmatrix = mda.lib.distances.distance_array(protein_heavy.positions, protein_heavy.positions)

            # Exclude i-j interactions below connectivity threshold from distance matrix
            for i in range(apsp.shape[0]):
                for j in range(apsp.shape[1]):
                    if i == j and apsp[i, j] > 0:
                        raise ValueError("Distance matrix is inconsistent: shortest path between same atom should be 0.")
                    if apsp[i, j] <= connthreshold:
                        dmatrix[i, j] = np.Inf

            # Impose distance cutoff
            contactmatrix = np.array(dmatrix < distcutoff)

            # Store timeseries
            times[tidx] = ts.time
            total_contacts[tidx] = np.sum(contactmatrix)

            # Add to mean
            mean_contactmatrix += contactmatrix

            if self.verbose:
                pbar.update(1)

        ts_contacts = timeseries.TimeSeries(times, total_contacts, labels=['Number of contacts'])
        mean_contactmatrix = mean_contactmatrix / len(utraj)

        return ts_contacts, mean_contactmatrix

    @profiling.timefunc
    def calc_trajcontacts_atomic_sh(self, u, distcutoff, connthreshold, start_time, end_time, skip):
        """
        Calculates contacts between side-chain heavy atoms along trajectory.

        The connectivity threshold is the number of bonds side chain heavy atoms
        have to be separated by on the shortest bond network path between them
        for the pair to be a candidate for contact formation. The distance cutoff
        is the distance between a candidate atomic pair within which it is
        considered to be a valid contact.

        Side chain heavy atoms i and j form a contact if
        N(i,j) > connthreshold and r(i,j) < distcutoff.
        """
        # MDAnalysis selection strings
        heavy_sel = "protein and not name H*"
        not_side_heavy_sel = "protein and (name N or name CA or name C or name O or name OC1 or name OC2 or name H*)"

        # Select heavy atoms only
        protein_heavy = u.select_atoms(heavy_sel)
        nheavy = len(protein_heavy.atoms)

        start_index = None
        stop_index = None
        for tidx, ts in enumerate(u.trajectory):
            if start_index is None and ts.time >= start_time:
                start_index = tidx
            if ts.time == end_time:
                stop_index = tidx + 1

        # Select trajectory to average over
        utraj = u.trajectory[start_index:stop_index:skip]

        # Determine indices to exclude based on connectivity
        apsp, all_to_heavy = self.protein_heavy_APSP(u)

        if connthreshold < 0:
            raise ValueError("Connectivity threshold must be an integer value 0 or greater.")

        # Determine indices to exclude because they are not side-chain-heavy
        not_sh_all_idx = self.u.select_atoms(not_side_heavy_sel).indices
        not_sh_heavy_idx = []
        for all_idx in not_sh_all_idx:
            try:
                not_sh_heavy_idx.append(all_to_heavy[all_idx])
            except KeyError:
                pass  # Not a heavy atom => already excluded

        # Variables to store computed contacts to
        times = np.zeros(len(utraj))
        total_contacts = np.zeros(len(utraj))
        mean_contactmatrix = np.zeros((nheavy, nheavy))

        if self.verbose:
            pbar = tqdm(desc="Calculating contacts", total=len(utraj))

        for tidx, ts in enumerate(utraj):
            # Fast MDAnalysis distance matrix computation
            dmatrix = mda.lib.distances.distance_array(protein_heavy.positions, protein_heavy.positions)

            # Exclude pairs containing non-side-chain-heavy atoms by setting their distances to infinity
            for i in not_sh_heavy_idx:
                dmatrix[i, :] = np.Inf
                dmatrix[:, i] = np.Inf

            # Exclude i-j interactions below connectivity threshold from distance matrix
            for i in range(apsp.shape[0]):
                for j in range(apsp.shape[1]):
                    if i == j and apsp[i, j] > 0:
                        raise ValueError("Distance matrix is inconsistent: shortest path between same atom should be 0.")
                    if apsp[i, j] <= connthreshold:
                        dmatrix[i, j] = np.Inf

            # Impose distance cutoff
            contactmatrix = np.array(dmatrix < distcutoff)

            # Store timeseries
            times[tidx] = ts.time
            total_contacts[tidx] = np.sum(contactmatrix)

            # Add to mean
            mean_contactmatrix += contactmatrix

            if self.verbose:
                pbar.update(1)

        ts_contacts = timeseries.TimeSeries(times, total_contacts, labels=['Number of contacts'])
        mean_contactmatrix = mean_contactmatrix / len(utraj)

        return ts_contacts, mean_contactmatrix

    def protein_heavy_APSP(self, u):
        """
        Constructs graph of protein-heavy atoms and calculates all-pairs-shortest-path
        distances using the Floyd-Warshall algorithm, assigning each bond an
        equal weight (of 1).

        Args:
            u (mda.Universe): Universe object containing all atoms with bond definitions.

        Returns:
            {
                D (np.array): Array of shape (nheavy, nheavy), where D[i,j] is the shortest
                    path distance between heavy atom i and heavy atom j.

                all_to_heavy (dict): Dictionary mapping atom i in the Universe to its heavy atom
                    index.
            }
        """
        # Connectivity graph
        protein_heavy = u.select_atoms("protein and not name H*")
        nheavy = len(protein_heavy)

        heavy_indices = protein_heavy.atoms.indices

        all_to_heavy = {}
        for heavyidx, allidx in enumerate(protein_heavy.atoms.indices):
            all_to_heavy[allidx] = heavyidx

        adj_matrix = np.zeros((nheavy, nheavy))

        for bond in protein_heavy.bonds:
            ati = bond.indices[0]
            atj = bond.indices[1]
            if ati in heavy_indices and atj in heavy_indices:
                heavyi = all_to_heavy[ati]
                heavyj = all_to_heavy[atj]
                adj_matrix[heavyi, heavyj] = 1
                adj_matrix[heavyj, heavyi] = 1

        # All pairs shortest paths between protein-heavy atoms
        csr_graph = csr_matrix(adj_matrix)  # Store data in compressed sparse matrix form
        apsp_matrix = floyd_warshall(csgraph=csr_graph, directed=False)  # Floyd-Warshall

        return apsp_matrix, all_to_heavy

    def plot_mean_contactmatrix(self, mean_contactmatrix):
        """
        Plots mean contact matrix.

        Args:
            mean_contactmatrix (np.array): Mean contactmatrix.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(mean_contactmatrix.T, origin="lower", cmap="hot")
        fig.colorbar(im)
        ax.set_xlabel('Atom $i$')
        ax.set_ylabel('Atom $j$')
        fig.set_dpi(self.dpi)
        self.save_figure(fig, suffix="mean_contactmatrix")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_total_fraction_contacts(self, ts_contacts, refcontacts):
        """
        Plots timeseries number of contacts and fraction of contacts.

        Args:
            ts_contacts (timeseries.TimeSeries): Timeseries of total contacts.
            refcontacts (float): Reference number of contacts
        """
        fig1 = ts_contacts.plot()
        ax = fig1.gca()
        ax.set_ylim([0, None])
        fig1.set_dpi(self.dpi)
        self.save_figure(fig1, suffix="contacts")

        ts_frac_contacts = copy.deepcopy(ts_contacts)
        ts_frac_contacts.data_array /= refcontacts
        fig2 = ts_frac_contacts.plot()
        ax = fig2.gca()
        ax.set_ylim([0, None])
        fig2.set_dpi(self.dpi)
        self.save_figure(fig2, suffix="frac_contacts")

        if self.show:
            plt.show()
        else:
            plt.close()

    def calc_plot_histogram_contacts(self, ts_contacts, bins):
        """
        Calculates and plots histogram of contacts, and saves histogram of
        contacts to file.

        Args:
            ts_contacts (timeseries.TimeSeries): Timeseries of total contacts.
            bins (int): Number of bins for histogram
        """
        hist, bin_edges = np.histogram(ts_contacts.data_array, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        histogram = np.zeros((2, len(hist)))
        histogram[0, :] = bin_centers
        histogram[1, :] = hist

        np.save(self.opref + "_hist.npy", histogram)

        fig, ax = plt.subplots()
        ax.plot(bin_centers, hist)
        ax.set_xlabel('Number of contacts')
        ax.set_ylabel('Frequency')
        fig.set_dpi(self.dpi)
        self.save_figure(fig, suffix="hist_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    """call"""
    def __call__(self):
        """Performs analysis."""

        # Calculate contacts along trajectory and mean contactmatrix
        ts_contacts, mean_contactmatrix = self.calc_trajcontacts(self.u, self.method, self.distcutoff, self.connthreshold,
                                                                 self.obsstart, self.obsend, self.skip)

        # Save data
        self.save_TimeSeries(ts_contacts, self.opref + "_contacts.pkl")
        np.save(self.opref + "_mean_contactmatrix.npy", mean_contactmatrix)

        # Calculate mean number of contacts along trajectory
        mean_contacts = ts_contacts[self.obsstart:self.obsend].mean()
        # If no reference is set, use this as the reference value for fraction of contacts
        if self.refcontacts is None:
            self.refcontacts = mean_contacts

        """Plots"""
        self.plot_mean_contactmatrix(mean_contactmatrix)
        self.plot_total_fraction_contacts(ts_contacts, self.refcontacts)
        self.calc_plot_histogram_contacts(ts_contacts, self.bins)
