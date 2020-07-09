"""
Defines class for analysing contacts along GROMACS simulation trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
import MDAnalysis.analysis.align

from itertools import combinations
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
        self.req_file_args.add_argument("structf", help="Topology or structure file (.tpr, .gro; .gro required for atomic-sh)")
        self.req_file_args.add_argument("trajf", help="Compressed trajectory file (.xtc)")

        self.calc_args.add_argument("-method", help="Method for calculating contacts (atomic-sh, 3res-sh; default=atomic-sh)")
        self.calc_args.add_argument("-distcutoff", help="Distance cutoff for contacts, in A")
        self.calc_args.add_argument("-connthreshold", help="Connectivity threshold for contacts (definition varies by method)")
        self.calc_args.add_argument("-skip", help="Number of frames to skip between analyses (default = 1)")
        self.calc_args.add_argument("-bins", help="Number of bins for histogram (default = 20)")
        self.calc_args.add_argument("-refcontacts", help="Reference number of contacts for fraction (default = mean)")

        self.out_args.add_argument("--genpdb", action="store_true", help="Write contacts density per atom to pdb file")

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
            self.method = "atomic-sh"

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

        self.genpdb = self.args.genpdb
        self.verbose = self.args.verbose

    def calc_trajcontacts(self, u, method, distcutoff, connthreshold, skip):
        """
        Calculates contacts along a trajectory.

        Args:
            u (MDAnalysis.Universe): Trajectory.
            method (str): Calculation method to use to compute contacts.
            distcutoff (float): Distance cutoff (in A).
            connthreshold (int): Connectivity threshold.
            skip (int): Resampling interval.

        Returns:
            TimeSeries object of contact matrices.

        Raises:
            ValueError if calculation method is not recognized.
        """
        if method == "atomic-sh":
            return self.calc_trajcontacts_atomic_sh(u, distcutoff, connthreshold, skip)
        else:
            raise ValueError("Method not recognized")

    @profiling.timefunc
    def calc_trajcontacts_atomic_sh(self, u, distcutoff, connthreshold, skip):
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
        not_side_heavy_sel = "protein and (name N or name CA or name C or name O or name OC1 or name OC2 or name H*)"

        # Calculate distances between all atoms using fast MDAnalysis function
        protein = u.select_atoms("protein")
        natoms = len(protein.atoms)

        utraj = u.trajectory[::skip]

        dmatrices = np.zeros((len(utraj), natoms, natoms))

        if self.verbose:
            pbar = tqdm(desc="Calculating distances", total=len(utraj))

        for tidx, ts in enumerate(utraj):
            dmatrix = mda.lib.distances.distance_array(protein.positions, protein.positions)
            dmatrices[tidx, :, :] = dmatrix
            if self.verbose:
                pbar.update(1)

        # Remove atoms that are not side chain heavy atoms
        for i in self.u.select_atoms(not_side_heavy_sel).indices:
            dmatrices[:, i, :] = np.Inf
            dmatrices[:, :, i] = np.Inf

        """Impose connectivity threshold"""
        if connthreshold < 0:
            raise ValueError("Connectivity threshold must be an integer value 0 or greater.")

        # Calculate connectivity graph using Floyd-Warshall algorithm
        apsp, idx_map = self.protein_heavy_APSP(u)

        # Exclude i-j interactions below threshold from distance matrix
        for i in range(apsp.shape[0]):
            for j in range(apsp.shape[1]):
                if i == j and apsp[i, j] > 0:
                    raise ValueError("Distance matrix is inconsistent: shortest path between same atom should be 0.")
                if apsp[i, j] <= connthreshold:
                    dmatrices[:, idx_map[i], idx_map[j]] = np.Inf

        """Impose distance cutoff"""
        contactmatrices = np.array(dmatrices < self.distcutoff)

        times = []
        for ts in utraj:
            times.append(ts.time)
        times = np.array(times)

        return timeseries.TimeSeries(times, contactmatrices, labels=['Contact duration', 'Atom i', 'Atom j'])

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
                    
                index_map (dict): Dictionary mapping heavy atom i to its atom
                    index in the Universe.
            }
        """
        # Connectivity graph
        protein_heavy = u.select_atoms("protein and not name H*")
        nheavy = len(protein_heavy)

        heavy_indices = protein_heavy.atoms.indices
        heavy_idx_map = {}
        idx_heavy_map = {}
        for heavyidx, idx in enumerate(protein_heavy.atoms.indices):
            heavy_idx_map[idx] = heavyidx
            idx_heavy_map[heavyidx] = idx

        adj_matrix = np.zeros((nheavy, nheavy))

        for bond in protein_heavy.bonds:
            ati = bond.indices[0]
            atj = bond.indices[1]
            if ati in heavy_indices and atj in heavy_indices:
                heavyi = heavy_idx_map[ati]
                heavyj = heavy_idx_map[atj]
                adj_matrix[heavyi, heavyj] = 1
                adj_matrix[heavyj, heavyi] = 1

        # All pairs shortest paths between protein-heavy atoms
        csr_graph = csr_matrix(adj_matrix)  # Store data in compressed sparse matrix form
        dist_matrix = floyd_warshall(csgraph=csr_graph, directed=False)  # Floyd-Warshall

        return dist_matrix, idx_heavy_map

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
        fig.set_dpi(300)
        self.save_figure(fig, suffix="mean_contactmatrix")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_contacts_per_atom(self, ts_contacts_per_atom):
        """
        Plots timeseries contacts per atom.
        """
        fig = ts_contacts_per_atom.plot_2d_heatmap(cmap='hot')
        fig.set_dpi(300)
        self.save_figure(fig, suffix="contacts_per_atom")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_total_fraction_contacts(self, ts_contacts, refcontacts):
        """
        Plots timeseries number of contacts and fraction of contacts.
        """
        fig1 = ts_contacts.plot()
        fig1.set_dpi(300)
        self.save_figure(fig1, suffix="contacts")

        ts_contacts.data_array /= refcontacts
        fig2 = ts_contacts.plot()
        fig2.set_dpi(300)
        self.save_figure(fig2, suffix="frac_contacts")

        if self.show:
            plt.show()
        else:
            plt.close()

    def calc_plot_histogram_contacts(self, ts_contacts, bins):
        """
        Calculates and plots histogram of contacts, and saves histogram of
        contacts to file.
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
        fig.set_dpi(300)
        self.save_figure(fig, suffix="hist_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    def write_contacts_per_atom_pdb(self, u, skip, ts_contacts_per_atom):
        """
        Writes per-atom-contacts to PDB file.

        Args:
            u (mda.Universe): Universe containing solvated protein.
            skip (int): Trajectory resampling interval.
            ts_contacts_per_atom (TimeSeries): Per-atom contacts timeseries data.

        Raises:
            ValueError if the time for the same index in u.trajectory[::skip]
            and ts_contacts_per_atom does not match.
        """
        protein = u.select_atoms("protein")
        u.add_TopologyAttr('tempfactors')
        pdbtrj = self.opref + "_contacts.pdb"

        utraj = u.trajectory[0::skip]

        if self.verbose:
            pbar = tqdm(desc="Writing PDB", total=len(utraj))

        with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
            for tidx, ts in enumerate(utraj):
                if np.isclose(ts.time, ts_contacts_per_atom.time_array[tidx]):
                    protein.atoms.tempfactors = ts_contacts_per_atom.data_array[tidx, :]
                    PDB.write(self.u.atoms)
                    if self.verbose:
                        pbar.update(1)

                else:
                    raise ValueError("Trajectory and TimeSeries times do not match at same index.")

    """call"""
    def __call__(self):
        """Performs analysis."""

        """Raw data"""
        if self.replot:
            ts_contactmatrices = self.load_TimeSeries(self.opref + "_contactmatrices")
            ts_contacts_per_atom = self.load_TimeSeries(self.opref + "_contacts_per_atom")
            ts_contacts = self.load_TimeSeries(self.opref + "_contacts")
        else:
            # Calculate contacts along trajectory
            ts_contactmatrices = self.calc_trajcontacts(self.u, self.method, self.distcutoff,
                                                        self.connthreshold, self.skip)
            # Compute reductions
            ts_contacts_per_atom = ts_contactmatrices.mean(axis=2)
            ts_contacts = ts_contacts_per_atom.mean(axis=1)

        self.save_TimeSeries(ts_contactmatrices, self.opref + "_contactmatrices")
        self.save_TimeSeries(ts_contacts_per_atom, self.opref + "_contacts_per_atom")
        self.save_TimeSeries(ts_contacts, self.opref + "_contacts")

        # Calculate mean contactmatrix
        mean_contactmatrix = ts_contactmatrices[self.obsstart:self.obsend].mean(axis=0)

        # Calculate mean number of contacts along trajectory
        mean_contacts = ts_contacts[self.obsstart:self.obsend].mean()
        # If no reference is set, use this as the reference value for fraction of contacts
        if self.refcontacts is None:
            self.refcontacts = mean_contacts

        """Plots"""
        self.plot_mean_contactmatrix(mean_contactmatrix)
        self.plot_contacts_per_atom(ts_contacts_per_atom)
        self.plot_total_fraction_contacts(ts_contacts, self.refcontacts)

        self.calc_plot_histogram_contacts(ts_contacts, self.bins)

        """Store per-atom contacts in PDB"""
        if self.genpdb:
            self.write_contacts_per_atom_pdb(self.u, self.skip, ts_contacts_per_atom)
