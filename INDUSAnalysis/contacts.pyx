"""
Time series contacts analysis (# of contacts and fraction of native contacts)

Supports PDB generation from saved data using the replot option with genpdb

Units:
- length: A
- time: ps

@Author: Akash Pallath
"""

from INDUSAnalysis.timeseries import TimeSeries
from INDUSAnalysis.lib.profiling import timefunc

import numpy as np

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.lib.distances  # for fast distance matrix calculation
from tqdm import tqdm  # for progress bars
from itertools import combinations

"""Cython"""
cimport numpy as np


class Contacts(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Topology or structure file (.tpr, .gro; .gro required for atomic-sh)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")

        # Calculation options
        self.parser.add_argument("-method", help="Method for calculating contacts (atomic-sh, 3res-sh; default=atomic-sh)")
        self.parser.add_argument("-distcutoff", help="Distance cutoff for contacts, in A")
        self.parser.add_argument("-refcontacts", help="Reference number of contacts for fraction (default = mean)")
        self.parser.add_argument("-skip", help="Number of frames to skip between analyses (default = 1)")
        self.parser.add_argument("-bins", help="Number of bins for histogram (default = 20)")

        # Output control
        self.parser.add_argument("--genpdb", action="store_true", help="Write contacts density per atom to pdb file")
        self.parser.add_argument("--verbose", action='store_true', help="Output progress of contacts calculation")

    def read_args(self):
        super().read_args()
        self.structf = self.args.structf
        self.trajf = self.args.trajf

        # Calculation options
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
            elif self.method == "3res-sh":
                self.distcutoff = 4.5
            else:
                pass

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

        # Output control
        self.genpdb = self.args.genpdb
        self.verbose = self.args.verbose

        # Prepare system from args
        self.u = mda.Universe(self.structf, self.trajf)

    """
    BEGIN
    Main analysis worker/helpers to analyse contacts along trajectory
    """

    def calc_trajcontacts(self):
        if self.method == "atomic-sh":
            return self.calc_trajcontacts_atomic_sh()
        elif self.method == "3res-sh":
            return self.calc_trajcontacts_3res_sh()
        else:
            raise Exception("Method not recognized")
            return None

    """
    Method: atomic-sh
    Contacts between side-chain heavy atoms
    """
    @timefunc
    def calc_trajcontacts_atomic_sh(self):
        not_side_heavy_sel = "protein and (name N or name CA or name C or name O or name OC1 or name OC2 or name H*)"

        # Calculate distances between all atoms using fast MDA function
        protein = self.u.select_atoms("protein")
        natoms = len(protein.atoms)

        utraj = self.u.trajectory[::self.skip]

        dmatrices = np.zeros((len(utraj), natoms, natoms))

        if self.verbose:
            pbar = tqdm(desc = "Calculating distances", total = len(utraj))

        for tidx, ts in enumerate(utraj):
            dmatrix = mda.lib.distances.distance_array(protein.positions, protein.positions)
            dmatrices[tidx,:,:] = dmatrix
            if self.verbose:
                pbar.update(1)

        if self.verbose:
            print("Processing distance matrices")

        # Process distance matrices
        for i in range(dmatrices.shape[1]):
            dmatrices[:,i,i] = np.Inf

        # Remove atoms that are not side chain heavy atoms
        for i in self.u.select_atoms(not_side_heavy_sel).indices:
            dmatrices[:,i,:] = np.Inf
            dmatrices[:,:,i] = np.Inf

        if self.verbose:
            print("Calculating contacts from distance matrices")

        self.contactmatrices = np.array(dmatrices < self.distcutoff)

        # Contacts per atom along trajectory
        if self.verbose:
            print("Calculating contacts per atom")
        self.contacts_per_atom = np.sum(self.contactmatrices, axis = 2)

        # Total number of contacts along trajectory
        contacts = np.sum(self.contacts_per_atom, axis=1)
        times = []
        for ts in utraj:
            times.append(ts.time)
        times = np.array(times)
        self.contacts = np.zeros((len(utraj), 2))
        self.contacts[:,0] = times
        self.contacts[:,1] = contacts

        # Normalized mean contactmatrix
        self.contactmatrix = np.mean(self.contactmatrices, axis=0)

    """
    Method: 3res-sh
    Contacts between side-chain heavy atoms belonging to residues that are at least 3 residues apart
    """
    @timefunc
    def calc_trajcontacts_3res_sh(self):
        side_heavy_sel = "protein and not(name N or name CA or name C or name O or name OC1 or name OC2 or name H*)"

        protein = self.u.select_atoms("protein")
        nres = len(protein.residues)
        natoms = len(protein.atoms)

        contactmatrices = []

        utraj = self.u.trajectory[::self.skip]

        if self.verbose:
            pbar = tqdm(desc = "Calculating contacts", total = len(utraj))

        for ts in utraj:
            box = ts.dimensions
            local_contactmatrix = np.zeros((natoms, natoms))

            for i in range(nres):
                heavy_side_i = protein.residues[i].atoms.select_atoms(side_heavy_sel)
                heavy_side_j = protein.residues[i+4:].atoms.select_atoms(side_heavy_sel)
                dmatrix = mda.lib.distances.distance_array(heavy_side_i.positions, heavy_side_j.positions, box)
                res_contactmatrix = np.array(dmatrix < self.distcutoff)
                iidx = heavy_side_i.indices
                jidx = heavy_side_j.indices

                for i in range(dmatrix.shape[0]):
                    for j in range(dmatrix.shape[1]):
                        local_contactmatrix[iidx[i], jidx[j]] += res_contactmatrix[i,j]

            #make matrix symmetric
            local_contactmatrix = np.maximum(local_contactmatrix[:,:], local_contactmatrix[:,:].transpose())
            contactmatrices.append(local_contactmatrix)

            # Print progress
            if self.verbose:
                pbar.update(1)

        # Contact matrices along trajectory
        self.contactmatrices = np.array(contactmatrices)

        # Contacts per atom along trajectory
        self.contacts_per_atom = np.sum(self.contactmatrices, axis = 2)

        # Total number of contacts along trajectory
        contacts = np.sum(self.contacts_per_atom, axis=1)
        times = []
        for ts in utraj:
            times.append(ts.time)
        times = np.array(times)
        self.contacts = np.zeros((len(utraj), 2))
        self.contacts[:,0] = times
        self.contacts[:,1] = contacts

        # Normalized mean contactmatrix
        self.contactmatrix = np.mean(self.contactmatrices, axis=0)

    """
    END
    Main analysis worker/helpers to analyse contacts along trajectory
    """

    """
    Save
    - Contact matrix along trajectory
    - Contacts per atom along trajectory
    - Averaged contact matrix
    """
    def save_rawcontactsdata(self):
        np.save(self.opref + "_contactmatrices", self.contactmatrices)
        np.save(self.opref + "_contacts_per_atom", self.contacts_per_atom)
        np.save(self.opref + "_contactmatrix", self.contactmatrix)

    """
    Save instantaneous contacts density data to pdb
    """
    def save_pdb(self):
        protein = self.u.select_atoms("protein")
        self.u.add_TopologyAttr('tempfactors')
        utraj = self.u.trajectory[0::self.skip]
        pdbtrj = self.opref + "_contacts.pdb"

        if self.verbose:
            pbar = tqdm(desc = "Writing PDB", total = len(utraj))

        with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=self.u.atoms.n_atoms) as PDB:
            for tidx, ts in enumerate(utraj):
                protein.atoms.tempfactors = self.contacts_per_atom[tidx,:]
                PDB.write(self.u.atoms)
                if self.verbose:
                    pbar.update(1)

    """
    Plot number of contacts
    """
    def plot_contacts(self, t, contacts):
        fig, ax = plt.subplots()
        ax.plot(t, contacts)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of contacts")
        self.save_figure(fig,suffix="contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot fraction of contacts
    """
    def plot_fraction_contacts(self, t, contacts, refcontacts):
        fig, ax = plt.subplots()
        ax.plot(t, contacts/self.refcontacts)
        ax.set_ylim(bottom = 0)
        ax.axhline(y=1.0, color='r', linestyle='--')
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Fraction of contacts")
        self.save_figure(fig, suffix="frac_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot histogram of contacts (as density)
    Notes:
    - Also stores histogram of contacts
    """
    def plot_histogram_contacts(self, contacts, bins):
        hist, bin_edges = np.histogram(contacts, bins=bins, density=True)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        #use timeseries function even though not timeseries
        self.save_timeseries(bin_centers, hist, label="hist_contacts")

        fig, ax = plt.subplots()
        ax.plot(bin_centers, hist)
        ax.set_xlabel('Number of contacts')
        ax.set_ylabel('Frequency')
        self.save_figure(fig, suffix="hist_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot contact matrix
    """
    def plot_matrix_contacts(self, contactmatrix):
        fig, ax = plt.subplots()
        im = ax.imshow(contactmatrix.T, origin="lower", cmap="hot")
        fig.colorbar(im)
        ax.set_xlabel('Atom $i$')
        ax.set_ylabel('Atom $j$')
        self.save_figure(fig, suffix="contactmatrix")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot contacts per atom
    """
    def plot_contacts_per_atom(self, times, contacts_per_atom):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(contacts_per_atom, origin="lower", cmap="hot", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Atom')
        ax.set_ylabel('Time (ps)')
        #SET TICKS
        ticks = ax.get_yticks().tolist()
        factor = times[-1]/ticks[-2]
        newlabels = [factor * item for item in ticks]
        ax.set_yticklabels(newlabels)
        self.save_figure(fig, suffix="contacts_per_atom")
        if self.show:
            plt.show()
        else:
            plt.close()

    """call"""
    def __call__(self):
        """Get contacts along trajectory"""
        if self.replot:
            replotcontactmatrices = np.load(self.replotpref + "_contactmatrices.npy")
            self.contactmatrices = replotcontactmatrices

            replotcontacts_per_atom = np.load(self.replotpref + "_contacts_per_atom.npy")
            self.contacts_per_atom = replotcontacts_per_atom

            replotcontactmatrix = np.load(self.replotpref + "_contactmatrix.npy")
            self.contactmatrix = replotcontactmatrix

            replotcontacts = np.load(self.replotpref + "_contacts.npy")
            self.contacts = np.transpose(replotcontacts)
        else:
            # Calculate contacts along trajectory
            self.calc_trajcontacts()

        t = self.contacts[:,0]
        contacts = self.contacts[:,1]

        """If no reference is set, use mean number of contacts along trajectory
           as reference (sensible choice for native state run)"""
        mean = self.ts_mean(t, contacts, self.obsstart, self.obsend)

        if self.refcontacts is None:
            self.refcontacts = mean

        """Log data"""
        # Matrices
        self.save_rawcontactsdata()

        # Timeseries
        self.save_timeseries(t, contacts, label="contacts")
        self.save_timeseries(t, contacts/self.refcontacts, label="frac_contacts")

        # PDB
        if self.genpdb:
            self.save_pdb()

        """Plots"""
        self.plot_contacts(t, contacts)
        self.plot_fraction_contacts(t, contacts, self.refcontacts)
        self.plot_histogram_contacts(contacts, bins=self.bins)
        self.plot_matrix_contacts(self.contactmatrix)
        self.plot_contacts_per_atom(t, self.contacts_per_atom)
