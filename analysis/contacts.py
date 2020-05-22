"""
Plot number of contacts, fraction of native contacts, and contact histograms
with time

Units:
- length: A
- time: ps

@Author: Akash Pallath

FEATURE:    Parallelize code
FEATURE:    Cythonize code
"""
from analysis.timeseries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.lib.distances #for fast distance matrix calculation

from meta_analysis.profiling import timefunc #for function run-time profiling

class Contacts(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Structure file (.gro)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
        self.parser.add_argument("-distcutoff", help="Distance cutoff for contacts, in A (default = 4.5 A)")
        self.parser.add_argument("-refcontacts", help="Reference number of contacts for fraction (default = mean)")
        self.parser.add_argument("-skip", help="Number of frames to skip between analyses")
        self.parser.add_argument("-bins", help="Number of bins for histogram")
        self.parser.add_argument("--verbose", action='store_true', help="Output progress of contacts calculation")

    def read_args(self):
        super().read_args()
        self.structf = self.args.structf
        self.trajf = self.args.trajf

        self.refcontacts = self.args.refcontacts
        if self.refcontacts is not None:
            self.refcontacts = float(self.refcontacts)

        self.skip = self.args.skip
        if self.skip is not None:
            self.skip = int(self.skip)

        self.bins = self.args.bins
        if self.bins is not None:
            self.bins = int(self.bins)
        else:
            self.bins = 20

        self.verbose = self.args.verbose
        self.distcutoff = self.args.distcutoff
        if self.distcutoff is not None:
            self.distcutoff = float(self.distcutoff)
        else:
            self.distcutoff = 4.5

        # Prepare system from args
        self.u = mda.Universe(self.structf, self.trajf)
        self.refu = mda.Universe(self.structf)
        if self.skip is None:
            self.skip = 1

    """
    Main analysis worker/helper to analyse contacts along trajectory
    """
    @timefunc
    def calc_trajcontacts(self, cutoff):
        side_heavy_sel = "protein and not(name N or name CA or name C or name O or name OC1 or name OC2 or type H)"

        protein = self.u.select_atoms("protein")

        nres = len(protein.residues)

        step = 0
        contacts = []
        l = len(protein.residues.atoms)
        contactmatrix = np.zeros((l,l))

        for ts in self.u.trajectory[0::self.skip]:
            box = ts.dimensions
            ncontacts = 0

            for i in range(nres):
                heavy_side_i = protein.residues[i].atoms.select_atoms(side_heavy_sel)
                heavy_side_j = protein.residues[i+4:].atoms.select_atoms(side_heavy_sel)
                dmatrix = mda.lib.distances.distance_array(heavy_side_i.positions, heavy_side_j.positions, box)
                local_contact_matrix = np.array(dmatrix < cutoff)
                ijc = np.count_nonzero(local_contact_matrix)
                ncontacts += ijc
                iidx = heavy_side_i.indices
                jidx = heavy_side_j.indices

                for i in range(dmatrix.shape[0]):
                    for j in range(dmatrix.shape[1]):
                        contactmatrix[iidx[i], jidx[j]] += local_contact_matrix[i,j]

            # Print progress
            if self.verbose:
                print("Step = {}, time = {} ps, contacts = {}".format(step*self.skip + 1,ts.time,ncontacts))

            contacts.append([ts.time, ncontacts])
            step += 1

        for i in range(contactmatrix.shape[0]):
            for j in range(contactmatrix.shape[1]):
                contactmatrix[i,j] = max(contactmatrix[i,j], contactmatrix[j,i])

        self.contacts = np.array(contacts)
        self.contactmatrix = contactmatrix

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

        if self.apref is not None:
            tcp = np.load(self.apref + "_contacts.npy")
            tp = tcp[0,:]
            contactsp = tcp[1,:]
            tn = tp[-1]+t

            #plot contacts time series
            fig, ax = plt.subplots()
            ax.plot(tp,contactsp,label=self.aprevlegend)
            ax.plot(tn,contacts,label=self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of contacts")
            ax.legend()
            self.save_figure(fig,suffix="app_contacts")
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
        self.save_figure(fig,suffix="frac_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

        if self.apref is not None:
            tcp = np.load(self.apref + "_contacts.npy")
            tp = tcp[0,:]
            contactsp = tcp[1,:]
            tn = tp[-1]+t

            #plot fraction of contacts time series
            fig, ax = plt.subplots()
            ax.plot(tp,contactsp/self.refcontacts,label=self.aprevlegend)
            ax.plot(tn,contacts/self.refcontacts,label=self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Fraction of contacts")
            ax.set_ylim(bottom = 0)
            ax.axhline(y=1, color='r', linestyle='--')
            ax.legend()
            self.save_figure(fig,suffix="app_frac_contacts")
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
        self.save_figure(fig,suffix="hist_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot contact matrix
    """
    def plot_matrix_contacts(self, contactmatrix):
        #save contact matrix
        pref = self.opref
        if pref == None:
            pref = "data"
        np.save(pref+"_contactmatrix", contactmatrix)

        fig, ax = plt.subplots()
        im = ax.imshow(contactmatrix)
        fig.colorbar(im)
        ax.set_xlabel('Atom $i$')
        ax.set_ylabel('Atom $j$')
        self.save_figure(fig,suffix="hist_contacts")
        if self.show:
            plt.show()
        else:
            plt.close()

    """call"""
    def __call__(self):
        """Get contacts along trajectory"""
        if self.replot:
            replotdata = np.load(self.replotpref + "_contacts.npy")
            self.contacts = np.transpose(replotdata)
        else:
            # Calculate contacts along trajectory
            self.calc_trajcontacts(self.distcutoff)

        t = self.contacts[:,0]
        contacts = self.contacts[:,1]

        """If no reference is set, use mean number of contacts along trajectory
           as reference (sensible choice for native state run)"""
        mean = self.ts_mean(t, contacts, self.obsstart, self.obsend)

        if self.refcontacts is None:
            self.refcontacts = mean

        """Log data"""
        self.save_timeseries(t, contacts, label="contacts")
        self.save_timeseries(t, contacts/self.refcontacts, label="frac_contacts")

        """Plots"""
        self.plot_contacts(t, contacts)
        self.plot_fraction_contacts(t, contacts, self.refcontacts)
        self.plot_histogram_contacts(contacts, bins=self.bins)
        self.plot_matrix_contacts(self.contactmatrix)

@timefunc
def main():
    warnings = "Proceed with caution: this script requires PBC-corrected protein structures!\n"
    contacts = Contacts()
    contacts.parse_args()
    contacts.read_args()
    startup_string = "#### Contacts ####\n" + warnings
    print(startup_string)
    contacts()
    plt.close('all')

if __name__=="__main__":
    main()
