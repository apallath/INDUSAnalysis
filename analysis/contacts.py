"""Plot number of contacts and fraction of native contacts with time

Units:
- length: A
- time: ps

@Author: Akash Pallath
"""
from analysis.timeseries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.lib.distances #for fast distance matrix calculation

class Contacts(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Structure file (.gro)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
        self.parser.add_argument("-distcutoff", help="Distance cutoff for contacts, in A (default = 4.5 A)")
        self.parser.add_argument("-refcontacts", help="Reference number of contacts for fraction (default = contacts for structf)")
        self.parser.add_argument("-skip", help="Number of frames to skip between analyses")
        self.parser.add_argument("--verbose", action='store_true', help="Output progress of contacts calculation")

    # Read arguments into member variables
    def read_args(self):
        super().read_args()
        self.structf = self.args.structf
        self.trajf = self.args.trajf
        self.refcontacts = self.args.refcontacts
        self.skip = self.args.skip
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

    # Reference contacts calculation
    def calc_refcontacts(self):
        side_heavy_sel = "protein and not(name N or name CA or name C or name O or name OC1 or name OC2 or type H)"

        refprotein = self.refu.select_atoms("protein")

        nres = len(refprotein.residues)
        box = self.refu.dimensions

        refcontacts = 0

        for i in range(nres):
            heavy_side_i = refprotein.residues[i].atoms.select_atoms(side_heavy_sel)
            heavy_side_j = refprotein.residues[i+4:].atoms.select_atoms(side_heavy_sel)
            da = mda.lib.distances.distance_array(heavy_side_i.positions, heavy_side_j.positions, box)
            refcontacts += np.count_nonzero(da < self.distcutoff)

        self.refcontants = refcontacts

    # Contacts analysis along trajectory
    def calc_trajcontacts(self):
        side_heavy_sel = "protein and not(name N or name CA or name C or name O or name OC1 or name OC2 or type H)"

        protein = self.u.select_atoms("protein")

        nres = len(protein.residues)

        step = 0
        contacts = []

        for ts in self.u.trajectory[0::self.skip]:
            box = ts.dimensions
            ncontacts = 0

            for i in range(nres):
                heavy_side_i = protein.residues[i].atoms.select_atoms(side_heavy_sel)
                heavy_side_j = protein.residues[i+4:].atoms.select_atoms(side_heavy_sel)
                da = mda.lib.distances.distance_array(heavy_side_i.positions, heavy_side_j.positions, box)
                ncontacts += np.count_nonzero(da < self.distcutoff)
            # Print progress
            if self.verbose:
                print("Step = {}, time = {} ps, contacts = {}".format(step*self.skip + 1,ts.time,ncontacts))

            contacts.append([ts.time, ncontacts])
            step += 1

        self.contacts = np.array(contacts)

    """call"""
    def __call__(self):
        """Contacts along trajectory plot"""
        # Calculate reference number of contacts if unavailable
        if self.refcontacts is None:
            self.calc_refcontacts()

        if self.replot:
            replotdata = np.load(self.apref + "_contacts.npy")
            self.contacts = np.transpose(replotdata)
        else:
            # Calculate contacts along trajectory
            self.calc_trajcontacts()

        # Plot number of contacts
        fig, ax = plt.subplots()
        contacts = self.contacts[:,1]
        t = self.contacts[:,0]
        ax.plot(t,contacts);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of contacts")
        self.save_figure(fig,suffix="contacts")
        self.save_timeseries(self.contacts[:,0], self.contacts[:,1], label="contacts")
        if self.show:
            plt.show()

        # Plot fraction of contacts
        fig, ax = plt.subplots()
        contacts = self.contacts[:,1]
        t = self.contacts[:,0]
        ax.plot(t, contacts/self.refcontacts);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Fraction of contacts")
        self.save_figure(fig,suffix="frac_contacts")
        self.save_timeseries(self.contacts[:,0], self.contacts[:,1], label="contacts")
        if self.show:
            plt.show()

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

            #plot fraction of contacts time series
            fig, ax = plt.subplots()
            ax.plot(tp,contactsp/self.refcontacts,label=self.aprevlegend)
            ax.plot(tn,contacts/self.refcontacts,label=self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Fractionc of contacts")
            ax.legend()
            self.save_figure(fig,suffix="app_frac_contacts")
            if self.show:
                plt.show()

warnings = "Proceed with caution: this script requires PBC-corrected protein structures!\n"

if __name__=="__main__":
    contacts = Contacts()
    contacts.read_args()
    startup_string = "#### Contacts ####\n" + warnings
    print(startup_string)
    contacts()
