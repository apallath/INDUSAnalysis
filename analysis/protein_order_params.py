"""Calculate time series
- Radius of gyration
- RMSD
given GROMACS trajectory of a protein

@Author: Akash Pallath
"""
import timeseries

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.rms as mda_rms

class OrderParams(timeseries.TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Structure file (.gro)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
        self.parser.add_argument("-select", help="atoms/groups to track order parameters for (MDA selection string)")
        self.parser.add_argument("-align", help="atoms/groups to align to reference frame (MDA selection string)")

    def calc_Rg(self):
        Rg = []
        sel = self.u.select_atoms(self.selection)
        for ts in self.u.trajectory:
            Rg.append((self.u.trajectory.time, sel.radius_of_gyration()))
        Rg = np.array(Rg)
        return Rg

    def calc_RMSD(self):
        RMSD = []
        sel = self.u.select_atoms(self.selection)
        #initial positions
        initpos = sel.positions.copy()
        for ts in self.u.trajectory:
            RMSD.append((self.u.trajectory.time, mda_rms.rmsd(initpos,sel.positions.copy(),superposition=True)))
        RMSD = np.array(RMSD)
        return RMSD

    def __call__(self):
        #definition in TimeSeries base class
        self.read_args()
        self.selection = self.args.selection
        if self.selection == None:
            self.selection = 'protein'
        self.u = mda.Universe(self.args.structf,self.args.trajf)

        #radius of gyration
        sel_rg = self.calc_Rg()
        #plot
        fig, ax = plt.subplots()
        ax.plot(sel_rg[:,0],sel_rg[:,1]);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Radius of gyration (nm)")
        self.save_figure(fig,suffix="Rg")

        #RMSD from initial structure
        sel_RMSD = self.calc_RMSD()
        #plot
        fig, ax = plt.subplots()
        ax.plot(sel_RMSD[:,0],sel_RMSD[:,1])
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("RMSD (nm)")
        self.save_figure(fig,suffix="RMSD")
        if self.args.show:
            plt.show()

if __name__=="__main__":
    prot = OrderParams()
    prot()
