"""Calculate time series
- Radius of gyration
- RMSD
given GROMACS trajectory of a protein

@Author: Akash Pallath
"""
from analysis.timeseries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.analysis.rms as mda_rms

class OrderParams(timeseries.TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Structure file (.gro)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
        self.parser.add_argument("-select", help="atoms/groups to track order parameters for (MDA selection string)")
        self.parser.add_argument("-align", help="atoms/groups to align to reference frame (MDA selection string)")

    #testable
    def align_traj(self, u):
        #fit to initial frame
        prealigner = align.AlignTraj(u, select=self.align)
        prealigner.run()

    #testable
    def calc_Rg(self,u,selection):
        Rg = []
        sel = u.select_atoms(selection)
        for ts in u.trajectory:
            Rg.append((u.trajectory.time, sel.radius_of_gyration()))
        Rg = np.array(Rg)
        return Rg

    #testable
    def calc_RMSD(self,u,align,selection):
        """REQUIRES REVIEW"""

        """
        RMSD = []
        #select atoms
        sel = self.u.select_atoms(self.selection)
        #initial positions
        initpos = sel.positions.copy()
        for ts in self.u.trajectory:
            RMSD.append((self.u.trajectory.time, mda_rms.rmsd(initpos,sel.positions.copy(),superposition=True)))
        RMSD = np.array(RMSD)
        """
        R = mda_rms.RMSD(u, select=align, groupselections=[selection])
        R.run()
        #print(R.rmsd)
        RMSD = np.transpose(np.vstack([R.rmsd[:,1], R.rmsd[:,3]]))
        #print(RMSD)
        return RMSD

    def __call__(self):
        #definition in TimeSeries base class
        self.read_args()

        self.selection = self.args.select
        if self.selection is None:
            self.selection = 'protein'
        self.align = self.args.align
        if self.align is None:
            self.align = 'backbone'

        self.u = mda.Universe(self.args.structf,self.args.trajf)

        #radius of gyration
        sel_rg = self.calc_Rg(self.u, self.selection)
        #plot
        fig, ax = plt.subplots()
        ax.plot(sel_rg[:,0],sel_rg[:,1]);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Radius of gyration (nm)")
        self.save_figure(fig,suffix="Rg")

        #RMSD from initial structure
        sel_RMSD = self.calc_RMSD(self.u, self.align, self.selection)
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
