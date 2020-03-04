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
import MDAnalysis.analysis.align

class OrderParams(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Structure file (.gro)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
        self.parser.add_argument("-select", help="Atoms/groups to track order parameters for (MDA selection string)")

    """tests in tests/test_orderparams.py"""
    def calc_Rg_worker(self,coords,masses):
        com = np.mean(coords, weights=masses, axis=0)
        sq_distances = np.sum((coords - com)**2, axis = 1)
        Rg = np.sqrt(np.mean(sq_distances, weights=masses))
        return Rg

    def calc_Rg(self,u,selection):
        Rg = []
        sel = u.select_atoms(selection)
        for ts in u.trajectory:
            Rgval = self.calc_Rg_worker(sel.positions, sel.masses)
            Rg.append((u.trajectory.time, Rgval))
        Rg = np.array(Rg)
        return Rg

    """tests in tests/test_orderparams.py"""
    def calc_RMSD_worker(self,initcoords,coords):
        #align centers of geometry of coordinates
        initcoords = initcoords - np.mean(initcoords, axis = 0)
        coords = coords - np.mean(coords, axis = 0)
        #get rotation matrix by minimizing RMSD
        R, min_rms = mda.analysis.align.rotation_matrix(coords, initcoords)
        #rotate coords
        coords = np.dot(coords, R.T)
        sq_distances = np.sum((coords - initcoords)**2, axis = 1)
        RMSD = np.sqrt(np.mean(sq_distances))
        return RMSD

    def calc_RMSD(self,u,selection):
        RMSD = []
        sel = u.select_atoms(selection)
        initpos = sel.positions.copy()
        for ts in u.trajectory:
            RMSDval = self.calc_RMSD_worker(initpos, sel.positions)
            RMSD.append((u.trajectory.time, RMSDval))
        RMSD = np.array(RMSD)
        return RMSD

    def __call__(self):
        #definition in TimeSeries base class
        self.read_args()

        self.selection = self.args.select
        if self.selection is None:
            self.selection = 'protein'

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
        sel_RMSD = self.calc_RMSD(self.u, self.selection)
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
    print("Warning: script requires PBC-corrected protein structures!")
    prot()
