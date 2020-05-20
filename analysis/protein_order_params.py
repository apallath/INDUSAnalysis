"""Calculate time series
- Radius of gyration
- RMSD
given GROMACS trajectory of a protein

Outputs
- Order parameter
- Order parameter (sliding window) moving average

Units:
- length: A
- time: ps

@Author: Akash Pallath

FEATURE:    Cythonize code
FEATURE:    Cumulative moving average for Rg and RMSD
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
        self.parser.add_argument("-reftrajf", help="Reference trajectory file (.xtc) for RMSD")
        self.parser.add_argument("-reftstep", help="Timestep to extract reference coordinates from reference trajectory file for RMSD")
        self.parser.add_argument("-select", help="Atoms/groups to track order parameters for (MDA selection string)")
        self.parser.add_argument("-align", help="Atoms/groups for aligning trajectories across timesteps (MDA selection string)")

    def read_args(self):
        super().read_args()
        self.structf = self.args.structf
        self.trajf = self.args.trajf
        self.reftrajf = self.args.reftrajf
        self.reftstep = self.args.reftstep
        self.select = self.args.select
        if self.select is None:
            self.select = 'protein'
        self.align = self.args.align
        if self.align is None:
            self.align= 'backbone'

        # Prepare system from args
        self.u = mda.Universe(self.structf,self.trajf)

        if self.reftrajf is not None:
            self.refu = mda.Universe(self.structf,self.reftrajf)
        else:
            self.refu = mda.Universe(self.structf,self.trajf)

        if self.reftstep is not None:
            self.reftstep = int(self.reftstep)
        else:
            self.reftstep = 0

    """Radius of gyration
    Tests in tests/test_orderparams.py"""
    def calc_Rg_worker(self,coords,masses):
        assert(coords.shape[1] == 3)

        com = np.average(coords, weights=masses, axis=0)
        sq_distances = np.sum((coords - com)**2, axis = 1)
        Rg = np.sqrt(np.average(sq_distances, weights=masses))
        return Rg

    def calc_Rg(self,u,selection):
        Rg = []
        sel = u.select_atoms(selection)
        for ts in u.trajectory:
            Rgval = self.calc_Rg_worker(sel.positions, sel.masses)
            Rg.append((u.trajectory.time, Rgval))
        Rg = np.array(Rg)
        return Rg

    """RMSD
    Tests in tests/test_orderparams.py"""
    def calc_RMSD_worker(self,initcoords,coords,aligninitcoords,aligncoords):
        assert(coords.shape[1] == 3 and initcoords.shape[1] == 3 and \
            aligncoords.shape[1] == 3 and aligninitcoords.shape[1] == 3)
        assert(coords.shape == initcoords.shape and \
            aligncoords.shape == aligninitcoords.shape)

        #calculate centers of geometry of alignment coordinates
        aligninitcog = np.mean(aligninitcoords, axis = 0)
        aligncog = np.mean(aligncoords, axis = 0)
        #center both alignment coordinates and coordinates
        aligninitcoords = aligninitcoords - aligninitcog
        initcoords = initcoords - aligninitcog
        aligncoods = aligncoords - aligncog
        coords = coords - aligncog

        #get rotation matrix by minimizing RMSD between centered alignment coordinates
        R, min_rms = mda.analysis.align.rotation_matrix(aligncoords, aligninitcoords)

        #rotate coords
        coords = np.dot(coords, R.T)

        #calculate RMSD
        sq_distances = np.sum((coords - initcoords)**2, axis = 1)
        RMSD = np.sqrt(np.mean(sq_distances))
        return RMSD

    def calc_RMSD(self,u,refu,reftstep,selection,alignment):
        sel = u.select_atoms(selection)
        refsel = refu.select_atoms(selection)
        align = u.select_atoms(alignment)
        refalign = refu.select_atoms(alignment)

        refu.trajectory[reftstep]
        initpos = refsel.positions.copy()
        aligninitpos = refalign.positions.copy()

        RMSD = []
        for ts in u.trajectory:
            RMSDval = self.calc_RMSD_worker(initpos, sel.positions, aligninitpos, align.positions)
            RMSD.append((u.trajectory.time, RMSDval))
        RMSD = np.array(RMSD)
        return RMSD

    """call"""
    def __call__(self):
        """Radius of gyration plots"""
        #radius of gyration
        sel_rg = self.calc_Rg(self.u, self.select)
        #plot
        fig, ax = plt.subplots()
        rg = sel_rg[:,1]
        t = sel_rg[:,0]
        ax.plot(t,rg)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig,suffix="Rg")
        self.save_timeseries(sel_rg[:,0],sel_rg[:,1],label="Rg")
        if self.show:
            plt.show()

        #radius of gyration moving average
        rg_ma = self.moving_average(t, rg, self.window)
        fig, ax = plt.subplots()
        ax.plot(t[len(t) - len(rg_ma):], rg_ma)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig,suffix="ma_Rg")
        if self.show:
            plt.show()

        if self.apref is not None:
            trgp = np.load(self.apref + "_Rg.npy")
            tp = trgp[0,:]
            rgp = trgp[1,:]
            tn = tp[-1]+t

            #plot time series data
            fig, ax = plt.subplots()
            ax.plot(tp,rgp,label=self.aprevlegend)
            ax.plot(tn,rg,label=self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(r"Radius of gyration ($\AA$)")
            ax.legend()
            self.save_figure(fig,suffix="app_Rg")
            if self.show:
                plt.show()

            #plot time series data moving average
            ttot = np.hstack([tp,tn])
            rgtot = np.hstack([rgp,rg])

            rgtot_ma = self.moving_average(ttot, rgtot,self.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(rgtot_ma):], rgtot_ma)
            #separator line
            ax.axvline(x=tp[-1])

            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(r"Radius of gyration ($\AA$)")
            self.save_figure(fig,suffix="app_ma_Rg")
            if self.show:
                plt.show()

        """RMSD plots"""
        #RMSD from initial structure
        sel_RMSD = self.calc_RMSD(self.u, self.refu, self.reftstep, self.select, self.align)
        #plot
        fig, ax = plt.subplots()
        rmsd = sel_RMSD[:,1]
        t = sel_RMSD[:,0]
        ax.plot(t,rmsd)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig,suffix="RMSD_"+self.align+"_"+self.select)
        self.save_timeseries(sel_RMSD[:,0],sel_RMSD[:,1],label="RMSD_"+self.align+"_"+self.select)
        if self.show:
            plt.show()

        #RMSD moving average
        rmsd_ma = self.moving_average(t, rmsd, self.window)
        fig, ax = plt.subplots()
        ax.plot(t[len(t) - len(rmsd_ma):], rmsd_ma)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig,suffix="ma_RMSD_"+self.align+"_"+self.select)
        if self.show:
            plt.show()

        if self.apref is not None:
            trmsdp = np.load(self.apref+"_RMSD_"+self.align+"_"+self.select+".npy")
            tp = trmsdp[0,:]
            rmsdp = trmsdp[1,:]
            tn = tp[-1]+t

            #plot time series data
            fig, ax = plt.subplots()
            ax.plot(tp,rmsdp,label=self.aprevlegend)
            ax.plot(tn,rmsd,label=self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(r"RMSD ($\AA$)")
            ax.legend()
            self.save_figure(fig,suffix="app_RMSD_"+self.align+"_"+self.select)
            if self.show:
                plt.show()

            #plot time series data moving average
            ttot = np.hstack([tp,tn])
            rmsdtot = np.hstack([rmsdp,rmsd])

            rmsdtot_ma = self.moving_average(ttot, rmsdtot,self.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(rmsdtot_ma):], rmsdtot_ma)
            #separator line
            ax.axvline(x=tp[-1])

            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(r"RMSD, moving average ($\AA$)")
            self.save_figure(fig,suffix="app_ma_RMSD_"+self.align+"_"+self.select)
            if self.show:
                plt.show()

warnings = "Proceed with caution: this script requires PBC-corrected protein structures!"

if __name__=="__main__":
    prot = OrderParams()
    prot.read_args()
    startup_string = "#### Order Parameter Analysis ####\n" + warnings + "\n"
    print(startup_string)
    prot()
