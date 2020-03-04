"""Calculate time series
- Radius of gyration
- RMSD
given GROMACS trajectory of a protein

Units:
- length: A
- time: ps

TODO: Clean up plotting sphagetti code?

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
        self.parser.add_argument("-reftrajf", help="Reference trajectory file (.xtc) for RMSD")
        self.parser.add_argument("-reftstep", help="Timestep to extract reference coordinates from reference trajectory file for RMSD")
        self.parser.add_argument("-select", help="Atoms/groups to track order parameters for (MDA selection string)")
        self.parser.add_argument("-align", help="Atoms/groups for aligning trajectories across timesteps (MDA selection string)")

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

    def __call__(self):
        #definition in TimeSeries base class
        self.read_args()

        self.u = mda.Universe(self.args.structf,self.args.trajf)
        self.refu = mda.Universe(self.args.structf,self.args.trajf)
        self.reftstep = 0
        #check if separate reference trajectory file was passed
        if self.args.reftrajf is not None:
            self.refu = mda.Universe(self.args.structf,self.args.reftrajf)
        #check if reference timestep was passed
        if self.args.reftstep is not None:
            self.reftstep = int(self.args.reftstep)

        self.selection = self.args.select
        self.align = self.args.align
        #defaults: align backbones, calculate RMSDs
        if self.selection is None:
            self.selection = 'protein'
        if self.align is None:
            self.align= 'backbone'

        """Radius of gyration"""
        #radius of gyration
        sel_rg = self.calc_Rg(self.u, self.selection)
        #plot
        fig, ax = plt.subplots()
        rg = sel_rg[:,1]
        t = sel_rg[:,0]
        ax.plot(t,rg);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig,suffix="Rg")
        self.save_timeseries(sel_rg[:,0],sel_rg[:,1],label="Rg")
        if self.args.show:
            plt.show()

        #radius of gyration moving average
        rg_ma = self.moving_average(rg, self.args.window)
        fig, ax = plt.subplots()
        ax.plot(t[len(t) - len(rg_ma):], rg_ma);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig,suffix="ma_Rg")
        if self.args.show:
            plt.show()

        if self.args.apref is not None:
            trgp = np.load(self.args.apref + "_Rg.npy")
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
            if self.args.show:
                plt.show()

            #plot time series data moving average
            ttot = np.hstack([tp,tn])
            rgtot = np.hstack([rgp,rg])

            rgtot_ma = self.moving_average(rgtot,self.args.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(rgtot_ma):], rgtot_ma)
            #separator line
            ax.axvline(x=tp[-1])

            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(r"Radius of gyration ($\AA$)")
            self.save_figure(fig,suffix="app_ma_Rg")
            if self.args.show:
                plt.show()

        """RMSD"""
        #RMSD from initial structure
        sel_RMSD = self.calc_RMSD(self.u, self.refu, self.reftstep, self.selection, self.align)
        #plot
        fig, ax = plt.subplots()
        rmsd = sel_RMSD[:,1]
        t = sel_RMSD[:,0]
        ax.plot(t,rmsd)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig,suffix="RMSD_"+self.align+"_"+self.selection)
        self.save_timeseries(sel_RMSD[:,0],sel_RMSD[:,1],label="RMSD_"+self.align+"_"+self.selection)
        if self.args.show:
            plt.show()

        #RMSD moving average
        rmsd_ma = self.moving_average(rmsd, self.args.window)
        fig, ax = plt.subplots()
        ax.plot(t[len(t) - len(rmsd_ma):], rmsd_ma);
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig,suffix="ma_RMSD_"+self.align+"_"+self.selection)
        if self.args.show:
            plt.show()

        if self.args.apref is not None:
            trmsdp = np.load(self.args.apref+"_RMSD_"+self.align+"_"+self.selection+".npy")
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
            self.save_figure(fig,suffix="app_RMSD_"+self.align+"_"+self.selection)
            if self.args.show:
                plt.show()

            #plot time series data moving average
            ttot = np.hstack([tp,tn])
            rmsdtot = np.hstack([rmsdp,rmsd])

            rmsdtot_ma = self.moving_average(rmsdtot,self.args.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(rmsdtot_ma):], rmsdtot_ma)
            #separator line
            ax.axvline(x=tp[-1])

            ax.set_xlabel("Time (ps)")
            ax.set_ylabel(r"RMSD, moving average ($\AA$)")
            self.save_figure(fig,suffix="app_ma_RMSD_"+self.align+"_"+self.selection)
            if self.args.show:
                plt.show()

warnings = "Proceed with caution: this script requires PBC-corrected protein structures!\n"

if __name__=="__main__":
    prot = OrderParams()
    startup_string = "#### Order Parameter Analysis ####\n" + warnings
    print(startup_string)
    prot()
