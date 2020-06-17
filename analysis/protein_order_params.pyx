"""
Calculate time series
- Radius of gyration
- RMSD
- Per-atom-deviation
given GROMACS trajectory of a protein

Units:
- length: A
- time: ps

@Author: Akash Pallath
"""

from analysis.timeseries import TimeSeries

import numpy as np

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.align
from tqdm import tqdm

from meta_analysis.profiling import timefunc #for function run-time profiling

"""Cython"""
cimport numpy as np

class OrderParams(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("structf", help="Structure file (.gro)")
        self.parser.add_argument("trajf", help="Compressed trajectory file (.xtc)")
        self.parser.add_argument("-reftrajf", help="Reference trajectory file (.xtc) for RMSD")
        self.parser.add_argument("-reftstep", help="Timestep to extract reference coordinates from reference trajectory file for RMSD")
        self.parser.add_argument("-select", help="Atoms/groups to track order parameters for (MDA selection string)")
        self.parser.add_argument("-align", help="Atoms/groups for aligning trajectories across timesteps (MDA selection string)")
        self.parser.add_argument("-skip", help="Number of frames to skip between analyses (default = 1)")

        #Output control
        self.parser.add_argument("--genpdb", action="store_true", help="Write atoms per probe volume data to pdb file")
        self.parser.add_argument("--verbose", help="Display progress", action="store_true")

    def read_args(self):
        #POPULATE AS REQUIRED
        self.selection_parser = {
            'protein': """protein""",
            'backbone': """name CA or name C or name N""",
            'side_chain': """protein and not (name N or name CA or name C or name O or name H
                or name H1 or name H2 or name H3 or name OC1 or name OC2)""",
            'heavy': "protein and not name H*",
            'side_chain_heavy': """protein and not(name N or name CA or name C or name O
                or name OC1 or name OC2 or name H*)"""
        }

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
            self.align = 'backbone'

        self.skip = self.args.skip
        if self.skip is not None:
            self.skip = int(self.skip)
        else:
            self.skip = 1

        # Prepare system from args
        self.u = mda.Universe(self.structf, self.trajf)

        if self.reftrajf is not None:
            self.refu = mda.Universe(self.structf, self.reftrajf)
        else:
            self.refu = mda.Universe(self.structf, self.trajf)

        if self.reftstep is not None:
            self.reftstep = int(self.reftstep)
        else:
            self.reftstep = 0

        self.genpdb = self.args.genpdb
        self.verbose = self.args.verbose

    """
    Radius of gyration worker and helper function

    Tests in tests/test_orderparams.py
    """
    def calc_Rg_worker(self, coords, masses):
        #quick test
        assert(coords.shape[1] == 3)

        com = np.average(coords, weights=masses, axis=0)
        sq_distances = np.sum((coords - com)**2, axis = 1)
        Rg = np.sqrt(np.average(sq_distances, weights=masses))
        return Rg

    def calc_Rg(self, u, selection):
        Rg = []
        sel = u.select_atoms(selection)
        for ts in u.trajectory[0::self.skip]:
            Rgval = self.calc_Rg_worker(sel.positions, sel.masses)
            Rg.append((u.trajectory.time, Rgval))
        Rg = np.array(Rg)
        return Rg

    """
    Plot radius of gyration
    """
    def plot_Rg(self, Rg_data):
        t = Rg_data[:,0]
        rg = Rg_data[:,1]

        fig, ax = plt.subplots()
        ax.plot(t,rg)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig, suffix="Rg")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot moving (sliding window) average of radius of gyration
    """
    def plot_ma_Rg(self, Rg_data):
        t = Rg_data[:,0]
        rg = Rg_data[:,1]

        rg_ma = self.moving_average(t, rg, self.window)
        fig, ax = plt.subplots()
        ax.plot(t[len(t) - len(rg_ma):], rg_ma)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig, suffix="ma_Rg")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot cumulative moving (running) average of radius of gyration
    """
    def plot_cma_Rg(self, Rg_data):
        t = Rg_data[:,0]
        rg = Rg_data[:,1]

        rg_cma = self.cumulative_moving_average(rg)
        fig, ax = plt.subplots()
        ax.plot(t, rg_cma)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Radius of gyration ($\AA$)")
        self.save_figure(fig, suffix="cma_Rg")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    RMSD worker and helper function

    Tests in tests/test_orderparams.py
    """
    def calc_RMSD_worker(self, initcoords, coords, aligninitcoords, aligncoords):
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

    def calc_RMSD(self, u, refu, reftstep, selection, alignment):
        sel = u.select_atoms(selection)
        refsel = refu.select_atoms(selection)
        align = u.select_atoms(alignment)
        refalign = refu.select_atoms(alignment)

        refu.trajectory[reftstep]
        initpos = refsel.positions.copy()
        aligninitpos = refalign.positions.copy()

        RMSD = []
        for ts in u.trajectory[0::self.skip]:
            RMSDval = self.calc_RMSD_worker(initpos, sel.positions, aligninitpos, align.positions)
            RMSD.append((u.trajectory.time, RMSDval))
        RMSD = np.array(RMSD)
        return RMSD

    """
    Plot RMSD
    """
    def plot_RMSD(self, RMSD_data):
        rmsd = RMSD_data[:,1]
        t = RMSD_data[:,0]

        fig, ax = plt.subplots()
        ax.plot(t,rmsd)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig,suffix="RMSD_"+self.align+"_"+self.select)
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot moving (sliding window) average of RMSD
    """
    def plot_ma_RMSD(self, RMSD_data):
        rmsd = RMSD_data[:,1]
        t = RMSD_data[:,0]

        rmsd_ma = self.moving_average(t, rmsd, self.window)
        fig, ax = plt.subplots()
        ax.plot(t[len(t) - len(rmsd_ma):], rmsd_ma)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig,suffix="ma_RMSD_"+self.align+"_"+self.select)
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot cumulative moving (running) average of RMSD
    """
    def plot_cma_RMSD(self, RMSD_data):
        rmsd = RMSD_data[:,1]
        t = RMSD_data[:,0]

        rmsd_ma = self.cumulative_moving_average(rmsd)
        fig, ax = plt.subplots()
        ax.plot(t, rmsd_ma)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"RMSD ($\AA$)")
        self.save_figure(fig, suffix="cma_RMSD"+self.align+"_"+self.select)
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Per-atom-deviation [each heavy atom] worker and helper function
    """
    def calc_deviation_worker(self, initcoords, coords, aligninitcoords, aligncoords):
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

        #calculate deviations
        deviation = np.sqrt(np.sum((coords - initcoords)**2, axis=1))
        return deviation

    def calc_deviation(self, u, refu, reftstep, selection, alignment):
        sel = u.select_atoms(selection)
        refsel = refu.select_atoms(selection)
        align = u.select_atoms(alignment)
        refalign = refu.select_atoms(alignment)

        refu.trajectory[reftstep]
        initpos = refsel.positions.copy()
        aligninitpos = refalign.positions.copy()

        deviations = []
        for ts in u.trajectory[0::self.skip]:
            deviation = self.calc_deviation_worker(initpos, sel.positions, aligninitpos, align.positions)
            deviations.append(deviation)
        deviations = np.array(deviations)
        return deviations

    """
    Plot per-atom-deviation
    """
    def plot_deviations(self, times, deviations):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(deviations, origin="lower", cmap="hot", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Atom')
        ax.set_ylabel('Time (ps)')
        #SET TICKS
        ticks = ax.get_yticks().tolist()
        factor = times[-1]/ticks[-2]
        newlabels = [factor * item for item in ticks]
        ax.set_yticklabels(newlabels)
        self.save_figure(fig, suffix="deviations")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Save instantaneous per-atom-deviation data to PDB
    """
    def save_pdb(self, selection):
        protein_subselection = self.u.select_atoms(selection)
        self.u.add_TopologyAttr('tempfactors')
        utraj = self.u.trajectory[0::self.skip]
        pdbtrj = self.opref + "_deviations.pdb"

        if self.verbose:
            pbar = tqdm(desc = "Writing PDB", total = len(utraj))

        with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=self.u.atoms.n_atoms) as PDB:
            for tidx, ts in enumerate(utraj):
                protein_subselection.atoms.tempfactors = self.deviations[tidx,:]
                PDB.write(self.u.atoms)
                if self.verbose:
                    pbar.update(1)

    """call"""
    def __call__(self):
        # Retrieve value stored in parser if exists, else use as-is
        mda_select = self.selection_parser.get(self.select, self.select)
        mda_align = self.selection_parser.get(self.align, self.align)
        mda_deviation_select = self.selection_parser.get("protein", "protein")

        if self.replot:
            #TODO
            sel_Rg = np.load(self.replotpref + "_Rg.npy")
            sel_Rg = sel_Rg.T
            sel_RMSD = np.load(self.replotpref + "_RMSD_"+self.align+"_"+self.select+".npy")
            sel_RMSD = sel_RMSD.T
            self.dev_times = np.load(self.replotpref + "_dev_times.npy")
            self.deviations = np.load(self.replotpref + "_deviations.npy")
        else:
            sel_Rg = self.calc_Rg(self.u, mda_select)
            sel_RMSD = self.calc_RMSD(self.u, self.refu, self.reftstep, mda_select, mda_align)

            """Log Rg and RMSD data"""
            self.save_timeseries(sel_Rg[:,0], sel_Rg[:,1], label="Rg")
            self.save_timeseries(sel_RMSD[:,0], sel_RMSD[:,1], label="RMSD_"+self.align+"_"+self.select)

            """Calculate per-atom deviations of all protein atoms"""
            times = sel_RMSD[:,0]
            deviations = self.calc_deviation(self.u, self.refu, self.reftstep, mda_deviation_select, mda_align)
            self.dev_times = times
            self.deviations = deviations

            """Log per-atom deviations"""
            np.save(self.opref + "_dev_times", self.dev_times)
            np.save(self.opref + "_deviations", self.deviations)

        """Radius of gyration plots"""
        self.plot_Rg(sel_Rg)
        self.plot_ma_Rg(sel_Rg)
        self.plot_cma_Rg(sel_Rg)

        """RMSD plots"""
        self.plot_RMSD(sel_RMSD)
        self.plot_ma_RMSD(sel_RMSD)
        self.plot_cma_RMSD(sel_RMSD)

        """Per-atom deviation plot"""
        self.plot_deviations(self.dev_times, self.deviations)

        """Store per-atom deviations in PDB"""
        if self.genpdb:
            self.save_pdb(mda_deviation_select)
