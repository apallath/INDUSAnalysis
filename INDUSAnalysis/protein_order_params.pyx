"""
Defines class for analysing protein order parameters along GROMACS
simulation trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
import MDAnalysis.analysis.align

from tqdm import tqdm

from INDUSAnalysis import timeseries
from INDUSAnalysis.lib import profiling

"""Cython"""
cimport numpy as np


class OrderParamsAnalysis(timeseries.TimeSeriesAnalysis):
    """
    Calculates order parameters.
    """
    def __init__(self):
        super().__init__()
        self.req_file_args.add_argument("structf", help="Structure file (.gro)")
        self.req_file_args.add_argument("trajf", help="Compressed trajectory file (.xtc)")

        self.opt_file_args.add_argument("-reftrajf",
                                        help="Reference trajectory file (.xtc) for RMSD (default: same as trajf)")

        self.calc_args.add_argument("-select",
                                    help="Atoms/groups to track order parameters for (MDA selection string)")
        self.calc_args.add_argument("-align",
                                    help="Atoms/groups for aligning trajectories across timesteps (MDA selection string)")
        self.calc_args.add_argument("-skip",
                                    help="Number of frames to skip between analyses (default = None)")
        self.calc_args.add_argument("-reftstep",
                                    help="Timestep to extract reference coordinates from reference trajectory file for RMSD (default = 0)")

        self.out_args.add_argument("--genpdb",
                                   action="store_true",
                                   help="Write atoms per probe volume data to pdb file")

        self.misc_args.add_argument("--verbose",
                                    action="store_true",
                                    help="Display progress")

    def read_args(self):
        """
        Stores arguments from TimeSeries `args` parameter in class variables.
        """
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

        self.u = mda.Universe(self.structf, self.trajf)

        self.reftrajf = self.args.reftrajf

        if self.reftrajf is not None:
            self.refu = mda.Universe(self.structf, self.reftrajf)
        else:
            self.refu = mda.Universe(self.structf, self.trajf)

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

        self.reftstep = self.args.reftstep
        if self.reftstep is not None:
            self.reftstep = int(self.reftstep)
        else:
            self.reftstep = 0

        self.genpdb = self.args.genpdb
        self.verbose = self.args.verbose

    def calc_Rg_worker(self, coords, masses):
        """
        Calculates radius of gyration of atoms with given coordinates and
        masses.

        Args:
            coords (np.array): Array of shape (N, 3) containing atomic coordinates.
            masses (np.array): Array of shape (N,) containing atomic masses.

        Returns:
            Radius of gyration (np.float).

        Raises:
            ValueError if coords does not have the right shape, or if the
            lengths of coords and masses do not match.
        """
        if coords.shape[1] != 3:
            raise ValueError("coords not 3 dimensional")

        if coords.shape[0] != masses.shape[0]:
            raise ValueError("coords and masses not of same length")

        com = np.average(coords, weights=masses, axis=0)
        sq_distances = np.sum((coords - com)**2, axis=1)
        Rg = np.sqrt(np.average(sq_distances, weights=masses))
        return Rg

    def calc_Rg(self, u, skip, selection):
        """
        Calculates radius of gyration of selection along trajectory.

        Args:
            u (mda.Universe): Universe.
            skip (int): Resampling interval.
            selection (str): MDAnalysis selection string.

        Returns:
            TimeSeries object containing Rg values along trajectory.
        """
        times = []
        Rgs = []
        sel = u.select_atoms(selection)
        for ts in u.trajectory[0::skip]:
            Rgval = self.calc_Rg_worker(sel.positions, sel.masses)
            times.append(ts.time)
            Rgs.append(Rgval)
        ts_Rg = timeseries.TimeSeries(np.array(times), np.array(Rgs),
                                      labels=['Rg'])
        return ts_Rg

    def plot_Rg(self, ts_Rg):
        """Plots Rg and saves figure to file."""
        fig = ts_Rg.plot()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="Rg")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_ma_Rg(self, ts_Rg, window):
        """Plots moving average Rg and saves figure to file."""
        fig = ts_Rg.moving_average(window=window).plot()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="ma_Rg")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_cma_Rg(self, ts_Rg):
        """Plots cumulative moving average Rg and saves figure to file."""
        fig = ts_Rg.cumulative_moving_average().plot()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="cma_Rg")
        if self.show:
            plt.show()
        else:
            plt.close()

    def calc_RMSD_worker(self, initcoords, coords, aligninitcoords, aligncoords):
        """
        Calculates RMSD of coordinates from reference after aligning them using
        a subset of coordinates.

        Aligns coords and initcoords by performing translation and rotation
        transformations which, when applied to aligncoords and aligninitcoords,
        minimizes the RMSD between them.

        Args:
            initcoords (np.array): Array of shape (N, 3) containing reference coordinates.
            coords (np.array): Array of shape (N, 3) containing coordinates.
            aligninitcoords (np.array): Array of shape (N, 3) containing alignment group of reference coordinates.
            aligncoords (np.Array): Array of shape (N, 3) containing alignment group of coordinates.

        Returns:
            RMSD (np.float).

        Raises:
            ValueError if the coordinates are not 3-dimensional or are inconsistent.
        """
        if not (coords.shape[1] == 3 and initcoords.shape[1] == 3 and aligncoords.shape[1] == 3 and aligninitcoords.shape[1] == 3):
            raise ValueError('One or more oordinates are not 3 dimensional')
        if not (coords.shape == initcoords.shape and aligncoords.shape == aligninitcoords.shape):
            raise ValueError('Set of coordinates and reference set coordinates do not have same shape')

        # Calculate centers of geometry of alignment coordinates
        aligninitcog = np.mean(aligninitcoords, axis=0)
        aligncog = np.mean(aligncoords, axis=0)

        # Center both alignment coordinates and coordinates
        aligninitcoords = aligninitcoords - aligninitcog
        initcoords = initcoords - aligninitcog
        aligncoods = aligncoords - aligncog
        coords = coords - aligncog

        # Get rotation matrix by minimizing RMSD between centered alignment coordinates
        R, min_rms = mda.analysis.align.rotation_matrix(aligncoords, aligninitcoords)

        # Rotate coords
        coords = np.dot(coords, R.T)

        # Calculate RMSD
        sq_distances = np.sum((coords - initcoords)**2, axis=1)
        RMSD = np.sqrt(np.mean(sq_distances))
        return RMSD

    def calc_RMSD(self, u, refu, reftstep, skip, selection, alignment):
        """
        Calculates RMSD of `selection` atom group in `u` from `selection` atom group
        in `refu` at `reftstep`, using `alignment` for alignment.

        Args:
            u (mda.Universe): Universe object.
            refu (mda.Universe): Reference Universe object.
            reftstep (int): Reference timestep to calculate RMSD from.
            skip (int): Resampling interval.
            selection (mda.AtomGroup): MDAnalysis AtomGroup object containing atoms
                to calculate RMSD for.
            alignment (mda.AtomGroup): MDAnalysis AtomGroup object containing atoms
                to use for alignment before calculating RMSD.

        Returns:
            TimeSeries object containing RMSD values along trajectory.
        """
        sel = u.select_atoms(selection)
        refsel = refu.select_atoms(selection)
        align = u.select_atoms(alignment)
        refalign = refu.select_atoms(alignment)

        refu.trajectory[reftstep]
        initpos = refsel.positions.copy()
        aligninitpos = refalign.positions.copy()

        times = []
        RMSDs = []
        for ts in u.trajectory[0::skip]:
            RMSDval = self.calc_RMSD_worker(initpos, sel.positions, aligninitpos, align.positions)
            times.append(ts.time)
            RMSDs.append(RMSDval)
        return timeseries.TimeSeries(np.array(times), np.array(RMSDs), labels=['RMSD'])

    def plot_RMSD(self, ts_RMSD):
        """Plots RMSD and saves figure to file."""
        fig = ts_RMSD.plot(label="RMSD between {} atoms, using {} atoms for alignment".format(self.select, self.align))
        ax = fig.gca()
        ax.legend()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="RMSD_" + self.align + "_" + self.select)
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_ma_RMSD(self, ts_RMSD, window):
        """Plots moving average RMSD and saves figure to file."""
        ts_RMSD_ma = ts_RMSD.moving_average(window=window)
        fig = ts_RMSD_ma.plot(label="RMSD between {} atoms, using {} atoms for alignment".format(self.select, self.align))
        ax = fig.gca()
        ax.legend()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="ma_RMSD_" + self.align + "_" + self.select)
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_cma_RMSD(self, ts_RMSD):
        """Plots cumulative moving average RMSD and saves figure to file."""
        ts_RMSD_cma = ts_RMSD.cumulative_moving_average()
        fig = ts_RMSD_cma.plot(label="RMSD between {} atoms, using {} atoms for alignment".format(self.select, self.align))
        ax = fig.gca()
        ax.legend()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="cma_RMSD_" + self.align + "_" + self.select)
        if self.show:
            plt.show()
        else:
            plt.close()

    def calc_deviation_worker(self, initcoords, coords, aligninitcoords, aligncoords):
        """
        Calculates deviations of coordinates from reference after aligning them using
        a subset of coordinates.

        Aligns coords and initcoords by performing translation and rotation
        transformations which, when applied to aligncoords and aligninitcoords,
        minimizes the RMSD between them.

        Args:
            initcoords (np.array): Array of shape (N, 3) containing reference coordinates.
            coords (np.array): Array of shape (N, 3) containing coordinates.
            aligninitcoords (np.array): Array of shape (N, 3) containing alignment group of reference coordinates.
            aligncoords (np.Array): Array of shape (N, 3) containing alignment group of coordinates.

        Returns:
            Array of deviations.

        Raises:
            ValueError if the coordinates are not 3-dimensional or are inconsistent.
        """
        if not (coords.shape[1] == 3 and initcoords.shape[1] == 3 and aligncoords.shape[1] == 3 and aligninitcoords.shape[1] == 3):
            raise ValueError('One or more oordinates are not 3 dimensional')
        if not (coords.shape == initcoords.shape and aligncoords.shape == aligninitcoords.shape):
            raise ValueError('Set of coordinates and reference set coordinates do not have same shape')

        # Calculate centers of geometry of alignment coordinates
        aligninitcog = np.mean(aligninitcoords, axis=0)
        aligncog = np.mean(aligncoords, axis=0)

        # Center both alignment coordinates and coordinates
        aligninitcoords = aligninitcoords - aligninitcog
        initcoords = initcoords - aligninitcog
        aligncoods = aligncoords - aligncog
        coords = coords - aligncog

        # Get rotation matrix by minimizing RMSD between centered alignment coordinates
        R, min_rms = mda.analysis.align.rotation_matrix(aligncoords, aligninitcoords)

        # Rotate coords
        coords = np.dot(coords, R.T)

        # Calculate deviations
        deviation = np.sqrt(np.sum((coords - initcoords)**2, axis=1))
        return deviation

    def calc_deviations(self, u, refu, reftstep, skip, selection, alignment):
        """
        Calculates deviations of `selection` AtomGroup atoms in `u` from `selection` AtomGroup atoms
        in `refu` at `reftstep`, using `alignment` for alignment.

        Args:
            u (mda.Universe): Universe object.
            refu (mda.Universe): Reference Universe object.
            reftstep (int): Reference timestep to calculate RMSD from.
            skip (int): Resampling interval.
            selection (mda.AtomGroup): MDAnalysis AtomGroup object containing atoms
                to calculate RMSD for.
            alignment (mda.AtomGroup): MDAnalysis AtomGroup object containing atoms
                to use for alignment before calculating RMSD.

        Returns:
            2-D TimeSeries object containing deviation values along trajectory.
        """
        sel = u.select_atoms(selection)
        refsel = refu.select_atoms(selection)
        align = u.select_atoms(alignment)
        refalign = refu.select_atoms(alignment)

        refu.trajectory[reftstep]
        initpos = refsel.positions.copy()
        aligninitpos = refalign.positions.copy()

        times = []
        deviations = []
        for ts in u.trajectory[0::skip]:
            times.append(ts.time)
            deviation = self.calc_deviation_worker(initpos, sel.positions, aligninitpos, align.positions)
            deviations.append(deviation)
        return timeseries.TimeSeries(np.array(times), np.array(deviations), labels=['Deviation', 'Atom index'])

    def plot_deviations(self, ts_deviations):
        """Plots deviations as a 2D heatmap."""
        fig = ts_deviations.plot_2d_heatmap(cmap='hot')
        fig.set_dpi(300)
        self.save_figure(fig, suffix="deviation")
        if self.show:
            plt.show()
        else:
            plt.close()

    def write_deviations_pdb(self, u, select, skip, ts_deviations):
        """
        Writes per-atom-deviations to PDB file.

        Args:
            u (mda.Universe): Universe containing solvated protein.
            select (str): MDAnalysis selection string describing atoms deviations are computed for.
            skip (int): Trajectory resampling interval.
            ts_deviations (TimeSeries): Atom deviations timeseries data.

        Raises:
            ValueError if the time for the same index in u.trajectory[::skip]
            and ts_deviations does not match.
        """
        protein_subselection = u.select_atoms(select)
        u.add_TopologyAttr('tempfactors')
        pdbtrj = self.opref + "_deviations.pdb"

        utraj = u.trajectory[::skip]

        if self.verbose:
            pbar = tqdm(desc="Writing PDB", total=len(utraj))

        with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
            for tidx, ts in enumerate(utraj):
                if np.isclose(ts.time, ts_deviations.time_array[tidx]):
                    protein_subselection.atoms.tempfactors = ts_deviations.data_array[tidx, :]
                    PDB.write(u.atoms)
                    if self.verbose:
                        pbar.update(1)
                else:
                    raise ValueError("Trajectory and TimeSeries times do not match at same index.")

    def __call__(self):
        """Performs analysis."""

        # Retrieve value stored in parser if exists, else use as-is
        mda_select = self.selection_parser.get(self.select, self.select)
        mda_align = self.selection_parser.get(self.align, self.align)
        mda_deviation_select = self.selection_parser.get("protein", "protein")

        """Raw data"""

        if self.replot:
            ts_Rg = self.load_TimeSeries(self.replotpref + "_Rg.pkl")
            ts_RMSD = self.load_TimeSeries(self.replotpref + "_RMSD_" + self.align + "_" + self.select + ".pkl")
            ts_deviations = self.load_TimeSeries(self.replotpref + "_deviations.pkl")
        else:
            ts_Rg = self.calc_Rg(self.u, self.skip, mda_select)
            ts_RMSD = self.calc_RMSD(self.u, self.refu, self.reftstep, self.skip, mda_select, mda_align)
            ts_deviations = self.calc_deviations(self.u, self.refu, self.reftstep, self.skip, mda_deviation_select, mda_align)

        self.save_TimeSeries(ts_Rg, self.opref + "_Rg.pkl")
        self.save_TimeSeries(ts_RMSD, self.opref + "_RMSD_" + self.align + "_" + self.select + ".pkl")
        self.save_TimeSeries(ts_deviations, self.opref + "_deviations.pkl")

        """Rg plots"""
        self.plot_Rg(ts_Rg)
        self.plot_ma_Rg(ts_Rg, self.window)
        self.plot_cma_Rg(ts_Rg)

        """RMSD plots"""
        self.plot_RMSD(ts_RMSD)
        self.plot_ma_RMSD(ts_RMSD, self.window)
        self.plot_cma_RMSD(ts_RMSD)

        """Per-atom deviations heatmap plot"""
        self.plot_deviations(ts_deviations)

        """Store per-atom deviations in PDB"""
        if self.genpdb:
            self.write_deviations_pdb(self.u, mda_deviation_select, self.skip, ts_deviations)
