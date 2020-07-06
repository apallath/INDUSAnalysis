"""
Defines class for analysing waters in INDUS probe volumes
"""

from INDUSAnalysis import timeseries
from INDUSAnalysis.lib import profiling

import numpy as np

import matplotlib.pyplot as plt
import MDAnalysis as mda
from tqdm import tqdm

"""Cython"""
cimport numpy as np


class WatersAnalysis(timeseries.TimeSeriesAnalysis):
    """
    Calculates number of waters in individual and union probe volumes. Generates
    plots and PDB files.
    """
    def __init__(self):
        super().__init__()
        self.req_file_args.add_argument("file",
                                        help="GROMACS-INDUS waters data file")
        self.req_file_args.add_argument("structf",
                                        help="Topology or structure file (.tpr, .gro)")
        self.req_file_args.add_argument("trajf",
                                        help="Compressed trajectory file (.xtc)")

        self.calc_args.add_argument("-radius",
                                    help="[per-probe waters, ignored during replot] Probe volume radius (in A) (default = 6 A)")
        self.calc_args.add_argument("-skip",
                                    help="[per-probe waters, ignored during replot] Sampling interval (default = 1)")

        self.out_args.add_argument("--genpdb",
                                   action="store_true",
                                   help="[per-probe waters] Write atoms per probe volume data to pdb file")

        self.misc_args.add_argument("--verbose",
                                    action="store_true",
                                    help="Display progress")

    def read_args(self):
        """
        Stores arguments from TimeSeries `args` parameter in class variables
        """
        super().read_args()
        self.file = self.args.file
        self.structf = self.args.structf
        self.trajf = self.args.trajf

        self.u = mda.Universe(self.structf, self.trajf)

        self.radius = self.args.radius
        if self.radius is not None:
            self.radius = float(self.radius)
        else:
            self.radius = 6.0

        self.skip = self.args.skip
        if self.skip is not None:
            self.skip = int(self.skip)
        else:
            self.skip = 1

        self.genpdb = self.args.genpdb
        self.verbose = self.args.verbose

    # Data calculation methods

    def read_waters(self, filename):
        """
        Reads data from GROMACS-INDUS phi/probe waters output file.

        Args:
            filename (str): Name of GROMACS-INDUS waters output file.

        Returns:
            {
                ts_N (TimeSeries): N values.
                ts_Ntw (TimeSeries): N~ values.
                mu (np.float): Value of mu.
            }.
        """
        t = []
        N = []
        Ntw = []
        mu = 0
        with open(filename) as f:
            # Read data file
            for l in f:
                lstrip = l.strip()
                # Parse comments
                if lstrip[0] == '#':
                    comment = lstrip[1:].split()
                    if comment[0] == 'mu':
                        mu = comment[2]
                # Parse data
                if lstrip[0] != '#':
                    (tcur, Ncur, Ntwcur) = map(float, lstrip.split())
                    t.append(tcur)
                    N.append(Ncur)
                    Ntw.append(Ntwcur)

        t = np.array(t)
        N = np.array(N)
        Ntw = np.array(Ntw)
        mu = np.float(mu)

        ts_N = timeseries.TimeSeries(t, N, labels=["N"])
        ts_Ntw = timeseries.TimeSeries(t, Ntw, labels=[r"N~"])

        return ts_N, ts_Ntw, mu

    def calc_probe_waters(self, u, skip, radius):
        """
        Calculates waters in individual probe volumes

        Args:
            u (mda.Universe): Universe containing solvated protein
            skip (int): Trajectory resampling interval
            radius (float): Radius of probe waters

        Returns:
            TimeSeries object containing probe waters
        """
        # Probes placed on protein-heavy atoms
        protein = u.select_atoms("protein")
        protein_heavy = u.select_atoms("protein and not name H*")

        utraj = u.trajectory[::skip]
        times = np.zeros(len(utraj))
        probe_waters = np.zeros((len(utraj), len(protein)))

        if self.verbose:
            bar = tqdm(desc="Calculating waters", total=len(utraj))

        for tidx, ts in enumerate(utraj):
            times[tidx] = ts.time
            for atom in protein_heavy.atoms:
                waters = u.select_atoms("name OW and (around {} (atom {} {} {}))".format(
                                        radius, atom.segid, atom.resid, atom.name))
                probe_waters[tidx, atom.index] = len(waters)
            if self.verbose:
                bar.update(1)

        return timeseries.TimeSeries(times, probe_waters,
                                     labels=['Number of waters', 'Heavy atom index'])

    def plot_waters(self, ts_Ntw):
        """Plots waters and saves figure to file"""
        fig = ts_Ntw.plot()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_ma_waters(self, ts_Ntw):
        """Plots moving average waters and saves figure to file"""
        fig = ts_Ntw.moving_average(window=self.window).plot()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="ma_waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_cma_waters(self, ts_Ntw):
        """Plots cumulative moving average waters and saves figure to file"""
        fig = ts_Ntw.cumulative_moving_average().plot()
        fig.set_dpi(300)
        self.save_figure(fig, suffix="cma_waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_probe_waters(self, ts_probe_waters):
        """Plots waters in each individual probe as a 2D heatmap"""
        fig = ts_probe_waters.plot_2d_heatmap(cmap='hot')
        fig.set_dpi(300)
        self.save_figure(fig, suffix="probe_waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    def write_mean_std_waters(self, mu, ts_Ntw):
        """Appends mean and std waters to text file"""
        meanstr = "{:.2f} {:.2f}\n".format(mu, ts_Ntw.mean())
        with open(self.obspref + "_mean.txt", 'a+') as meanf:
            meanf.write(meanstr)

        stdstr = "{:.2f} {:.2f}\n".format(mu, ts_Ntw.std())
        with open(self.obspref + "_std.txt", 'a+') as stdf:
            stdf.write(stdstr)

    def write_probe_waters_pdb(self, u, skip, ts_probe_waters):
        """
        Writes instantaneous probe waters to PDB file.

        Args:
            u (mda.Universe): Universe containing solvated protein
            skip (int): Trajectory resampling interval
            ts_probe_waters (TimeSeries): Probe waters timeseries data.

        Raises:
            ValueError if the time for the same index in u.trajectory[::skip]
            and ts_probe_waters does not match.
        """
        protein = u.select_atoms("protein")
        u.add_TopologyAttr('tempfactors')
        pdbtrj = self.opref + "_waters.pdb"

        utraj = u.trajectory[::skip]

        if self.verbose:
            pbar = tqdm(desc="Writing PDB", total=len(utraj))

        with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
            for tidx, ts in enumerate(utraj):
                if np.isclose(ts.time, ts_probe_waters.time_array[tidx]):
                    protein.atoms.tempfactors = ts_probe_waters.data_array[tidx, :]
                    PDB.write(u.atoms)
                    if self.verbose:
                        pbar.update(1)
                else:
                    raise ValueError("Trajectory and TimeSeries times do not match at same index.")

    def __call__(self):
        """Performs analysis"""

        """Raw data"""
        # Overall probe waters
        ts_N, ts_Ntw, mu = self.read_waters(self.file)
        self.save_TimeSeries(ts_N, self.opref + "_N.pkl")
        self.save_TimeSeries(ts_Ntw, self.opref + "_Ntw.pkl")

        # Individual probe waters
        u = mda.Universe(self.structf, self.trajf)
        ts_probe_waters = None
        if self.replot:
            ts_probe_waters = self.load_TimeSeries(self.replotpref + "_probe_waters.pkl")
        else:
            ts_probe_waters = self.calc_probe_waters(u, self.skip, self.radius)

        self.save_TimeSeries(ts_probe_waters, self.opref + "_probe_waters.pkl")

        """Plots and averages"""
        # Plot waters, moving average waters, cumulative moving average waters,
        # and save figures
        self.plot_waters(ts_Ntw)
        self.plot_ma_waters(ts_Ntw)
        self.plot_cma_waters(ts_Ntw)

        # Plot heatmap of waters in individual probe volumes, and save figure
        self.plot_probe_waters(ts_probe_waters)

        """Trajectories"""
        # Write waters in individual probe volumes to PDB
        if self.genpdb:
            u = mda.Universe(self.structf, self.trajf)
            self.write_probe_waters_pdb(u, self.skip, ts_probe_waters)

        """Observables"""
        # Write mean and std of waters to text files
        self.write_mean_std_waters(mu, ts_Ntw[self.obsstart:self.obsend])
