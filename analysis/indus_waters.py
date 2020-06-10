"""
Plot number of waters in probe volume output by GROMACS-INDUS simulation

Outputs
- Number of waters in probe volume
- Number of waters in probe volume - moving (sliding window) average
- Number of waters in probe volume - cumulative moving (running) average

Units:
- time: ps

@Author: Akash Pallath
"""
from analysis.timeseries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from tqdm import tqdm

from meta_analysis.profiling import timefunc #for function run-time profiling

class IndusWaters(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("file", help="GROMACS-INDUS waters data file")
        self.parser.add_argument("--genpdb", help="Count atoms in each spherical probe volume, and generate PDB with data", action="store_true")
        self.parser.add_argument("-structf", help="[genpdb] Topology or structure file (.tpr, .gro)")
        self.parser.add_argument("-trajf", help="[genpdb] Compressed trajectory file (.xtc)")
        self.parser.add_argument("-radius", help="[genpdb] Probe volume radiu (in A)")
        self.parser.add_argument("-skip", help="[genpdb] Frame-selection interval (default = 1)")
        self.parser.add_argument("--verbose", help="[genpdb] Display progress", action="store_true")

    def read_args(self):
        super().read_args()
        self.file = self.args.file

        self.genpdb = self.args.genpdb
        self.structf = self.args.structf
        self.trajf = self.args.trajf

        self.radius = self.args.radius
        if self.radius is not None:
            self.radius = float(self.radius)

        self.skip = self.args.skip
        if self.skip is not None:
            self.skip = int(self.skip)
        else:
            self.skip = 1

        self.verbose = self.args.verbose

        # INDUS waters data file
        self.t, self.N, self.Ntw, self.mu = self.get_data(self.file)

        # Prepare system from args
        if self.genpdb:
            self.u = mda.Universe(self.structf, self.trajf)

    """
    Read data from file to prepare system
    """
    def get_data(self, file):
        t = []
        N = []
        Ntw = []
        mu = 0
        with open(file) as f:
            #read data file
            for l in f:
                lstrip = l.strip()
                #parse comments
                if lstrip[0]=='#':
                    comment = lstrip[1:].split()
                    if comment[0] == 'mu':
                        mu = comment[2]
                #parse data
                if lstrip[0]!='#':
                    (tcur,Ncur,Ntwcur) = map(float,lstrip.split())
                    t.append(tcur)
                    N.append(Ncur)
                    Ntw.append(Ntwcur)

        t = np.array(t)
        N = np.array(N)
        Ntw = np.array(Ntw)
        mu = np.float(mu)

        return t, N, Ntw, mu

    """
    Plot waters in probe volume
    """
    def plot_waters(self):
        fig, ax = plt.subplots()
        ax.plot(self.t,self.Ntw,label=r"$\tilde{N}$")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Continuous number of waters")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot moving (sliding window) average of waters in probe volume
    """
    def plot_ma_waters(self):
        maNtw = self.moving_average(self.t, self.Ntw, self.window)

        fig, ax = plt.subplots()
        ax.plot(self.t[len(self.t) - len(maNtw):], maNtw, label=r"$\tilde{N}$, moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Continuous number of waters, moving (window) average")
        ax.legend()
        self.save_figure(fig,suffix="ma_waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Plot cumulative moving (running) average of waters in probe volume
    """
    def plot_cma_waters(self):
        cmaNtw = self.cumulative_moving_average(self.Ntw)
        fig, ax = plt.subplots()
        ax.plot(self.t, cmaNtw, label=r"$\tilde{N}$, cum. moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Continuous number of waters, cumulative moving (running) average")
        ax.legend()
        self.save_figure(fig,suffix="cma_waters")
        if self.show:
            plt.show()
        else:
            plt.close()

    """
    Append mean waters to text file
    """
    def report_mean(self):
        meanstr = "{:.2f} {:.2f}\n".format(self.mu, self.ts_mean(self.t, self.Ntw, self.obsstart, self.obsend))
        with open(self.obspref+"_mean.txt", 'a+') as meanf:
            meanf.write(meanstr)

    """
    Append standard deviation of waters to text file
    """
    def report_std(self):
        stdstr = "{:.2f} {:.2f}\n".format(self.mu, self.ts_std(self.t, self.Ntw, self.obsstart, self.obsend))
        with open(self.obspref+"_std.txt", 'a+') as stdf:
            stdf.write(stdstr)

    """
    Calculate probe waters
    - Note: also saves calculated probe waters data to file
    """
    def calc_probe_waters(self):
        protein = self.u.select_atoms("protein")
        protein_heavy = self.u.select_atoms("protein and not name H*")
        utraj = self.u.trajectory[0::self.skip]
        self.atom_waters = np.zeros((len(utraj), len(protein)))

        if self.verbose:
            bar = tqdm(desc = "Calculating waters", total = len(utraj))

        self.times = []

        for tidx, ts in enumerate(utraj):
            self.times.append(ts.time)
            for atom in protein_heavy.atoms:
                waters = self.u.select_atoms("name OW and (around {} (atom {} {} {}))".format(\
                                        self.radius, atom.segid, atom.resid, atom.name))
                self.atom_waters[tidx, atom.index] = len(waters)
            if self.verbose:
                bar.update(1)

        np.save(self.opref + "_waters", self.atom_waters)

    """
    Save probe waters to PDB
    """
    def save_pdb(self):
        protein = self.u.select_atoms("protein")
        self.u.add_TopologyAttr('tempfactors')
        utraj = self.u.trajectory[0::self.skip]
        pdbtrj = self.opref + "_waters.pdb"

        if self.verbose:
            pbar = tqdm(desc = "Writing PDB", total = len(utraj))

        with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=self.u.atoms.n_atoms) as PDB:
            for tidx, ts in enumerate(utraj):
                protein.atoms.tempfactors = self.atom_waters[tidx,:]
                PDB.write(self.u.atoms)
                if self.verbose:
                    pbar.update(1)

    """
    Plot waters around each heavy atom
    """
    def plot_heavy_waters(self, times, atom_waters):
        fig, ax = plt.subplots(dpi=300)
        im = ax.imshow(atom_waters, origin="lower", cmap="hot", aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Atom')
        ax.set_ylabel('Time (ps)')
        ax.set_yticklabels([str(t) for t in times])
        self.save_figure(fig, suffix="waters_per_heavy_atom")
        if self.show:
            plt.show()
        else:
            plt.close()

    def __call__(self):
        """Log data"""
        self.save_timeseries(self.t, self.N, label="N")
        self.save_timeseries(self.t, self.Ntw, label="Ntw")

        """Plots"""
        self.plot_waters()
        self.plot_ma_waters()
        self.plot_cma_waters()

        """Report observables to text files"""
        self.report_mean()
        self.report_std()

        """Generate PDB and save"""
        if self.genpdb:
            self.calc_probe_waters()
            self.save_pdb()
            self.plot_heavy_waters(self.times, self.atom_waters)

@timefunc
def main():
    warnings = ""
    waters = IndusWaters()
    waters.parse_args()
    waters.read_args()
    startup_string = "#### INDUS Waters ####\n" + warnings + "\n"
    print(startup_string)
    waters()
    plt.close('all')

if __name__=="__main__":
    main()
