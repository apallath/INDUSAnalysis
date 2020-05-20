"""Plot number of waters in probe volume output by GROMACS-INDUS simulation
Outputs
- Number of waters in probe volume
- Number of waters in probe volume - moving (sliding window) average
- Number of waters in probe volume - cumulative moving (running) average

Units:
- time: ps

@Author: Akash Pallath

FEATURE:    Cythonize code
"""
from analysis.timeseries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt

class IndusWaters(TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("file", help="GROMACS-INDUS waters data file")

    def read_args(self):
        super().read_args()
        self.file = self.args.file

        # Prepare system from args
        self.t, self.N, self.Ntw, self.mu = self.get_data(self.file)

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
        ax.set_ylabel("CG number of waters")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.show:
            plt.show()

        #Append
        if self.apref is not None:
            tNtwp = np.load(self.apref + "_Ntw.npy")
            tp = tNtwp[0,:]
            Ntwp = tNtwp[1,:]

            tn = tp[-1]+self.t

            fig, ax = plt.subplots()
            ax.plot(tp,Ntwp,label=r"$\tilde{N}$, " + self.aprevlegend)
            ax.plot(tn,self.Ntw,label=r"$\tilde{N}$, " + self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("CG number of waters")
            ax.legend()
            self.save_figure(fig,suffix="app_waters")
            if self.show:
                plt.show()

    """
    Plot moving (sliding window) average of waters in probe volume
    """
    def plot_ma_waters(self):
        maNtw = self.moving_average(self.t, self.Ntw, self.window)

        fig, ax = plt.subplots()
        ax.plot(self.t[len(self.t) - len(maNtw):], maNtw, label=r"$\tilde{N}$, moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("CG number of waters, moving (window) average")
        ax.legend()
        self.save_figure(fig,suffix="ma_waters")
        if self.show:
            plt.show()

        #Append
        if self.apref is not None:
            tNtwp = np.load(self.apref + "_Ntw.npy")
            tp = tNtwp[0,:]
            Ntwp = tNtwp[1,:]

            tn = tp[-1]+self.t

            # Plot time series data moving average
            ttot = np.hstack([tp,tn])
            Ntwtot = np.hstack([Ntwp,self.Ntw])
            maNtw = self.moving_average(ttot, Ntwtot, self.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(maNtw):], maNtw, label=r"$\tilde{N}$, moving average")
            #separator line
            ax.axvline(x=tp[-1])
            #labels
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("CG number of waters, moving (window) average")
            ax.legend()
            self.save_figure(fig,suffix="app_ma_waters")
            if self.show:
                plt.show()

    """
    Plot cumulative moving (running) average of waters in probe volume
    """
    def plot_cma_waters(self):
        cmaNtw = self.cumulative_moving_average(self.Ntw)
        fig, ax = plt.subplots()
        ax.plot(self.t, cmaNtw, label=r"$\tilde{N}$, cum. moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("CG number of waters, cumulative moving (running) average")
        ax.legend()
        self.save_figure(fig,suffix="cma_waters")
        if self.show:
            plt.show()

        #Append
        if self.apref is not None:
            tNtwp = np.load(self.apref + "_Ntw.npy")
            tp = tNtwp[0,:]
            Ntwp = tNtwp[1,:]

            tn = tp[-1]+self.t

            # Plot time series data cumulative moving average
            ttot = np.hstack([tp,tn])
            Ntwtot = np.hstack([Ntwp,self.Ntw])
            cmaNtw = self.cumulative_moving_average(Ntwtot)
            fig, ax = plt.subplots()
            ax.plot(ttot, cmaNtw, label=r"$\tilde{N}$, cum. moving average")
            #separator line
            ax.axvline(x=tp[-1])
            #labels
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("CG number of waters, cumulative moving (running) average")
            ax.legend()
            self.save_figure(fig,suffix="app_cma_waters")
            if self.show:
                plt.show()

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

    def __call__(self):
        """Log data"""
        self.save_timeseries(self.t,self.N,label="N")
        self.save_timeseries(self.t,self.Ntw,label="Ntw")

        """Plots"""
        self.plot_waters()
        self.plot_ma_waters()
        self.plot_cma_waters()

        """Report observables to text files"""
        self.report_mean()
        self.report_std()

warnings = ""

if __name__=="__main__":
    waters = IndusWaters()
    waters.parse_args()
    waters.read_args()
    startup_string = "#### INDUS Waters ####\n" + warnings + "\n"
    print(startup_string)
    waters()
