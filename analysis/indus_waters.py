"""Plot number of waters in probe volume output by GROMACS-INDUS simulation

Units:
- time: ps

@Author: Akash Pallath
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

    # Read data from file to prepare system
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

    def __call__(self):
        # Log data
        self.save_timeseries(self.t,self.N,label="N")
        self.save_timeseries(self.t,self.N,label="Ntw")

        # Plot time series data
        fig, ax = plt.subplots()
        ax.plot(self.t,self.N,label="$N$")
        ax.plot(self.t,self.Ntw,label="$N_{tw}$")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of waters")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.show:
            plt.show()

        # Plot moving average data
        maN = self.moving_average(self.N,self.window)
        maNtw = self.moving_average(self.Ntw,self.window)
        fig, ax = plt.subplots()
        ax.plot(self.t[len(self.t) - len(maN):], maN, label="$N$, moving average")
        ax.plot(self.t[len(self.t) - len(maN):], maNtw, label="$N_{tw}$, moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of waters, moving average")
        ax.legend()
        self.save_figure(fig,suffix="ma_waters")
        if self.show:
            plt.show()

        # Plot cumulative moving average data
        cmaN = self.cumulative_moving_average(self.N)
        cmaNtw = self.cumulative_moving_average(self.Ntw)
        fig, ax = plt.subplots()
        ax.plot(self.t, cmaN, label="$N$, cum. moving average")
        ax.plot(self.t, cmaNtw, label="$N_{tw}$, cum. moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of waters, cumulative moving average")
        ax.legend()
        self.save_figure(fig,suffix="cma_waters")
        if self.show:
            plt.show()

        # Append to previous data (if available), and plot
        if self.apref is not None:
            tNp = np.load(self.apref + "_N.npy")
            tp = tNp[0,:]
            Np = tNp[1,:]
            tNtwp = np.load(self.apref + "_Ntw.npy")
            Ntwp = tNtwp[1,:]

            tn = tp[-1]+self.t

            # Plot time series data
            fig, ax = plt.subplots()
            ax.plot(tp,Np,label="$N$, " + self.aprevlegend)
            ax.plot(tn,self.N,label="$N$, " + self.acurlegend)
            ax.plot(tp,Ntwp,label="$N_{tw}$, " + self.aprevlegend)
            ax.plot(tn,self.Ntw,label="$N_{tw}$, " + self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of waters")
            ax.legend()
            self.save_figure(fig,suffix="app_waters")
            if self.show:
                plt.show()

            # Plot time series data moving average
            ttot = np.hstack([tp,tn])
            Ntot = np.hstack([Np,self.N])
            Ntwtot = np.hstack([Ntwp,self.Ntw])

            maN = self.moving_average(Ntot,self.window)
            maNtw = self.moving_average(Ntwtot,self.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(maN):], maN, label="$N$, moving average")
            ax.plot(ttot[len(ttot) - len(maN):], maNtw, label="$N_{tw}$, moving average")
            #separator line
            ax.axvline(x=tp[-1])
            #labels
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of waters, moving average")
            ax.legend()
            self.save_figure(fig,suffix="app_ma_waters")
            if self.show:
                plt.show()

            # Plot time series data cumulative moving average
            ttot = np.hstack([tp,tn])
            Ntot = np.hstack([Np,self.N])
            Ntwtot = np.hstack([Ntwp,self.Ntw])

            cmaN = self.cumulative_moving_average(Ntot)
            cmaNtw = self.cumulative_moving_average(Ntwtot)
            fig, ax = plt.subplots()
            ax.plot(ttot, cmaN, label="$N$, cum. moving average")
            ax.plot(ttot, cmaNtw, label="$N_{tw}$, cum. moving average")
            #separator line
            ax.axvline(x=tp[-1])

            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of waters, cumulative moving average")
            ax.legend()
            self.save_figure(fig,suffix="app_cma_waters")
            if self.show:
                plt.show()

warnings = ""

if __name__=="__main__":
    waters = IndusWaters()
    waters.read_args()
    startup_string = "#### INDUS Waters ####\n" + warnings
    print(startup_string)
    waters()
