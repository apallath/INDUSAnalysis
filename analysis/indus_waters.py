"""Plot number of waters in probe volume output by GROMACS-INDUS simulation
Outputs
- Number of waters
- Number of waters (sliding window) moving average
- Number of waters cumulative moving average

Units:
- time: ps

@Author: Akash Pallath

TODO:   Mean and CI plot for appended plots
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
        self.save_timeseries(self.t,self.Ntw,label="Ntw")

        # Plot time series data
        fig, ax = plt.subplots()
        ax.plot(self.t,self.Ntw,label=r"$\tilde{N}$")
        # Plot mean and errors
        mean, serr, ci_95_low, ci_95_high = self.average(self.t, self.Ntw, self.avgstart, self.avgend)
        meanline = mean*np.ones(len(self.t))
        ci_low_line = ci_95_low*np.ones(len(self.t))
        ci_high_line = ci_95_high*np.ones(len(self.t))
        ax.plot(self.t,meanline,color='green')
        ax.fill_between(self.t,ci_low_line,ci_high_line,alpha=0.2,facecolor='green',edgecolor='green')
        # Plot properties
        plt.title('Mean = {:.2f}\n95% CI = [{:.2f}, {:.2f}]'.format(mean, ci_95_low, ci_95_high))
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("CG number of waters")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.show:
            plt.show()

        # Plot moving average data
        maNtw = self.moving_average(self.t, self.Ntw, self.window)
        fig, ax = plt.subplots()
        ax.plot(self.t[len(self.t) - len(maNtw):], maNtw, label=r"$\tilde{N}$, moving average")
        # Plot mean and errors
        meanline = mean*np.ones(len(maNtw))
        ci_low_line = ci_95_low*np.ones(len(maNtw))
        ci_high_line = ci_95_high*np.ones(len(maNtw))
        ax.plot(self.t[len(self.t) - len(maNtw):], meanline, color='green')
        ax.fill_between(self.t[len(self.t) - len(maNtw):], ci_low_line, ci_high_line,\
            alpha=0.2,facecolor='green',edgecolor='green')
        # Plot properties
        plt.title('Mean = {:.2f}\n95% CI = [{:.2f}, {:.2f}]'.format(mean, ci_95_low, ci_95_high))
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("CG number of waters, moving average")
        ax.legend()
        self.save_figure(fig,suffix="ma_waters")
        if self.show:
            plt.show()

        # Plot cumulative moving average data
        cmaNtw = self.cumulative_moving_average(self.Ntw)
        fig, ax = plt.subplots()
        ax.plot(self.t, cmaNtw, label=r"$\tilde{N}$, cum. moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("CG number of waters, cumulative moving average")
        ax.legend()
        self.save_figure(fig,suffix="cma_waters")
        if self.show:
            plt.show()

        # Append to previous data (if available), and plot
        if self.apref is not None:
            tNtwp = np.load(self.apref + "_Ntw.npy")
            tp = tNtwp[0,:]
            Ntwp = tNtwp[1,:]

            tn = tp[-1]+self.t

            # Plot time series data
            fig, ax = plt.subplots()
            ax.plot(tp,Ntwp,label=r"$\tilde{N}$, " + self.aprevlegend)
            ax.plot(tn,self.Ntw,label=r"$\tilde{N}$, " + self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("CG number of waters")
            ax.legend()
            self.save_figure(fig,suffix="app_waters")
            if self.show:
                plt.show()

            # Plot time series data moving average
            ttot = np.hstack([tp,tn])
            Ntwtot = np.hstack([Ntwp,self.Ntw])
            maNtw = self.moving_average(ttot, Ntwtot, self.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(maN):], maNtw, label=r"$\tilde{N}$, moving average")
            #separator line
            ax.axvline(x=tp[-1])
            #labels
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("CG number of waters, moving average")
            ax.legend()
            self.save_figure(fig,suffix="app_ma_waters")
            if self.show:
                plt.show()

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
            ax.set_ylabel("CG number of waters, cumulative moving average")
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
