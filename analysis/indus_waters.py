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

    #read data
    def get_data(self):
        self.read_args()
        t = []
        N = []
        Ntw = []
        mu = 0
        with open(self.args.file) as f:
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

            self.t = np.array(t)
            self.N = np.array(N)
            self.Ntw = np.array(Ntw)
            self.mu = np.float(mu)

    def __call__(self):
        self.get_data()
        #log data
        self.save_timeseries(self.t,self.N,label="N")
        self.save_timeseries(self.t,self.N,label="Ntw")

        #plot time series data
        fig, ax = plt.subplots()
        ax.plot(self.t,self.N,label="$N$")
        ax.plot(self.t,self.Ntw,label="$N_{tw}$")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of waters")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.args.show:
            plt.show()

        #plot moving average data
        maN = self.moving_average(self.N,self.args.window)
        maNtw = self.moving_average(self.Ntw,self.args.window)
        fig, ax = plt.subplots()
        ax.plot(self.t[len(self.t) - len(maN):], maN, label="$N$, moving average")
        ax.plot(self.t[len(self.t) - len(maN):], maNtw, label="$N_{tw}$, moving average")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Number of waters, moving average")
        ax.legend()
        self.save_figure(fig,suffix="ma_waters")
        if self.args.show:
            plt.show()

        #getaverages
        N_eff_samp, N_mean, N_std, N_sem, N_tau = self.average(self.t, self.N, self.args.avgstart, self.args.avgend)
        Ntw_eff_samp, Ntw_mean, Ntw_std, Ntw_sem, Ntw_tau = self.average(self.t, self.Ntw, self.args.avgstart, self.args.avgend)

        #append averages
        if self.args.avgto is not None:
            with open(self.args.avgto, "a+") as f:
                f.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(\
                self.mu, N_eff_samp, N_mean, N_std, N_sem, Ntw_eff_samp, Ntw_mean, Ntw_std, Ntw_sem, N_tau, Ntw_tau))

        #append to previous data (if available), and plot
        if self.args.apref is not None:
            tNp = np.load(self.args.apref + "_N.npy")
            tp = tNp[0,:]
            Np = tNp[1,:]
            tNtwp = np.load(self.args.apref + "_Ntw.npy")
            Ntwp = tNtwp[1,:]

            tn = tp[-1]+self.t

            #plot time series data
            fig, ax = plt.subplots()
            ax.plot(tp,Np,label="$N$, " + self.aprevlegend)
            ax.plot(tn,self.N,label="$N$, " + self.acurlegend)
            ax.plot(tp,Ntwp,label="$N_{tw}$, " + self.aprevlegend)
            ax.plot(tn,self.Ntw,label="$N_{tw}$, " + self.acurlegend)
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of waters")
            ax.legend()
            self.save_figure(fig,suffix="app_waters")
            if self.args.show:
                plt.show()

            #plot time series data moving average
            ttot = np.hstack([tp,tn])
            Ntot = np.hstack([Np,self.N])
            Ntwtot = np.hstack([Ntwp,self.Ntw])

            maN = self.moving_average(Ntot,self.args.window)
            maNtw = self.moving_average(Ntwtot,self.args.window)
            fig, ax = plt.subplots()
            ax.plot(ttot[len(ttot) - len(maN):], maN, label="$N$, moving average")
            ax.plot(ttot[len(ttot) - len(maN):], maNtw, label="$N_{tw}$, moving average")
            #separator line
            ax.axvline(x=tp[-1])

            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Number of waters, moving average")
            ax.legend()
            self.save_figure(fig,suffix="app_ma_waters")
            if self.args.show:
                plt.show()

warnings = ""

if __name__=="__main__":
    waters = IndusWaters()
    startup_string = "#### INDUS Waters ####\n" + warnings
    print(startup_string)
    waters()
