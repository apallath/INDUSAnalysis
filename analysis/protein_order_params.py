"""Plot number of waters in probe volume output by GROMACS-INDUS simulation

@Author: Akash Pallath
"""
import timeseries

import numpy as np
import matplotlib.pyplot as plt

class IndusWaters(timeseries.TimeSeries):
    def __init__(self):
        super().__init__()
        self.parser.add_argument("file", help="GROMACS-INDUS waters data file")

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
        ax.set_xlabel("Time, in ps")
        ax.set_ylabel("$N_v$ and coarse-grained $N_v$")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.args.show:
            plt.show()

        #plot moving average data
        maN = self.moving_average(self.N)
        maNtw = self.moving_average(self.Ntw)
        fig, ax = plt.subplots()
        ax.plot(self.t, maN, label="$N$, moving average")
        ax.plot(self.t, maNtw, label="$N_{tw}$, moving average")
        ax.set_xlabel("Time, in ps")
        ax.set_ylabel("Number of waters, moving average")
        ax.legend()
        self.save_figure(fig,suffix="ma_waters")
        if self.args.show:
            plt.show()

        #getaverages
        N_eff_samp, N_mean, N_std, N_sem, N_tau = self.average(self.t, self.N)
        Ntw_eff_samp, Ntw_mean, Ntw_std, Ntw_sem, Ntw_tau = self.average(self.t, self.Ntw)
        #append averages
        if self.args.avgto is not None:
            with open(self.args.avgto, "a+") as f:
                f.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(self.mu, N_eff_samp, N_mean, N_std, N_sem, Ntw_eff_samp, Ntw_mean, Ntw_std, Ntw_sem, N_tau, Ntw_tau))

if __name__=="__main__":
    waters = IndusWaters()
    waters()
