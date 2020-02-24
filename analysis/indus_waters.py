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
            self.mu = mu

    def __call__(self):
        self.get_data()
        #log data
        self.save_timeseries(self.t,self.N,label="N")
        self.save_timeseries(self.t,self.N,label="Ntw")
        #plotting
        fig, ax = plt.subplots()
        ax.plot(self.t,self.N,label="$N$")
        ax.plot(self.t,self.Ntw,label="$Ntwiddle$")
        ax.set_xlabel("Time, in ps")
        ax.set_ylabel("$N_v$ and coarse-grained $N_v$")
        ax.legend()
        self.save_figure(fig,suffix="waters")
        if self.args.show:
            print("Displaying figure")
            plt.show()

if __name__=="__main__":
    waters = IndusWaters()
    waters()
