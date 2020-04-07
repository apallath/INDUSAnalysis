"""Template for handling timeseries data produced by molecular simulations

@Author: Akash Pallath
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

#For time series analysis
import pymbar
from scipy import stats
from numpy import convolve

class TimeSeries:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #parser arguments common to all timeseries classes
        #averaging
        self.parser.add_argument("-avgstart", help="time to start averaging at")
        self.parser.add_argument("-avgend", help="time to stop averaging at")
        self.parser.add_argument("-avgto", help="file to append averages to")
        self.parser.add_argument("-window", help="window for moving average (default = 10)")
        #plotting
        self.parser.add_argument("-opref", help="output image and data prefix")
        self.parser.add_argument("-oformat", help="output image format")
        self.parser.add_argument("-dpi", type=int, help="dpi of output image(s)")
        self.parser.add_argument("--show", action='store_true', help="show interactive plot(s)")
        #for classes that choose to append to saved data from another run before plotting
        self.parser.add_argument("-apref", help="Append current quantities to previous quantities (from saved .npy files) and plot")
        self.parser.add_argument("-aprevlegend", help="String describing what the previous run currently being appended to is (for plot legend)")
        self.parser.add_argument("-acurlegend", help="String describing what the current run being appended is (for plot legend)")

    def read_args(self):
        self.args = self.parser.parse_args()

        #Common corner case handling
        self.aprevlegend = self.args.aprevlegend
        self.acurlegend = self.args.acurlegend
        if self.aprevlegend is None:
            aprevlegend = "Previous"
        if self.acurlegend is None:
            acurlegend = "Current"

    def save_timeseries(self,t,x,label=""):
        ts = np.stack((t,x))
        pref = self.args.opref
        if pref == None:
            pref = "data"
        np.save(pref+"_"+label,ts)
        print("Saving data > "+pref+"_"+label+".npy")

    """tests in tests/test_timeseries.py"""
    def moving_average(self,x,window):
        if window is None:
            window = np.float(10.0)
        else:
            window = np.float(window)
        w = np.repeat(1.0, window)/window
        ma = np.convolve(x, w, 'valid')
        return ma

    """tests in tests/test_timeseries.py"""
    def cumulative_moving_average(self,x):
        csum = np.cumsum(x)
        nvals = range(1,len(x)+1)
        cma = csum/nvals
        return cma

    """tests in tests/test_timeseries.py"""
    def average(self,t,x,avgstart,avgend):
        tstep = t[1] - t[0]
        start = 0
        end = len(x) - 1
        if avgstart is not None:
            start = int(np.floor(np.float(avgstart)/tstep))
        if avgend is not None:
            end = int(np.floor(np.float(avgend)/tstep))
        te = t[start:end]
        xe = x[start:end]
        tau_step = pymbar.timeseries.integratedAutocorrelationTime(xe)
        Neff = len(xe)/tau_step

        #Calculate std and sem
        mean = 0
        std = 0
        sem = 0
        for i in range(1000):
            #sample without replacement
            sidx = np.random.choice(np.array(range(len(xe))), size = int(Neff), replace=False)
            mean += np.mean(xe[sidx])
            std += np.std(xe[sidx])
            sem += stats.sem(xe[sidx])
        mean/= 1000
        std/= 1000
        sem/= 1000
        return Neff, mean, std, sem, tau_step

    def save_figure(self,fig,suffix=""):
        oformat = self.args.oformat
        pref = self.args.opref
        imgdpi= self.args.dpi
        if oformat == None:
            oformat = "ps"
        if pref == None:
            pref = "fig"

        filename = pref+"_"+suffix+"."+oformat

        if imgdpi is not None:
            fig.savefig(filename, dpi=imgdpi)
        else:
            fig.savefig(filename)

        print("Saving figure > "+filename)
