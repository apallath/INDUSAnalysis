"""Template for handling timeseries data produced by molecular simulations

@Author: Akash Pallath

TODO:   Replace statistical inefficiency analysis for standard errors with block
        bootstrapping
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

#For time series analysis
from scipy import stats
from numpy import convolve

#Temporary use of pymbar
import pymbar.timeseries

class TimeSeries:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #parser arguments common to all timeseries classes
        #averaging
        self.parser.add_argument("-avgstart", help="Time to start analysis at")
        self.parser.add_argument("-avgend", help="Time to stop analysis at")
        self.parser.add_argument("-avgto", help="File to append averages to")
        self.parser.add_argument("-window", help="Time window for moving average (default = 1 time unit)")
        #plotting
        self.parser.add_argument("-opref", help="Output image and data prefix")
        self.parser.add_argument("-oformat", help="Output image format")
        self.parser.add_argument("-dpi", type=int, help="DPI of output image(s)")
        self.parser.add_argument("--show", action='store_true', help="Show interactive plot(s)")
        #replot arguments, for classes that choose to implement them
        self.parser.add_argument("--replot", action="store_true", help="Replot from saved data")
        self.parser.add_argument("-replotpref", help="Prefix (pref.npy) of data file to replot from")
        #for classes that choose to append to saved data from another run before plotting
        self.parser.add_argument("-apref", \
            help="Append current quantities to previous quantities (from saved .npy files) and plot")
        self.parser.add_argument("-aprevlegend", \
            help="String describing what the previous run currently being appended to is (for plot legend)")
        self.parser.add_argument("-acurlegend", \
            help="String describing what the current run being appended is (for plot legend)")
        #for running on remote server
        self.parser.add_argument("--remote", action='store_true')

    def read_args(self):
        self.args = self.parser.parse_args()
        #parse into class variables
        self.avgstart = self.args.avgstart
        if self.avgstart is not None:
            self.avgstart = np.float(self.avgstart)

        self.avgend = self.args.avgend
        if self.avgend is not None:
            self.avgend = np.float(self.avgend)

        self.avgto = self.args.avgto
        self.window = self.args.window
        if self.window is not None:
            self.window = np.float(self.window)

        self.opref = self.args.opref
        self.oformat = self.args.oformat
        self.dpi = self.args.dpi
        self.show = self.args.show

        self.replot = self.args.replot
        self.replotpref = self.args.replotpref
        if self.replotpref is None:
            self.replotpref = "data"

        self.apref = self.args.apref
        self.aprevlegend = self.args.aprevlegend
        if self.aprevlegend is None:
            aprevlegend = "Previous"
        self.acurlegend = self.args.acurlegend
        if self.acurlegend is None:
            acurlegend = "Current"

        # Force matplotlib to not use any Xwindows backend if run on remote server
        self.remote = self.args.remote
        if self.remote:
            matplotlib.use('Agg')

    """tests in tests/test_timeseries.py"""
    def moving_average(self,t,x,window):
        t = np.reshape(t, (len(t),))
        tstep = t[1] - t[0]

        if window is None:
            window = int(np.floor(1.0/tstep))
        else:
            window = int(np.floor(window/tstep))
        w = np.repeat(1.0, window)/window
        ma = np.convolve(x, w, 'valid')
        return ma

    """tests in tests/test_timeseries.py"""
    def cumulative_moving_average(self,x):
        csum = np.cumsum(x)
        nvals = range(1,len(x)+1)
        cma = csum/nvals
        return cma

    """TODO
       tests in tests/test_timeseries.py"""
    def average(self,t,x,avgstart,avgend):
        t = np.reshape(t, (len(t),))
        tstep = t[1] - t[0]

        if avgstart is None:
            startidx = 0
        else:
            startidx = int(np.floor(avgstart/tstep))

        if avgend is None:
            endidx = len(x)
        else:
            endidx = int(np.floor(avgend/tstep))

        xsub = x[startidx:endidx]

        #mean
        mean = np.mean(xsub)

        #standard error
        """
        References:
        #[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2671659/
        #[2] http://www.hep.fsu.edu/~berg/teach/mcmc08/material/lecture07mcmc3.pdf
        """
        stat_ineff = pymbar.timeseries.statisticalInefficiency(xsub, fft=True)
        actual_samples = len(xsub)/stat_ineff
        serr = np.std(xsub)/np.sqrt(actual_samples)

        #95% CI
        ci_95_low = mean - 1.96*serr
        ci_95_high = mean + 1.96*serr

        return mean, serr, ci_95_low, ci_95_high

    def save_timeseries(self,t,x,label=""):
        ts = np.stack((t,x))
        pref = self.opref
        if pref == None:
            pref = "data"
        np.save(pref+"_"+label,ts)
        print("Saving data > "+pref+"_"+label+".npy")

    def save_figure(self,fig,suffix=""):
        oformat = self.oformat
        pref = self.opref
        imgdpi= self.dpi
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
