"""
Template for handling timeseries data produced by molecular simulations

@Author: Akash Pallath

FEATURE:    Replace autocorrelation analysis for errors with block averaging/block bootstrapping analysis
FEATURE:    Cythonize code
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

#For time series analysis
from scipy import stats
from numpy import convolve

#Pymbar for autocorrelation time
import pymbar.timeseries

class TimeSeries:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        #discard portion of data for analysis
        self.parser.add_argument("-obsstart", help="Timestep to begin computation of observables at")
        self.parser.add_argument("-obsend", help="Timestep to end computation of observables at")
        self.parser.add_argument("-obspref", help="Prefix of text file to append computed observables to")

        #window for sliding window averaging
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

    def parse_args(self, args=None):
        if args is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(args)

    def read_args(self):
        #parse into class variables
        self.obsstart = self.args.obsstart
        if self.obsstart is not None:
            self.obsstart = np.float(self.obsstart)

        self.obsend = self.args.obsend
        if self.obsend is not None:
            self.obsend = np.float(self.obsend)

        self.obspref = self.args.obspref

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
            self.replotpref = "fig"

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

    """
    Compute moving average of time series data using a sliding window defined
    in units of time

    tests in tests/test_timeseries.py"""
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

    """
    Compute cumulative moving average (running average) of time series data

    tests in tests/test_timeseries.py"""
    def cumulative_moving_average(self,x):
        csum = np.cumsum(x)
        nvals = range(1,len(x)+1)
        cma = csum/nvals
        return cma


    """
    Compute mean of timeseries over range [obsstart, obsend]
    """
    def ts_mean(self,t,x,obsstart,obsend):
        t = np.reshape(t, (len(t),))
        tstep = t[1] - t[0]

        if obsstart is None:
            startidx = 0
        else:
            startidx = int(np.floor(obsstart/tstep))

        if obsend is None:
            endidx = len(x)
        else:
            endidx = int(np.floor(obsend/tstep))

        xsub = x[startidx:endidx]
        return np.mean(xsub)

    """
    Compute std of timeseries over range [obsstart, obsend]
    """
    def ts_std(self,t,x,obsstart,obsend):
        t = np.reshape(t, (len(t),))
        tstep = t[1] - t[0]

        if obsstart is None:
            startidx = 0
        else:
            startidx = int(np.floor(obsstart/tstep))

        if obsend is None:
            endidx = len(x)
        else:
            endidx = int(np.floor(obsend/tstep))

        xsub = x[startidx:endidx]
        return np.std(xsub)

    """
    Compute standard error of mean (estimate of the standard deviation of the
    distribution of the mean) for the time series data using autocorrelation
    analysis for estimating number of independent samples

    tests in tests/test_timeseries.py
    """
    def serr_mean(self,t,x,obsstart,obsend):
        t = np.reshape(t, (len(t),))
        tstep = t[1] - t[0]

        if obsstart is None:
            startidx = 0
        else:
            startidx = int(np.floor(obsstart/tstep))

        if obsend is None:
            endidx = len(x)
        else:
            endidx = int(np.floor(obsend/tstep))

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

        return serr

    """
    Helper function to save timeseries data to file
    """
    def save_timeseries(self,t,x,label=""):
        ts = np.stack((t,x))
        pref = self.opref
        if pref == None:
            pref = "data"
        np.save(pref+"_"+label,ts)
        print("Saving data > "+pref+"_"+label+".npy")


    """
    Helper function to save figures to file
    """
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
