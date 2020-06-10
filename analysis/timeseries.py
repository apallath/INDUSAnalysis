"""
Template for handling timeseries data produced by molecular simulations

@Author: Akash Pallath
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

#Profiling
from meta_analysis.profiling import timefunc

class TimeSeries:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Averaging options
        # Analysis region
        self.parser.add_argument("-obsstart", help="Timestep to begin computation of observables at")
        self.parser.add_argument("-obsend", help="Timestep to end computation of observables at")
        self.parser.add_argument("-obspref", help="Prefix of text file to append computed observables to")
        # Sliding window
        self.parser.add_argument("-window", help="Time window for moving average (default = 1 time unit)")

        # Plotting options
        self.parser.add_argument("-opref", help="Output image and data prefix")
        self.parser.add_argument("-oformat", help="Output image format")
        self.parser.add_argument("-dpi", type=int, help="DPI of output image(s)")
        self.parser.add_argument("--show", action='store_true', help="Show interactive plot(s)")

        self.parser.add_argument("--replot", action="store_true", help="Replot from saved data")
        self.parser.add_argument("-replotpref", help="[replot] Prefix (REPLOTPREF[.npy]) of .npy file to replot from")

        # Remote cluster run modifications
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
        if self.opref is None:
            self.opref = "analysis"

        self.oformat = self.args.oformat
        if self.oformat is None:
            self.oformat = "png"

        self.dpi = self.args.dpi
        if self.dpi is not None:
            self.dpi = int(self.dpi)
        else:
            self.dpi = 150

        self.show = self.args.show

        self.replot = self.args.replot
        self.replotpref = self.args.replotpref
        if self.replotpref is None:
            self.replotpref = "analysis"

        # Force matplotlib to not use any Xwindows backend if run on remote server
        self.remote = self.args.remote
        if self.remote:
            matplotlib.use('Agg')

    """
    Compute moving average of time series data using a sliding window defined
    in units of time

    tests in tests/test_timeseries.py"""
    def moving_average(self, t, x, window):
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
    def cumulative_moving_average(self, x):
        csum = np.cumsum(x)
        nvals = range(1,len(x)+1)
        cma = csum/nvals
        return cma

    """
    Compute mean of timeseries over range [obsstart, obsend]
    """
    def ts_mean(self, t, x, obsstart, obsend):
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
    def ts_std(self, t, x, obsstart, obsend):
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
    @timefunc
    def serr_mean(self, t, x, obsstart, obsend):
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
    def save_timeseries(self, t, x, label=""):
        ts = np.stack((t,x))
        np.save(self.opref+"_"+label,ts)
        print("Saving data > "+self.opref+"_"+label+".npy")

    """
    Helper function to save figures to file
    """
    def save_figure(self, fig, suffix=""):
        filename = self.opref+"_"+suffix+"."+self.oformat
        if self.dpi is not None:
            fig.savefig(filename, dpi=self.dpi)
        else:
            fig.savefig(filename)
        print("Saving figure > "+filename)
