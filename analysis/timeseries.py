"""Template for handling timeseries data produced by molecular simulations

@Author: Akash Pallath

TODO: Block bootstrapping for averaging

"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

#For time series analysis
from scipy import stats
from numpy import convolve

class TimeSeries:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #parser arguments common to all timeseries classes
        #averaging
        self.parser.add_argument("-avgstart", help="time to start analysis at")
        self.parser.add_argument("-avgend", help="time to stop analysis at")
        self.parser.add_argument("-avgto", help="file to append averages to")
        self.parser.add_argument("-window", help="window for moving average (default = 10)")
        #plotting
        self.parser.add_argument("-opref", help="output image and data prefix")
        self.parser.add_argument("-oformat", help="output image format")
        self.parser.add_argument("-dpi", type=int, help="dpi of output image(s)")
        self.parser.add_argument("--show", action='store_true', help="show interactive plot(s)")
        #for classes that choose to append to saved data from another run before plotting
        self.parser.add_argument("-apref", \
            help="Append current quantities to previous quantities (from saved .npy files) and plot")
        self.parser.add_argument("-aprevlegend", \
            help="String describing what the previous run currently being appended to is (for plot legend)")
        self.parser.add_argument("-acurlegend", \
            help="String describing what the current run being appended is (for plot legend)")

    def read_args(self):
        self.args = self.parser.parse_args()
        #parse into class variables
        self.avgstart = self.args.avgstart
        self.avgend = self.args.avgend
        self.avgto = self.args.avgto
        self.window = self.args.window
        self.opref = self.args.opref
        self.oformat = self.args.oformat
        self.dpi = self.args.dpi
        self.show = self.args.show
        self.apref = self.args.apref
        self.aprevlegend = self.args.aprevlegend
        if self.aprevlegend is None:
            aprevlegend = "Previous"
        self.acurlegend = self.args.acurlegend
        if self.acurlegend is None:
            acurlegend = "Current"

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

    """TODO
       tests in tests/test_timeseries.py"""
    def average(self,t,x,avgstart,avgend):
        """Compute averages for correlated time series data"""
        pass

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
