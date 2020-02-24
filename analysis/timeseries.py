"""Template for handling timeseries data produced by molecular simulations

@Author: Akash Pallath
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

#For time series analysis
import pymbar

class TimeSeries:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #parser arguments common to all timeseries classes
        #averaging
        self.parser.add_argument("-avgstart", help="time to start averaging at")
        self.parser.add_argument("-avgend", help="time to stop averaging at")
        self.parser.add_argument("-avgto", help="file to append averages to")
        #plotting
        self.parser.add_argument("-opref", help="output image and data prefix")
        self.parser.add_argument("-oformat", help="output image format")
        self.parser.add_argument("-dpi", type=int, help="dpi of output image(s)")
        self.parser.add_argument("--show", action='store_true', help="show interactive plot(s)")

    def read_args(self):
        self.args = self.parser.parse_args()

    def save_timeseries(self,t,x,label=""):
        ts = np.stack((t,x))
        np.save(self.args.opref+"_"+label,ts)

    def average(self,t,x,label=""):
        tstep = t[1] - t[0]
        [t0, g, Neff_max] = pymbar.timeseries.detectEquilibration(x)
        print("Equilibrated at {}".format(tstep * t0))

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
