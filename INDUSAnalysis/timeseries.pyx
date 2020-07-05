"""
Classes for storing and analysing timeseries data.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import argparse

from scipy import stats
from numpy import convolve

import pymbar.timeseries

from INDUSAnalysis.lib import profiling

# Cython
cimport numpy as np


class TimeSeries:
    """Stores N-dimensional time series data.

    Attributes:
        times (ndarray): 1-dimensional array of length N.
        data (ndarray): D-dimensional array of shape (N, ...).
        labels (list): List of strings describing what each dimension represents.
    """

    def __init__(self, times, data, labels):
        """
        Creates time series class.

        Raises:
            ValueError if time and data lengths do not match, or if length of
            labels does not equal number of dimensions of data.
        """
        self._t = times
        self._x = data
        if self._t.shape[0] != self._x.shape[0]:
            raise ValueError("Time and data do not match along axis 0")

        self._labels = labels
        if len(self._labels) < self._x.ndim:
            raise ValueError("Too few labels for data dimensions")
        if len(self._labels) > self._x.ndim:
            raise ValueError("Too many labels for data dimensions")

    @property
    def time_array(self):
        return self._t

    @property
    def data_array(self):
        return self._x

    def __getitem__(self, key):
        """
        Performs indexing/slicing based on time values,
        [start-time:end-time:resample-freq].

        Returns:
            Sliced and resampled TimeSeries object.
        """
        if isinstance(key, int):
            idx = np.where(self._t == key)
            return TimeSeries(self._t[idx], self._x[idx], labels=self._labels)

        if isinstance(key, slice):
            sidx = key.start
            if sidx is not None:
                sidx = np.searchsorted(self._t, key.start, side='left')

            eidx = key.stop
            if eidx is not None:
                eidx = np.searchsorted(self._t, key.stop, side='right')

            return TimeSeries(self._t[sidx:eidx:key.step],
                              self._x[sidx:eidx:key.step], labels=self._labels)

    def __len__(self):
        return len(self._t)

    def __repr__(self):
        return "<{} object, {} with shape {}, {} time frames>".format(
            self.__class__.__name__, self._labels, self._x.shape, len(self._t))

    def moving_average(self, window):
        """
        Computes moving (rolling) average of 1-d data.

        Args:
            window (int): Number of data points to smooth over.

        Raises:
            ValueError if data is not 1-dimensional.
        """
        if self._x.ndim != 1:
            raise ValueError("Data is not 1-dimensional")

        w = np.repeat(1.0, window) / window
        ma = np.convolve(self._x, w, 'same')
        return TimeSeries(self._t, ma, labels=self._labels)

    def cumulative_moving_average(self):
        """
        Computes cumulative moving average of 1-d data.

        Raises:
            ValueError if data is not 1-dimensional.
        """
        if self._x.ndim != 1:
            raise ValueError("Data is not 1-dimensional")

        csum = np.cumsum(self._x)
        nvals = range(1, len(self) + 1)
        cma = csum / nvals
        return TimeSeries(self._t, cma, labels=self._labels)

    def mean(self, axis=None):
        """
        Computes mean of timeseries data.

        Args:
            axis (int): Optional, ignored for 1-d data.
        """
        if axis is None or self._x.ndim == 1:
            return np.mean(self._x)
        else:
            mean = np.mean(self._x, axis=axis)
            labels = [x for i, x in enumerate(self._labels) if i != axis]
            return TimeSeries(self._t, mean, labels=labels)

    def std(self, axis=None):
        """
        Computes std of timeseries data.

        Args:
            axis (int): Optional, ignored for 1-d data.
        """
        if axis is None or self._x.ndim == 1:
            return np.std(self._x)
        else:
            std = np.std(self._x, axis=axis)
            labels = [x for i, x in enumerate(self._labels) if i != axis]
            return TimeSeries(self._t, std, labels=labels)

    # TODO: Implement
    def sem(self, axis=0):
        """
        Computes standard error of mean (estimate of the standard deviation
        of the distribution of the mean) using block bootstrapping.

        Args:
            axis (int)
        """
        raise NotImplementedError

    # TODO: Figure options
    def plot(self):
        """Plots 1-d timeseries data.

        Returns:
            Matplotlib figure object containing plotted data.

        Raises:
            ValueError if data is not 1-dimensional.
        """
        fig, ax = plt.subplots()
        ax.plot(self._t, self._x)
        ax.set_xlabel("Time")
        ax.set_ylabel(self.labels[0])
        return fig

    # TODO: Figure options
    def plot_2d_heatmap(self):
        """Plots 2-d timeseries data as a heatmap.

        Returns:
            Matplotlib figure object containing plotted data.

        Raises:
            ValueError if data is not 2-dimensional.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self._x, origin="lower", cmap="hot", aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel(self.labels[1])

        # Ticks
        ticks = ax.get_yticks().tolist()
        factor = self._t[-1] / ticks[-2]
        newlabels = [factor * item for item in ticks]
        ax.set_yticklabels(newlabels)

        return fig


class TimeSeriesAnalysis:
    """
    Template class for timeseries analysis of simulation data.

    Attributes:
        args: Command line argument parser.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Argument groups
        self.req_file_args = self.parser.add_argument_group('required filename arguments')
        self.opt_file_args = self.parser.add_argument_group('optional filename arguments')
        self.avg_args = self.parser.add_argument_group('averaging optional arguments')
        self.plot_args = self.parser.add_argument_group('plotting optional arguments')
        self.replot_args = self.parser.add_argument_group('replot optional arguments')
        self.misc_args = self.parser.add_argument_group('miscellanious optional arguments')

        # Averaging options
        self.avg_args.add_argument("-obsstart", help="Time to begin computation of observables at")
        self.avg_args.add_argument("-obsend", help="Time to end computation of observables at")
        self.avg_args.add_argument("-obspref", help="Prefix of text file to append computed observables to")
        self.avg_args.add_argument("-window", help="Window size (number of points) for moving average")

        # Plotting options
        self.plot_args.add_argument("-opref", help="Output image and data prefix [Default = indus]")
        self.plot_args.add_argument("-oformat", help="Output image format [Default = png]")
        self.plot_args.add_argument("-dpi", type=int, help="DPI of output image(s) [Default = 150]")
        self.plot_args.add_argument("--show", action='store_true', help="Show interactive plot(s)")

        # Replot options
        self.replot_args.add_argument("--replot", action="store_true", help="Replot from saved data")
        self.replot_args.add_argument("-replotpref", help="[replot] Prefix (REPLOTPREF[.npy]) of .npy file to replot from [Default = indus]")

        # Miscellanious optinos
        self.misc_args.add_argument("--remote", action='store_true', help="Run with text-only backend on remote cluster")

    def parse_args(self, args=None):
        """
        Parses arguments

        Args:
            args (list): Arguments to parse, optional
        """
        if args is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(args)

    def read_args(self):
        """
        Stores arguments from TimeSeries `args` parameter in class variables

        Raises:
            ValueError if arguments parsed are inconsistent
        """
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
            self.opref = "indus"

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
            self.replotpref = "indus"

        # Forces matplotlib to not use any Xwindows backend if run on remote server
        self.remote = self.args.remote
        if self.remote:
            matplotlib.use('Agg')

    # TODO: Implement
    def save_TimeSeries(self, tso):
        """
        Saves TimeSeries object to file using pickle dump.

        Args:
            tso (TimeSeries): TimeSeries object to pickle and dump.
        """
        raise NotImplementedError

    def save_figure(self, fig, suffix=""):
        """
        Exports figure to image file.

        The figure is exported to an image file OPREF_SUFFIX.OFORMAT,
        where OPREF is the class attribute containing the output prefix and
        OFORMAT is the class attribute storing the output image extension.

        Args:
            fig: Matplotlib figure.
            suffix: Suffix of filename to save data to.
        """
        filename = self.opref + "_" + suffix + "." + self.oformat
        if self.dpi is not None:
            fig.savefig(filename, dpi=self.dpi)
        else:
            fig.savefig(filename)
        print("Saving figure > " + filename)
