"""
Defines classes for storing and analysing timeseries data.
"""
import argparse
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import convolve
from scipy import stats

from INDUSAnalysis.lib import profiling

# Cython
cimport numpy as np


class TimeSeries:
    """Stores N-dimensional time series data, accessible and sliceable by time.

    Attributes:
        times (ndarray): 1-dimensional array of length N.
        data (ndarray): D-dimensional array of shape (N, ...).
        labels (list): List of strings describing what each dimension represents.
        correct_contiguous (boolean): Ensure that for elements (i, i+1), the time at i+1 > time at i.
            If a pair is found such that the time at i+1 <= time at i, delete the preceding timeseries'
            data which is repeated. Useful for correcting INDUS outputs on restart from checkpoint. (Default=True)

    Examples:
        >>> ts = TimeSeries([0, 100, 200], [10, 20, 10], ["Sample data"])
        >>> ts
        <TimeSeries object, ['Sample data'] with shape (3,), 3 time frames>
        >>> len(ts)
        3
        >>> ts.time_array
        array([  0, 100, 200])
        >>> ts.data_array
        array([10, 20, 10])
        >>> ts.labels
        ['Sample data']
        >>> ts[100:]
        <TimeSeries object, ['Sample data'] with shape (2,), 2 time frames>
        >>> ts[100:].time_array
        array([100, 200])
        >>> ts[100:150]
        <TimeSeries object, ['Sample data'] with shape (1,), 1 time frames>
        >>> ts[100:150].time_array
        array([100])
        >>> ts[::2].time_array
        array([  0, 200])
        >>> ts.time_array = [0, 100, 200, 300, 400, 500]
        >>> ts.time_array
        array([  0, 100, 200, 300, 400, 500])
        >>> ts.data_array = [10, 20, 20, 20, 10, 10]
        >>> ts
        <TimeSeries object, ['Sample data'] with shape (6,), 6 time frames>
    """

    def __init__(self, times, data, labels, correct_contiguous=True):
        """
        Creates time series class.

        Raises:
            ValueError if time and data lengths do not match, or if length of
            labels does not equal number of dimensions of data.
        """
        self._t = np.array(times)
        self._x = np.array(data)
        if self._t.shape[0] != self._x.shape[0]:
            raise ValueError("Time and data do not match along axis 0")

        self._labels = labels
        if len(self._labels) < self._x.ndim:
            raise ValueError("Too few labels for data dimensions")
        if len(self._labels) > self._x.ndim:
            raise ValueError("Too many labels for data dimensions")
        if correct_contiguous:
            self._correct_contiguous()


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

    # TODO: Add unit tests for setters
    @property
    def time_array(self):
        return self._t

    @time_array.setter
    def time_array(self, t):
        self._t = np.array(t)

    @property
    def data_array(self):
        return self._x

    @data_array.setter
    def data_array(self, x):
        if x.shape[0] != len(self._t):
            raise ValueError("Time and data do not match along axis 0.")
        self._x = np.array(x)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if len(labels) < self._x.ndim:
            raise ValueError("Too few labels for data dimensions")
        if len(labels) > self._x.ndim:
            raise ValueError("Too many labels for data dimensions")
        self._labels = labels

    def _correct_contiguous(self):
        delete_indices = []

        last_lookup = 0
        for tidx in range(len(self)):
            # If discrepancy
            if self._t[tidx] <= self._t[tidx - 1]:
                # Find (from left) value equal to or just greater than this entry
                start_del_slice = 0
                for prev_tidx in range(last_lookup, tidx):
                    if self._t[prev_tidx] >= self._t[tidx]:
                        start_del_slice = prev_tidx
                        break
                # Add indices starting from this index to the index just before tidx to the list of indices to be deleted
                delete_indices.extend(list(range(start_del_slice, tidx)))
                # Ensure that future searches start from this index
                last_lookup = tidx

        new_t = np.delete(self._t, list(set(delete_indices)), axis=0)
        new_x = np.delete(self._x, list(set(delete_indices)), axis=0)

        self._t = new_t
        self._x = new_x

        if self._t.shape[0] != self._x.shape[0]:
            raise ValueError("Time and data do not match along axis 0")


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
        ma = np.convolve(self._x, w, 'valid')
        return TimeSeries(self._t[window - 1:], ma, labels=self._labels)

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
            axis (int): Optional.

        Returns:
            np.float if mean is computed for flattened data (axis = None).
            np.array if mean is computed along axis 0.
            TimeSeries object if mean is computed along axis > 0.
        """
        if axis is None or axis == 0:
            return np.mean(self._x, axis=axis)
        else:
            mean = np.mean(self._x, axis=axis)
            labels = [x for i, x in enumerate(self._labels) if i != axis]
            return TimeSeries(self._t, mean, labels=labels)

    def std(self, axis=None):
        """
        Computes std of timeseries data.

        Args:
            axis (int): Optional.

        Returns:
            np.float if mean is computed for flattened data (axis = None).
            np.array if mean is computed along axis 0.
            TimeSeries object if mean is computed along axis > 0.
        """
        if axis is None or axis == 0:
            return np.std(self._x, axis=axis)
        else:
            std = np.std(self._x, axis=axis)
            labels = [x for i, x in enumerate(self._labels) if i != axis]
            return TimeSeries(self._t, std, labels=labels)

    # TODO: Implement
    @profiling.timefunc
    def sem(self, axis=None):
        """
        Computes standard error of mean (estimate of the standard deviation
        of the distribution of the mean) using block bootstrapping.

        Args:
            axis (int): Optional
        """
        raise NotImplementedError()

    def plot(self, *plotargs, **plotkwargs):
        """Plots 1-d timeseries data.

        Args:
            *plotargs: Arguments for Matplotlib's plot function.
            **plotkwargs: Keyword arguments for Matplotlib's plot function.

        Returns:
            Matplotlib figure object containing plotted data.

        Raises:
            ValueError if data is not 1-dimensional.
        """
        fig, ax = plt.subplots()
        ax.plot(self._t, self._x, *plotargs, **plotkwargs)
        ax.set_xlabel("Time")
        ax.set_ylabel(self._labels[0])
        return fig

    def plot_2d_heatmap(self, **imshowkwargs):
        """Plots 2-d timeseries data as a heatmap.

        Args:
            *imshowkwargs: Keyword arguments for Matplotlib's imshow function.

        Returns:
            Matplotlib figure object containing plotted data.

        Raises:
            ValueError if data is not 2-dimensional.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self._x, origin='lower', aspect='auto', **imshowkwargs)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(self._labels[0])
        ax.set_xlabel(self._labels[1])
        ax.set_ylabel('Time')

        # Ticks
        ticks = ax.get_yticks().tolist()
        start = self._t[0]
        sep = (self._t[-1] - self._t[0]) / (ticks[-2] - ticks[1])
        newlabels = [(start + sep * axval) for axval in ticks]
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
        self.calc_args = self.parser.add_argument_group('calculation arguments')
        self.avg_args = self.parser.add_argument_group('averaging optional arguments')
        self.out_args = self.parser.add_argument_group('output control arguments')
        self.replot_args = self.parser.add_argument_group('replot control arguments')
        self.misc_args = self.parser.add_argument_group('miscellanious arguments')

        # Averaging options
        self.avg_args.add_argument("-obsstart", help="Time to begin computation of observables at")
        self.avg_args.add_argument("-obsend", help="Time to end computation of observables at")
        self.avg_args.add_argument("-obspref", help="Prefix of text file to append computed observables to")
        self.avg_args.add_argument("-window", help="Window size (number of points) for moving average")

        # Plotting options
        self.out_args.add_argument("-opref", help="Output image and data prefix [Default = indus]")
        self.out_args.add_argument("-oformat", help="Output image format [Default = png]")
        self.out_args.add_argument("-dpi", type=int, help="DPI of output image(s) [Default = 150]")
        self.out_args.add_argument("--show", action='store_true', help="Show interactive plot(s)")

        # Replot options
        self.replot_args.add_argument("--replot", action="store_true", help="Replot from saved data")
        self.replot_args.add_argument("-replotpref", help="[replot] Prefix (REPLOTPREF[.npy]) of .npy file to replot from [Default = indus]")

        # Miscellanious options
        self.misc_args.add_argument("--remote", action='store_true', help="Run with text-only backend on remote cluster")

    @classmethod
    def save_TimeSeries(cls, tso, filename):
        """
        Saves TimeSeries object to file using pickle dump.

        Args:
            tso (TimeSeries): TimeSeries object to pickle and dump.
        """
        with open(filename, 'wb+') as f:
            pickle.dump(tso, f)

    @classmethod
    def load_TimeSeries(cls, filename):
        """
        Loads pickled TimeSeries object from file

        Args:
            filename: Name of file to load pickled TimeSeries object from

        Returns:
            TimeSeries object loaded from file
        """
        with open(filename, 'rb') as f:
            tso = pickle.load(f)
            return tso

    def parse_args(self, args=None):
        """
        Parses arguments.

        Args:
            args (list): Arguments to parse, optional.
        """
        if args is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(args)

    def read_args(self):
        """
        Stores arguments from TimeSeries `args` parameter in class variables.
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
            self.window = np.int(self.window)

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
