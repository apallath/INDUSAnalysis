import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from INDUSAnalysis import timeseries


def test_TimeSeries_1d_basic():
    t = [10, 20, 30, 40, 50, 60, 70, 80]
    t = np.array(t)
    x = [2, 4, 6, 8, 10, 10, 10, 30]
    x = np.array(x)
    ts = timeseries.TimeSeries(t, x, ['RMSD'])

    # Test getter methods
    assert(np.allclose(ts.time_array, t))
    assert(np.allclose(ts.data_array, x))

    # Test indexing
    assert(np.allclose(ts[:50].data_array, np.array([2, 4, 6, 8, 10])))
    assert(np.allclose(ts[10:30].data_array, np.array([2, 4, 6])))

    # Test mean
    assert(np.isclose(ts.mean(), 10.0))
    assert(np.isclose(ts[10:40].mean(), 5))
    assert(np.isclose(ts[50:70].mean(), 10))

    # Test std
    assert(np.isclose(ts[10:20].std(), 1.0))
    assert(np.isclose(ts[50:70].std(), 0.0))


def test_TimeSeries_1d_vis():
    pass


def test_TimeSeries_2d_basic():
    pass


def test_TimeSeries_2d_vis():
    pass


def test_TimeSeries_3d_basic():
    pass

# TODO: Write tests for standard error of mean


def test_TimeSeriesAnalysis_argparse():
    pass


def test_TimeSeriesAnalysis_savefig():
    pass
