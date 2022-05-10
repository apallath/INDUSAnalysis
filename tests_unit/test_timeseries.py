import numpy as np
import matplotlib.pyplot as plt
import pymbar

from INDUSAnalysis import timeseries


def test_TimeSeries_1d_basic():
    """Tests getters, slicing, mean and std for 1-d data"""
    t = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    x = np.array([2, 4, 6, 8, 10, 10, 10, 30])
    ts = timeseries.TimeSeries(t, x, ['RMSD'])

    # Test getter methods
    assert(np.allclose(ts.time_array, t))
    assert(np.allclose(ts.data_array, x))

    # Test slicing
    assert(np.allclose(ts[:50].data_array, np.array([2, 4, 6, 8, 10])))
    assert(np.allclose(ts[10:30].data_array, np.array([2, 4, 6])))
    assert(np.allclose(ts[60:].data_array, np.array([10, 10, 30])))
    assert(np.allclose(ts[::2].data_array, np.array([2, 6, 10, 10])))
    assert(np.allclose(ts[20::3].data_array, np.array([4, 10, 30])))

    # Test mean
    assert(np.isclose(ts.mean(), 10.0))
    assert(np.isclose(ts[10:40].mean(), 5))
    assert(np.isclose(ts[50:70].mean(), 10))

    # Test std
    assert(np.isclose(ts[10:20].std(), 1.0))
    assert(np.isclose(ts[50:70].std(), 0.0))


def test_TimeSeries_contiguous_correct_1D():
    """Checks correction of erroneous 1-D timeseries from a process which restarts from a previous checkpoint"""
    t = np.array([0, 10, 20, 30, 40, 50, 20, 30, 40, 50, 40, 50, 50])
    x = np.array([1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    ts = timeseries.TimeSeries(t, x, ['Test'])

    # Check
    assert(np.allclose(ts.time_array, np.array([0, 10, 20, 30, 40, 50])))
    assert(np.allclose(ts.data_array, np.array([1, 11, 16, 17, 20, 22])))


def test_TimeSeries_contiguous_correct_2D():
    """Checks correction of erroneous 2-D timeseries from a process which restarts from a previous checkpoint (to check array removal in multi-D)"""
    t = np.array([0, 10, 20, 30, 40, 50, 20, 30, 40, 50, 40, 50, 50])
    x = np.array([[0, 1],
                  [1, 1],
                  [1, 2],
                  [1, 3],
                  [1, 4],
                  [1, 5],
                  [1, 6],
                  [1, 7],
                  [1, 8],
                  [1, 9],
                  [2, 0],
                  [2, 1],
                  [2, 2]])
    ts = timeseries.TimeSeries(t, x, ['Test 1', 'Test 2'])

    # Check
    assert(np.allclose(ts.time_array, np.array([0, 10, 20, 30, 40, 50])))
    assert(np.allclose(ts.data_array, np.array([[0, 1], [1, 1], [1, 6], [1, 7], [2, 0], [2, 2]])))


def test_TimeSeries_1d_ma_cma():
    """Tests moving average and cumulative moving average for 1-d data"""
    t = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    x = np.array([2, 1, 2, 1, 2, 1, 2, 1])
    ts = timeseries.TimeSeries(t, x, ['RMSD'])

    # Test moving average
    assert(np.allclose(ts.moving_average(window=2).data_array, np.array(7 * [1.5])))
    assert(np.allclose(ts.moving_average(window=4).data_array, np.array(5 * [3 / 2])))
    # Test cumulative moving average
    assert(np.allclose(ts.cumulative_moving_average().data_array,
                       np.array([2, 3 / 2, 5 / 3, 6 / 4, 8 / 5, 9 / 6, 11 / 7, 12 / 8])))


def test_TimeSeries_1d_vis():
    """Tests plotting for 1-d data"""
    t = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    x = np.array([2, 4, 6, 8, 10, 10, 10, 30])
    ts = timeseries.TimeSeries(t, x, ['RMSD'])

    fig = ts.plot('s--', lw=1.2, label='Backbone-Backbone RMSD')
    fig.set_dpi(300)
    ax = fig.gca()
    ax.legend()
    fig.savefig("timeseries_test_data/1d.png")


def test_TimeSeries_2d_basic():
    """Tests getters, slicing, mean and std for 2-d data"""
    t = np.array([10, 20, 30, 40, 50])
    x = np.array([[2, 10, 0], [4, 10, 0], [6, 10, 5], [8, 10, 5], [10, 10, 10]])
    ts = timeseries.TimeSeries(t, x, ['Atomic deviations', 'Atom index'])

    # Test getter methods
    assert(np.allclose(ts.time_array, t))
    assert(np.allclose(ts.data_array, x))

    # Test slicing
    assert(np.allclose(ts[40:].data_array, np.array([[8, 10, 5], [10, 10, 10]])))
    assert(np.allclose(ts[20:30].data_array, np.array([[4, 10, 0], [6, 10, 5]])))
    assert(np.allclose(ts[:20].data_array, np.array([[2, 10, 0], [4, 10, 0]])))
    assert(np.allclose(ts[::2].data_array,
                       np.array([[2, 10, 0], [6, 10, 5], [10, 10, 10]])))

    # Test mean
    assert(np.isclose(ts.mean(), x.mean()))
    assert(np.allclose(ts.mean(axis=0), x.mean(axis=0)))
    assert(np.allclose(ts.mean(axis=1).data_array, x.mean(axis=1)))

    # Test std
    assert(np.isclose(ts.std(), x.std()))
    assert(np.allclose(ts.std(axis=0), x.std(axis=0)))
    assert(np.allclose(ts.std(axis=1).data_array, x.std(axis=1)))


def test_TimeSeries_2d_vis():
    """Tests heatmap visualization for 2-d data"""
    # Test with time starting from 0
    t = np.array([0, 10, 20, 30, 40, 50])
    x = np.array([[0, 10, 0], [2, 10, 0], [4, 10, 0], [6, 10, 5], [8, 10, 5], [10, 10, 10]])
    ts = timeseries.TimeSeries(t, x, ['Atomic deviations', 'Atom index'])

    fig = ts.plot_2d_heatmap(cmap='cool')
    fig.set_dpi(300)
    fig.savefig("timeseries_test_data/2d_0_index.png")

    # Test with time starting from 10
    t = np.array([10, 20, 30, 40, 50])
    x = np.array([[2, 10, 0], [4, 10, 0], [6, 10, 5], [8, 10, 5], [10, 10, 10]])
    ts = timeseries.TimeSeries(t, x, ['Atomic deviations', 'Atom index'])

    fig = ts.plot_2d_heatmap(cmap='cool')
    fig.set_dpi(300)
    fig.savefig("timeseries_test_data/2d_10_index.png")

    # Test with larger data
    t = np.linspace(0, 10000, 41)
    x = np.random.random((41, 100))
    ts = timeseries.TimeSeries(t, x, ['Atomic deviations', 'Atom index'])

    fig = ts.plot_2d_heatmap(cmap='hot')
    fig.set_dpi(300)
    fig.savefig("timeseries_test_data/2d_large.png")

    # Test with sliced larger data
    t = np.linspace(0, 10000, 41)
    x = np.random.random((41, 100))
    ts = timeseries.TimeSeries(t, x, ['Atomic deviations', 'Atom index'])
    ts_slice = ts[5000:]
    fig = ts_slice.plot_2d_heatmap(cmap='hot')
    fig.set_dpi(300)
    fig.savefig("timeseries_test_data/2d_large_slice.png")


def test_TimeSeries_3d_dimred_vis():
    t = np.linspace(0, 10000, 41)
    x = np.random.random((41, 100, 100))
    x_T = np.transpose(x, axes=(0, 2, 1))
    x_sym = (x + x_T) / 2
    ts = timeseries.TimeSeries(t, x_sym, ['Contact duration', 'Atom i', 'Atom j'])

    # Test average matrix
    mean_contacts_matrix = ts.mean(axis=0)
    assert(mean_contacts_matrix.shape == (100, 100))

    # Test if reductions along alternate axes work symmetrically
    mean_per_atom = ts.mean(axis=1)
    mean_per_atom_1 = ts.mean(axis=2)
    assert(np.allclose(mean_per_atom.data_array, mean_per_atom_1.data_array))

    # Plot 2d reduction
    fig = mean_per_atom.plot_2d_heatmap(cmap='hot')
    fig.set_dpi(300)
    fig.savefig("timeseries_test_data/3d_dimred_2d.png")

    # Plot 2d reduction
    mean_per_frame = mean_per_atom.mean(axis=1)
    assert(mean_per_frame.data_array.shape == (41,))
    fig = mean_per_frame.plot('s--', lw=1.2, label='Mean contacts per timestep')
    fig.set_dpi(300)
    fig.savefig("timeseries_test_data/3d_dimred_1d.png")


def test_TimeSeries_bootstrap():
    """Stochastic test. There is a chance this might fail"""
    x = pymbar.testsystems.correlated_timeseries_example(N=10000, tau=50, seed=415389)
    ts = timeseries.create1DTimeSeries(x)

    # Compute standard error of the mean with 100 bootstrap samples
    se1 = ts.standard_error()
    print(se1)
    assert(se1 < 0.10)

    # Compute standard error of the mean with 1000 bootstrap samples
    se2 = ts.standard_error(nboot=1000)
    print(se2)
    assert(se2 < 0.10)


def test_TimeSeries_bootstrap_correlated():
    """Completely correlated test."""
    x = np.ones(100)
    assert(timeseries.create1DTimeSeries(x).standard_error(use_pymbar=True) == 0)
    assert(timeseries.create1DTimeSeries(x).standard_error(use_pymbar=False) == 0)


def test_loadTimeSeriesFromDAT():
    # Check that loading does not fail
    ts = timeseries.loadTimeSeriesFromDAT("plumed.dat", datacols=[1, 2], labels=["N", "N~"])
    print(ts)


def test_TimeSeriesAnalysis_save_load_TimeSeries():
    t = np.linspace(0, 10000, 41)
    x = np.random.random((41, 100, 100))
    x_T = np.transpose(x, axes=(0, 2, 1))
    x_sym = (x + x_T) / 2
    ts = timeseries.TimeSeries(t, x_sym, ['Contact duration', 'Atom i', 'Atom j'])

    tsa = timeseries.TimeSeriesAnalysis()
    tsa.save_TimeSeries(ts, "timeseries_test_data/ts.pkl")

    tsl = tsa.load_TimeSeries("timeseries_test_data/ts.pkl")
    assert(np.allclose(ts.time_array, tsl.time_array))
    assert(np.allclose(ts.data_array, tsl.data_array))
    assert(ts.labels == tsl.labels)


def test_TimeSeriesAnalysis_save_load_fig():
    """Tests plotting for 1-d data"""
    t = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    x = np.array([2, 4, 6, 8, 10, 10, 10, 30])
    ts = timeseries.TimeSeries(t, x, ['RMSD'])

    fig = ts.plot('s--', lw=1.2, label='Backbone-Backbone RMSD')
    fig.set_dpi(300)
    ax = fig.gca()
    ax.legend()

    tsa = timeseries.TimeSeriesAnalysis()
    tsa.opref = "timeseries_test_data/tsa"
    tsa.oformat = "png"
    tsa.dpi = 300
    tsa.save_figure(fig, suffix="test_fig")
