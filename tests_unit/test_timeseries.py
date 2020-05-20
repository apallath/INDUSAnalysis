import MDAnalysis as mda
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from analysis.timeseries import TimeSeries

"""CMA tests"""

def test_cma_basic():
    ts = TimeSeries()
    x = np.array(range(5))
    xcma = np.array([0, 0.5, 1, 1.5, 2])
    assert(np.isclose(np.linalg.norm(ts.cumulative_moving_average(x) - xcma),0))

def test_cma_visualization():
    ts = TimeSeries()
    x = np.linspace(0, 8, 1000)
    y = np.sin(np.pi * x)
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(x, y, label="Data")
    plt.plot(x, ts.cumulative_moving_average(y), label="CMA")
    plt.legend()
    plt.savefig("cma_vis.png")
    assert(np.isclose(ts.cumulative_moving_average(y)[-1], 0))
