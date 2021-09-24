"""Compares Rg to Debdas' Rg."""
import numpy as np
from INDUSAnalysis.timeseries import TimeSeriesAnalysis

n_star_win = [-240, -220, -200, -180, -160, -140, -120, -100, -80, -60, -40, -20, 0,
              20, 40, 60, 80, 100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260,
              280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580]

Rg_win_1 = []
Rg_win_2 = []

tsa = TimeSeriesAnalysis()

for nstar in [str(ns) for ns in n_star_win]:
    ts = tsa.load_TimeSeries('K0.0243/K0.0243N{}/C45_Rg.pkl'.format(nstar))
    Rg_win_1.append(ts.data_array)

    with open('../C45_T298/K0.0243/K0.0243N{}/rg_ply.dat'.format(nstar)) as f:
        Rg_i = []
        for line in f:
            if line.strip()[0] != '#':
                vals = line.strip().split()
                Rg_i.append(float(vals[1]))
        Rg_i = np.array(Rg_i)
        Rg_win_2.append(Rg_i)

for i in range(len(Rg_win_1)):
    print(n_star_win[i])

    #print(Rg_win_1[i].shape)
    #print(Rg_win_2[i].shape)

    #Norm between two vectors
    print(np.sqrt(np.sum((Rg_win_1[i] - 10 * Rg_win_2[i]) ** 2)))
