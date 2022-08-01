import os

import numpy as np


def read_table(table_file):
    x = []
    f = []
    mfp = []
    g = []
    mgp = []
    h = []
    mhp = []
    with open(table_file) as file:
        for line in file:
            if line.strip()[0] != '#':
                xi, fi, mfpi, gi, mgpi, hi, mhpi = [float(i) for i in line.strip().split()]
                x.append(xi)
                f.append(fi)
                mfp.append(mfpi)
                g.append(gi)
                mgp.append(mgpi)
                h.append(hi)
                mhp.append(mhpi)

    x = np.array(x)
    f = np.array(f)
    g = np.array(g)
    h = np.array(h)
    mfp = np.array(mfp)
    mgp = np.array(mgp)
    mhp = np.array(mhp)

    return x, f, g, h, mfp, mgp, mhp


# First test: reproduce LJ
os.system("python ../gen_WCA_table.py X X 1 1 1 1 -scale 1 -rcut 1 -rext 2")

x0, f0, g0, h0, mfp0, mgp0, mhp0 = read_table("table6-12_ref.xvg")
x, f, g, h, mfp, mgp, mhp = read_table("table_X_X.xvg")

assert(np.allclose(x, x0))
assert(np.allclose(f, f0))
assert(np.allclose(g, g0))
assert(np.allclose(h, h0))
assert(np.allclose(mfp, mfp0))
assert(np.allclose(mgp, mgp0))
assert(np.allclose(mhp, mhp0))

print("First set of checks (reproducing LJ) passed!")

# Second test: reproduce WCA potential used in Athawale et al (2007).
os.system("python ../gen_WCA_table.py DI OW 3.73000e-01 5.85600e-01 3.16557e-01 6.50194e-01 -scale 0 -rcut 1 -rext 1 -rspace 5e-4 --nocoulomb --noscale")

x0, f0, g0, h0, mfp0, mgp0, mhp0 = read_table("table_DI_OW_ref.xvg")
x, f, g, h, mfp, mgp, mhp = read_table("table_DI_OW.xvg")

assert(np.allclose(x, x0))
assert(np.allclose(f, f0))
assert(np.allclose(g, g0))
assert(np.allclose(h, h0))
assert(np.allclose(mfp, mfp0))
assert(np.allclose(mgp, mgp0))
assert(np.allclose(mhp, mhp0))

print("Second set of checks (reproducing Athawale et al) passed!")
