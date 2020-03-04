"""test using PyTest"""
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

from analysis.protein_order_params import OrderParams

#Test radius of gyration

def Rg_compare_snapshot(natoms,boxlen):
    #generate coordinates
    coords = []
    for i in range(natoms):
        c = [boxlen * np.random.rand(), boxlen * np.random.rand(), boxlen * np.random.rand()]
        coords.append(c)
    coords = np.array(coords)

    #simple Rg calculation
    com = np.mean(coords, axis=0)
    sq_distances = np.sum((coords - com)**2, axis = 1)
    true_Rg = np.sqrt(np.mean(sq_distances))
    print(true_Rg)

    #create Universe with H-atoms at same coordinates
    u = mda.Universe.empty(natoms, trajectory=True)
    u.add_TopologyAttr('mass', [1.0]*natoms)
    u.atoms.positions = coords
    sel = u.select_atoms("all")

    op_test = OrderParams()
    test_Rg = op_test.calc_Rg(u, "all")[0,1]

    assert(np.isclose(true_Rg, test_Rg))

def test_Rg():
    for i in range(100):
        boxlen = np.random.randint(100)
        natoms = np.random.randint(1000)
        Rg_compare_snapshot(natoms,boxlen)
