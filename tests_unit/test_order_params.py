"""On the fly Cython compilation for development versions"""
import pyximport; pyximport.install()

import MDAnalysis as mda
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from analysis.protein_order_params import OrderParams

def coords_generator(boxlen, natoms):
    """generates random coordinates for test cases"""
    coords = []
    for i in range(natoms):
        c = [boxlen * np.random.rand(), boxlen * np.random.rand(), boxlen * np.random.rand()]
        coords.append(c)
    coords = np.array(coords)
    print(coords.shape)
    return coords

"""RMSD tests"""

def test_RMSD_basic():
    """no translations"""
    op = OrderParams()
    """random system with itself"""
    coords = coords_generator(10, 10)
    assert(np.isclose(op.calc_RMSD_worker(coords,coords,coords,coords),0))

    """known sample system"""
    coords1 = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]])
    coords2 = 2.0 * coords1
    assert(np.isclose(op.calc_RMSD_worker(coords1,coords2,coords1,coords2),1))

def test_RMSD_translate():
    """test RMSD translational invariance"""
    coords = coords_generator(10, 10)
    newcoords = coords + 100.0 * np.random.random_sample(3)
    op = OrderParams()
    assert(np.isclose(op.calc_RMSD_worker(coords,newcoords,coords,newcoords),0))

def test_RMSD_known_rotate():
    """test RMSD rotational invariance"""
    coords = coords_generator(10, 10)
    Rmat = np.array([[0, -1.0, 0],
                     [1.0, 0, 0],
                     [0, 0, 1.0]])
    newcoords = np.dot(coords, Rmat.T)
    op = OrderParams()
    assert(np.isclose(op.calc_RMSD_worker(coords,newcoords,coords,newcoords),0))

def test_RMSD_rand_rotate():
    """test RMSD rotational invariance"""
    coords = coords_generator(10, 10)
    #generate random rotation matrix
    rot = scipy_R.from_rotvec(np.random.random_sample(3))
    newcoords = rot.apply(coords)
    op = OrderParams()
    assert(np.isclose(op.calc_RMSD_worker(coords,newcoords,coords,newcoords),0))

def test_RMSD_translate_rotate():
    """test RMSD translation and rotational invariance"""
    coords = coords_generator(10, 10)
    newcoords = coords + 100.0 * np.random.random_sample(3)
    rot = scipy_R.from_rotvec(np.random.random_sample(3))
    newcoords = rot.apply(newcoords)
    op = OrderParams()
    assert(np.isclose(op.calc_RMSD_worker(coords,newcoords,coords,newcoords),0))

"""Per-atom-deviation test"""
def test_deviation_translate_rotate_known():
    coords = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]])
    newcoords = 2.0 * coords + 100.0 * np.random.random_sample(3)
    rot = scipy_R.from_rotvec(np.random.random_sample(3))
    newcoords = rot.apply(newcoords)
    op = OrderParams()
    calc_deviations = op.calc_deviation_worker(coords,newcoords,coords,newcoords)
    known_deviations = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert(np.allclose(calc_deviations,known_deviations))
