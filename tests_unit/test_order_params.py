import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from INDUSAnalysis import protein_order_params


def coords_generator(boxlen, natoms):
    """Generates random coordinates for test cases."""
    coords = []
    for i in range(natoms):
        c = [boxlen * np.random.rand(), boxlen * np.random.rand(), boxlen * np.random.rand()]
        coords.append(c)
    coords = np.array(coords)
    print(coords.shape)
    return coords


def test_RMSD_basic():
    """Tests RMSD for known transformation."""

    op = protein_order_params.OrderParamsAnalysis()
    # Random system with itself
    coords = coords_generator(10, 10)
    assert(np.isclose(op.calc_RMSD_worker(coords, coords, coords, coords), 0))

    # Known sample system
    coords1 = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]])
    coords2 = 2.0 * coords1
    assert(np.isclose(op.calc_RMSD_worker(coords1, coords2, coords1, coords2), 1))


def test_RMSD_translate():
    """Tests RMSD translational invariance."""
    coords = coords_generator(10, 10)
    newcoords = coords + 100.0 * np.random.random_sample(3)
    op = protein_order_params.OrderParamsAnalysis()
    assert(np.isclose(op.calc_RMSD_worker(coords, newcoords, coords, newcoords), 0))


def test_RMSD_known_rotate():
    """Tests RMSD rotational invariance with known rotation."""
    coords = coords_generator(10, 10)
    Rmat = np.array([[0, -1.0, 0],
                     [1.0, 0, 0],
                     [0, 0, 1.0]])
    newcoords = np.dot(coords, Rmat.T)
    op = protein_order_params.OrderParamsAnalysis()
    assert(np.isclose(op.calc_RMSD_worker(coords, newcoords, coords, newcoords), 0))


def test_RMSD_rand_rotate():
    """Tesst RMSD rotational invariance with random rotation."""
    coords = coords_generator(10, 10)
    # Generates random rotation matrix
    rot = scipy_R.from_rotvec(np.random.random_sample(3))
    newcoords = rot.apply(coords)
    op = protein_order_params.OrderParamsAnalysis()
    assert(np.isclose(op.calc_RMSD_worker(coords, newcoords, coords, newcoords), 0))


def test_RMSD_translate_rotate():
    """Tests RMSD translation and rotational invariance."""
    coords = coords_generator(10, 10)
    newcoords = coords + 100.0 * np.random.random_sample(3)
    rot = scipy_R.from_rotvec(np.random.random_sample(3))
    newcoords = rot.apply(newcoords)
    op = protein_order_params.OrderParamsAnalysis()
    assert(np.isclose(op.calc_RMSD_worker(coords, newcoords, coords, newcoords), 0))


def test_deviation_translate_rotate_known():
    """Tests per-atom-deviation for known transformation."""
    coords = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])
    newcoords = 2.0 * coords + 100.0 * np.random.random_sample(3)
    rot = scipy_R.from_rotvec(np.random.random_sample(3))
    newcoords = rot.apply(newcoords)
    op = protein_order_params.OrderParamsAnalysis()
    calc_deviations = op.calc_deviation_worker(coords, newcoords, coords, newcoords)
    known_deviations = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert(np.allclose(calc_deviations, known_deviations))
