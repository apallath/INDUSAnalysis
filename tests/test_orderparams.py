"""test using PyTest"""
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

from analysis.protein_order_params import OrderParams

#Radius of gyration
def generate_xyz(natoms,boxlen):
    with open('test.xyz','w+') as tf:
        tf.write('{}\n'.format(natoms))
        tf.write('testsystem\n')
        for i in range(natoms):
            [x,y,z] = [boxlen * np.random.rand(), boxlen * np.random.rand(), boxlen * np.random.rand()]
            tf.write('{:.4f} {:.4f} {:.4f}\n'.format(x,y,z))

def test_Rg():
    generate_xyz(30,10)
    assert(1==1)
