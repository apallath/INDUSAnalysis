from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
from Cython.Distutils import build_ext

ext_modules = [ Extension("INDUSAnalysis.timeseries", ["INDUSAnalysis/timeseries.pyx"],
                            include_dirs=[numpy_include]),
                Extension("INDUSAnalysis.indus_waters", ["INDUSAnalysis/indus_waters.pyx"],
                            include_dirs=[numpy_include]),
                Extension("INDUSAnalysis.protein_order_params", ["INDUSAnalysis/protein_order_params.pyx"],
                            include_dirs=[numpy_include]),
                Extension("INDUSAnalysis.contacts", ["INDUSAnalysis/contacts.pyx"],
                            include_dirs=[numpy_include]),
                Extension("INDUSAnalysis.lib.profiling", ["INDUSAnalysis/lib/profiling.pyx"],
                            include_dirs=[numpy_include])
              ]

setup(
    name='INDUSAnalysis',
    version='0.3-alpha',
    packages=['INDUSAnalysis'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
