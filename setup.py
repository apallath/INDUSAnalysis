from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
from Cython.Distutils import build_ext

ext_modules = [ Extension("analysis.timeseries", ["analysis/timeseries.pyx"],
                            include_dirs=[numpy_include]),
                Extension("analysis.indus_waters", ["analysis/indus_waters.pyx"],
                            include_dirs=[numpy_include]),
                Extension("analysis.protein_order_params", ["analysis/protein_order_params.pyx"],
                            include_dirs=[numpy_include]),
                Extension("analysis.contacts", ["analysis/contacts.pyx"],
                            include_dirs=[numpy_include]),
                Extension("meta_analysis.profiling", ["meta_analysis/profiling.pyx"],
                            include_dirs=[numpy_include])
              ]

setup(
    name='AnalysisScripts',
    version='0.2dev',
    packages=['analysis', 'meta_analysis'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
