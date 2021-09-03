from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext_modules = [Extension("INDUSAnalysis.timeseries", ["INDUSAnalysis/timeseries.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("INDUSAnalysis.indus_waters", ["INDUSAnalysis/indus_waters.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("INDUSAnalysis.protein_order_params", ["INDUSAnalysis/protein_order_params.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("INDUSAnalysis.contacts", ["INDUSAnalysis/contacts.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("INDUSAnalysis.lib.profiling", ["INDUSAnalysis/lib/profiling.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("INDUSAnalysis.lib.collective", ["INDUSAnalysis/lib/collective.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]

setup(
    name='INDUSAnalysis',
    version='0.2a0',
    packages=['INDUSAnalysis'],
    ext_modules=cythonize(ext_modules,
                          compiler_directives={'language_level' : "3"})
)
