from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("./src/solver/solver.pyx"),
    include_dirs=[numpy.get_include()]
)
