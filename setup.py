from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["sgd_opt.pyx", "loss_functions.pyx"])
)
