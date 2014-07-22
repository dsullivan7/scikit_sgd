from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = []
extensions.append(Extension("sgd_opt", ["sgd_opt.pyx"],
                            libraries=["cblas"],
                            include_dirs=[np.get_include(), "cblas"]
                            ))
extensions.append(Extension("loss_functions", ["loss_functions.pyx"]
                            ))

setup(ext_modules=cythonize(extensions))
