from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules=cythonize('update_cython.py'))
setup(ext_modules=cythonize('dynamics_cython.py'))
