from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy  # Import NumPy here to get its include path

setup(
    ext_modules=cythonize(Extension(
        "sumsqdiff",  # Replace with your actual module name
        ["sumsqdiff.pyx"],  # Replace with your Cython file
        include_dirs=[numpy.get_include()]  # Adds the NumPy header files
    )),
    zip_safe=False,
)