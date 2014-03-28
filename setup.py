#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
from numpy.distutils.system_info import get_info

includes = [np.get_include()]

extensions = [
    Extension(
        "mental_rotation.model.model_c", 
        ["lib/mental_rotation/model/model_c.pyx"],
        include_dirs=includes,
        libraries=["m"]
    ),
]

setup(
    name='mental_rotation',
    version="0.0.1",
    description='Project analysing mental rotation',
    author='Jessica B. Hamrick',
    author_email='jhamrick@berkeley.edu',
    url='https://github.com/jhamrick/optimal-mental-rotation',
    package_dir={'': 'lib'},
    packages=['mental_rotation'],
    ext_modules=cythonize(extensions),
    keywords='bayesian statistics psychology',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    install_requires=[
        'numpy',
        'scipy',
        'Cython',
        'gaussian_processes',
        'bayesian_quadrature'
    ]
)
