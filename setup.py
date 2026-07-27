from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np
np_include = np.get_include()



extensions = [
    Extension(
        name="lupy.signalutils._sosfilt",
        sources=["src/lupy/signalutils/_sosfilt.pyx"],
        include_dirs=[np_include],
        language="c",
    ),
    Extension(
        name="lupy.signalutils._upfirdn_apply",
        sources=["src/lupy/signalutils/_upfirdn_apply.pyx"],
        include_dirs=[np_include],
        language="c",
    ),
]

setup(ext_modules=cythonize(extensions, annotate=True))
