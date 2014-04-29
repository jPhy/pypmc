try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension

from Cython.Build import cythonize

# set the version number
with open('pypmc/_version.py') as f:
    exec(f.read())

n = 'pypmc'

extra_compile_args=["-Wno-unused-but-set-variable",
                    "-Wno-unused-function",
                    "-O3"]

extensions = [Extension('*', ['pypmc/*/*.pyx' ], extra_compile_args=extra_compile_args),
              Extension('etos', ['pypmc/mix_adapt/etos.cxx'],
                        extra_compile_args=extra_compile_args + ['-std=c++11'],
                        libraries= ['lapack', 'blas'])]

setup(
    name=n,
    packages=[n],
    version=__version__,
    author='Frederik Beaujean, Stephan Jahn',
    author_email='Frederik.Beaujean@lmu.de, stephan.jahn@mytum.de',
    license='GPLv2',
    install_requires=['numpy', 'scipy', 'cython'],
    extras_require={'testing': ['nose'], 'plotting': ['matplotlib']},
    ext_modules=cythonize(extensions,
                          compiler_directives=dict(profile=False, boundscheck=False,
                                                   wraparound=False, cdivision=True),
                          )
    )
