"""Levenshtein distance calculation using Cython for performance optimization."""
""" NOTE: This code requires Cython to compile and run. """
""" Create a file named setup.py with the following content to compile this code:
from setuptools import setup
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("levenshteinDistance.pyx", annotate=True, compiler_directives={'language_level' : "3"})
)
Then run the command: python setup.py build_ext --inplace
"""

def levenshteinDistance(s1, s2):

    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
