# -*- coding: utf-8 -*-

"""
## Python package dpvssp
"""

__version__ = '0.0.0'

import numpy as np


def generate_random_array(shape, dtype=np.float64) -> np.ndarray:
    """Generate an array of random numbers between 0 and 1.

    Args:
        shape: shape of the array
        dtype: dtype of the array, a floating point type.

    Returns:
        an array with shape=`shape` and dtype=`dtype`.
    """

    a = np.random.random(shape)
    if not dtype in (float, np.float64):
        a = a.astype(dtype=dtype)
    return a

