"""
Escape
------

Finite-state guard for custom escape predicates

"""
from typing import Callable

import numpy
from numpy import float64
from numpy.typing import NDArray
from numba import njit

Escape = Callable[[NDArray[float64], float, NDArray[float64]], bool]


@njit
def escaped(
    state:NDArray, 
    radius:float64, 
    parameters:NDArray, 
    escape:Escape
) -> bool:
    for value in state:
        if not numpy.isfinite(value):
            return True
    if parameters is None:
        return escape(state, radius, numpy.empty(0, dtype=float64))
    return escape(state, radius, parameters)
