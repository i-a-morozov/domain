"""
Adapters
--------

- (n, m) mapping -> (n,  ) mapping (scalarize)
- (n,  ) mapping -> (n, m) mapping (vectorize)

"""
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from numba import njit
from numba import prange


def scalarize(
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Scalarize input (n, m) mapping to act like (n, ) mapping

    Parameters
    ----------
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        input (n, m) mapping
    
    Returns
    -------
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        output (n, ) mapping

    """
    @njit
    def closure(
        state: NDArray[np.float64],
        parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        local = np.ascontiguousarray(state)
        return mapping(local.reshape(-1, 1), parameters).reshape(-1)
    return closure


def vectorize(
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], 
    parallel: bool = False
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Vectorize input (n, ) mapping to act like (n, m) mapping

    Parameters
    ----------
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        input (n, ) mapping
    parallel: bool, default=False
        parallel flag
    
    Returns
    -------
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        output (n, m) mapping

    """
    @njit(parallel=parallel)
    def closure(
        state: NDArray[np.float64],
        parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n, m = state.shape
        result = np.empty((n, m), dtype=np.float64)
        for i in prange(m):
            result[:, i] = mapping(state[:, i], parameters)
        return np.ascontiguousarray(result)
    return closure
