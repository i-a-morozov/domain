"""
Scan functions

"""
from typing import Callable
from typing import Union

from numpy import bool_
from numpy import int64
from numpy import float64
from numpy.typing import NDArray
import numpy

from numba import njit
from numba import prange


def iterate(
    length:int,
    radius:float,
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]], 
) -> Callable[[NDArray[float64], NDArray[float64]], bool]:
    """
    Iteration factory
    Creates a function acting on initial condition
    Returns True/False if corresponding orbit is bounded/unbounded within given threshold radius

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    radius: float
        threshold radius
    mapping: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        mapping

    Returns
    -------
    Callable[[NDArray[float64], NDArray[float64]], bool]
    ((dimension, ), (...)) -> bool
    
    """
    threshold = radius*radius
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64]
    ) -> bool:
        dimension = len(state)
        distance = 0.0
        for i in range(dimension):
            distance += state[i]*state[i]
        if distance > threshold:
            return False
        local = numpy.copy(state)
        for _ in range(length):
            local = mapping(local, parameters)
            distance = 0.0
            for i in range(dimension):
                distance += local[i]*local[i]
            if distance > threshold or numpy.isnan(distance):
                return False
        return True
    return closure


def count(
    length:int,
    radius:float,
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]], 
) -> Callable[[NDArray[float64], NDArray[float64]], int]:
    """
    Count (survival) factory
    Creates a function acting on initial condition
    Returns the number of iterations for which the corresponding orbit remains within given threshold radius

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    radius: float
        threshold radius
    mapping: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        mapping

    Returns
    -------
    Callable[[NDArray[float64], NDArray[float64]], int]
    ((dimension, ), (...)) -> int
    
    """
    threshold = radius*radius
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64]
    ) -> int:
        count = 0
        dimension = len(state)
        distance = 0.0
        for i in range(dimension):
            distance += state[i]*state[i]
        if distance > threshold or numpy.isnan(distance):
            return count
        local = numpy.copy(state)
        for _ in range(length):
            local = mapping(local, parameters)
            distance = 0.0
            for i in range(dimension):
                distance += local[i]*local[i]
            if distance > threshold or numpy.isnan(distance):
                return count
            count += 1
        return count
    return closure


def orbit(
    length:int,
    radius:float,
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
) ->  Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]:
    """
    Orbit factory
    Creates a function acting on initial condition
    Returns corresponding full length orbit
    Padded by NaN values if initial escapes the threshold radius

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    radius: float
        threshold radius
    mapping: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        mapping

    Returns
    -------
    Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
    ((dimension, ), (...)) -> (length, dimension)
    
    """    
    threshold = radius*radius
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64]
    ) -> NDArray[float64]:
        dimension = len(state)
        local = numpy.copy(state)
        orbit = numpy.full((length, dimension), numpy.nan, dtype=float64)
        distance = 0.0
        for i in range(dimension):
            distance += state[i]*state[i]
        if distance > threshold or numpy.isnan(distance):
            return orbit
        for i in range(length):
            local = mapping(local, parameters)
            orbit[i] = local
            distance = 0.0
            for j in range(dimension):
                distance += local[j]*local[j]
            if distance > threshold or numpy.isnan(distance):
                return orbit
        return orbit
    return closure


def final(
    length:int,
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
) ->  Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]:
    """
    Final factory
    Creates a function acting on initial condition
    Returns only the final value after given number of iterations (can be NaN)

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    mapping: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        mapping

    Returns
    -------
    Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
    ((dimension, ), (...)) -> (dimension, )
    
    """      
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64]
    ) -> NDArray[float64]:
        local = numpy.copy(state)
        for i in range(length):
            local = mapping(local, parameters)
        return local
    return closure


def rem(
    limit:int,
    forward:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    inverse:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    epsilon:float=1.0E-16
) -> Callable[[NDArray[float64], NDArray[float64]], float64]:
    """
    REM indicator factory
    Creates a function acting on initial condition

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    forward: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        forward mapping
    inverse: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        inverse mapping
    epsilon: float, default=1.0E-16
        epsilon perturbation added after forward iteration (before backward iteration)

    Returns
    -------
    Callable[[NDArray[float64], NDArray[float64]], float64]
    ((dimension, ), (...)) -> float64
    
    """
    final_forward = final(limit, forward)
    final_inverse = final(limit, inverse)
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64]
    ) -> float64:
        dimension = len(state)
        local = numpy.copy(state)
        local = final_forward(local, parameters)
        local = final_inverse(local + epsilon, parameters)
        delta = local - state
        error = 0
        for i in range(dimension):
            error += delta[i]*delta[i]
        return numpy.sqrt(error)
    return closure


@njit(parallel=True)
def scan(
    grid:NDArray[float64],
    container:Union[NDArray[bool_], NDArray[int64], NDArray[float64]],
    generator:Callable[[NDArray[float64], NDArray[float64]], Union[bool, int, NDArray[float64]]],
    parameters:NDArray[float64]
) -> None:
    """
    Scan function
    Performs (parallel) scan over a set of initials for given generator
    Modifies input result container

    Parameters
    ----------
    grid: NDArray[float64]
        grid of initial conditions
    container: Union[NDArray[bool_], NDArray[int64], NDArray[float64]]
        result container
    generator: Callable[[NDArray[float64], NDArray[float64]], Union[bool, int, NDArray[float64]]]
        result generator acting on a single initial
    parameters: NDArray[float64]
        additional parameters passed to generator

    Returns
    -------
    None
    
    """
    for i in prange(len(grid)):
        container[i] = generator(grid[i], parameters)
