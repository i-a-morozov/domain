"""
Survival DA computation along rays

"""
from typing import Tuple
from typing import Callable

from numpy import float64
from numpy.typing import NDArray
import numpy

from numba import njit
from numba import prange

@njit(parallel=True)
def da(
    dimension:int,
    step:float,
    radius:float,   
    origin:NDArray[float64],
    directions:NDArray[float64],    
    objective:Callable[[NDArray[float64], NDArray[float64]], bool], 
    parameters:NDArray[float64],
    start:float=0.0,
    unstable:bool=False,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    DA computation over directions (last stable radius)

    Parameters
    ----------
    dimension: int
        phase space dimension
    step: float
        radial step size
    radius: float
        maximum radius
    origin: NDArray[float64]
        origin
    directions: NDArray[float64]
        search directions, (..., dimension)
    objective: Callable[[NDArray[float64], NDArray[float64]], bool]
        search objective (search is stopped once objective returns False)
        ((dimension, ), (...)) -> bool
    parameters: NDArray[float64]
        additional parameters passed to objective
    start: float, default=0.0
        starting radius for all directions
    unstable: bool, default=False
        flag to return last unstable point
    
    Returns
    -------
    Tuple[NDArray[float64], NDArray[float64]]
        radii (len(direction), ) and endpoints (len(direction), dimension) for each direction
        
    """
    count = len(directions)
    limit = int(numpy.maximum(1, numpy.ceil(radius/step))) + 1
    rb = numpy.zeros(count, dtype=float64)
    xb = numpy.zeros((count, dimension), dtype=float64)
    for i in prange(count):
        direction = directions[i]
        final = -1
        for k in range(limit):
            r = start + k*step
            if r > radius:
                r = radius
            x = origin + r*direction
            if objective(x, parameters):
                final = k
                if r >= radius:
                    break
            else:
                break
        if final < 0:
            last = 0
        else:
            last = final + (1 if unstable else 0)
        r = start + last*step
        if r > radius:
            r = radius
        rb[i] = r
        xb[i] = origin + r*direction
    return rb, xb


@njit(parallel=True)
def refine(
    dimension:int,
    step:float,
    radius:float,
    origin:NDArray[float64],
    directions:NDArray[float64],
    refine:int,
    start:NDArray[float64],    
    objective:Callable[[NDArray[float64], NDArray[float64]], bool],
    parameters:NDArray[float64]
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    DA refinement over directions (refines last stable radius using bisection)

    Parameters
    ----------
    dimension: int
        phase space dimension
    step: float
        radial step size
    radius: float
        maximum radius
    origin: NDArray[float64]
        origin
    directions: NDArray[float64]
        search directions, (..., dimension)
    refine: int
        number of bisection steps
    objective: Callable[[NDArray[float64], NDArray[float64]], bool]
        search objective (search is stopped once objective returns False)
        ((dimension, ), (...)) -> bool
    parameters: NDArray[float64]
        additional parameters passed to objective
    
    Returns
    -------
    Tuple[NDArray[float64], NDArray[float64]]
        radii (len(direction), ) and endpoints (len(direction), dimension) for each direction
        
    """    
    count = len(directions)
    rb = numpy.empty(count, dtype=float64)
    xb = numpy.empty((count, dimension), dtype=float64)
    for i in prange(count):
        direction = directions[i]
        rl = start[i]
        if rl < 0.0:
            rl = 0.0
        if rl >= radius:
            rl = radius
            rb[i] = rl
            xb[i] = origin + rl*direction
            continue
        ru = rl + step
        if ru > radius:
            ru = radius
        if ru <= rl:
            rb[i] = rl
            xb[i] = origin + rl*direction
            continue
        for _ in range(refine):
            rm = 0.5*(rl + ru)
            if rm == rl or rm == ru:
                break
            xm = origin + rm * direction
            if objective(xm, parameters):
                rl = rm
            else:
                ru = rm
        rb[i] = rl
        xb[i] = origin + rl*direction
    return rb, xb
