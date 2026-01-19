"""
Orbits masking and initial perturbation

"""
from typing import Optional

from numpy import bool_
from numpy import float64
from numpy.typing import NDArray
import numpy

from numba import njit
from numba import prange

@njit(parallel=True)
def mask(
    orbits: NDArray[float64],
    radius: float,
    cut: Optional[int] = None,
) -> NDArray[bool_]:
    """
    Mask escaped orbits

    An orbit is marked lost if:
      - any NaN appears in any point of the orbit
      - any point lies outside the hyperball of radius

    The cut controls when loss is detected:
      - cut is None: detect loss at any iteration (full scan)
      - cut is int: detect loss only for iterations at or after cut

    Parameters
    ----------
    orbits: NDArray[float64]
        orbits
    radius: float
        escape radius
    cut: Optional[int]
        threshold iteration index

    Returns
    -------
    NDArray[bool_]

    """
    size, length, dimension = orbits.shape
    threshold = radius*radius
    mask = numpy.zeros(size, dtype=bool_)
    flag = cut is not None
    for i in prange(size):
        lost = False
        for j in range(length):
            nan = False
            square = 0.0
            for k in range(dimension):
                point = orbits[i, j, k]
                if numpy.isnan(point):
                    nan = True
                    break
                else:
                    square += point*point
            bad = nan or (square > threshold)
            if not flag:
                if bad:
                    lost = True
                    break
            else:
                if j >= cut and bad:
                    lost = True
                    break
        mask[i] = lost
    return mask


@njit
def normalize(point:NDArray[float64]) -> NDArray[float64]:
    norm = 0.0
    for i in range(len(point)):
        norm += point[i]*point[i]
    norm = numpy.sqrt(norm)
    return point/norm


@njit
def shell(
    point:NDArray[float64],
    basis:NDArray[float64],
    scale:float,
    nball:int,
) -> NDArray[float64]:
    """
    K-ball random shell sampling

    Parameters
    ----------
    point: Array
        fixed point (ball center)
    basis: Array
        orthonormal basis
    scale: float
        ball radius
    nball: int
        number of sample points

    Returns
    -------
    Array

    """
    dimension, size = basis.shape
    points = numpy.empty((nball, dimension), dtype=float64)
    exponent = 1.0/size
    for i in range(nball):
        vector = normalize(numpy.random.randn(size))
        radius = numpy.random.random()**exponent
        delta = numpy.zeros(dimension, dtype=float64)
        for j in range(dimension):
            total = 0.0
            for k in range(size):
                total += vector[k]*basis[j, k]
            delta[j] = total
        for j in range(dimension):
            points[i, j] = point[j] + scale*radius*delta[j]
    return points


@njit(parallel=True)
def sample(
    count:int,
    scale:float,
    cloud:NDArray[float64],
) -> NDArray[float64]:
    """
    Generate sample by perturbation (K-ball volume)

    Parameters
    ----------
    count: int
        number of new points for each cloud point
    scale: float
        perturbation scale
    cloud: NDArray[float64]
        cloud

    Returns
    -------
    NDArray[float64]

    """
    length, dimension = cloud.shape
    points = numpy.empty((length * count, dimension), dtype=float64)
    exponent = 1.0/dimension
    for i in prange(length):
        shift = i*count
        point = cloud[i]
        for j in range(count):
            idx = shift + j
            vector = normalize(numpy.random.randn(dimension))
            radius = numpy.random.random()**exponent
            for j in range(dimension):
                points[idx, j] = point[j] + scale*radius*vector[j]
    return points


@njit
def filter(
    cloud:NDArray[float64],
    radius:float,
) -> NDArray[float64]:
    """
    Filter points by NaNs and hyperball radius

    Parameters
    ----------
    cloud: NDArray[float64]
        input cloud of shape (length, dimension)
    radius: float
        radius threshold

    Returns
    -------
    NDArray[float64]

    """
    length, dimension = cloud.shape
    threshold = radius*radius
    keep = numpy.zeros(length, dtype=bool_)
    count = 0
    for i in range(length):
        nan = False
        square = 0.0
        for j in range(dimension):
            value = cloud[i, j]
            if numpy.isnan(value):
                nan = True
                break
            square += value*value
        good = (not nan) and (square < threshold)
        keep[i] = good
        if good:
            count += 1
    out = numpy.empty((count, dimension), dtype=float64)
    k = 0
    for i in range(length):
        if keep[i]:
            for j in range(dimension):
                out[k, j] = cloud[i, j]
            k += 1
    return out
