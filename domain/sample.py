"""
Orbits masking and initial perturbation

"""
from typing import Optional
from itertools import combinations

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


def construct(
    points:NDArray[float64],
    epsilon:float=1.0E-9,
    threshold:float=1.0E-3,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[bool_]]:
    """
    Compute general angular data for boundary exploration

    Parameters
    ----------
    points: NDArray[float64]
        input phase space points (cartesian)
    epsilon: float, default=1.0E-9
        radius threshold
    threshold  float, default=1.0E-3
        phase threshold

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[bool_]]

    """
    _, dimension = points.shape
    size = dimension // 2
    qs = points[:, :size]
    ps = points[:, size:]
    rs = numpy.sqrt(qs*qs + ps*ps)
    rho = numpy.sqrt(numpy.sum(rs*rs, axis=1))
    us = numpy.zeros_like(rs)
    mask = rho > epsilon
    us[mask] = rs[mask]**2/rho[mask, None]**2
    fs = numpy.mod(numpy.arctan2(ps, qs), 2.0*numpy.pi)
    gs = numpy.zeros_like(us, dtype=bool_)
    gs[mask] = us[mask] > threshold
    return rho, rs, us, fs, gs


def binarize(
    values:NDArray[float64],
    bins:int,
    period:float=1.0,
    periodic:bool=False,
) -> NDArray[int]:
    """
    Bin values into integer indices

    Parameters
    ----------
    values: NDArray[float64]
        input values
    bins: int
        number of bins
    period: float, default=1.0
        interval length used for scaling
    periodic: bool, default=False
        periodic data flag

    Returns
    -------
    NDArray[int]
        bin indices in the range [0, bins - 1]

    """
    scaled = numpy.mod(values, period)/period if periodic else values/period
    indices = numpy.floor(scaled*bins).astype(int)
    return numpy.clip(indices, 0, bins - 1)


def weights(
    points:NDArray[float64],
    bins_plane:int=16,
    bins_phase:int=32,
    threshold:float=1.0E-3,
    alpha_plane:float=1.0,
    alpha_phase:float=0.5,
    boost:float=0.50,
    delta:float=1.0,
    uniform:float=0.05,
    power:float=1.0
) -> tuple[NDArray[float64], dict]:
    """
    Compute exploration weights for boundary cells (4D phase space)

    Parameters
    ----------
    points: NDArray[float64]
        marked cell centers
    bins_plane: int
        number of bins for the mixing coordinate in [0, 1]
    bins_phase: int
        number of bins for each phase_i in [0, 2*pi) in (plane, phase) histograms
    threshold:  float, default=1.0E-3
        phase threshold
    alpha_plane: float, default=1.0
        plane weight factor
    alpha_phase: float, default=0.5
        phase weight factor
    boost: float, default=0.5
        weight boost near edges
    delta: float, default=1.0
         inverse-density score regularization
    uniform: float, default=0.05
        uniform sampling fraction to use
    power: float, default=1.0
        score power

    Returns
    -------
    tuple[NDArray[float64], dict]
        probabilities: NDArray[float64]
            sampling probabilities
        statistics: dict
            coverage statistics
    """
    points = numpy.asarray(points, dtype=float64)
    *_, us, fs, gs = construct(points, threshold=threshold)
    n, k = us.shape
    score = numpy.zeros(n, dtype=float64)
    us_coverage = []
    fs_coverage = []
    bus = binarize(us[:, 1], bins_plane)
    counts = numpy.bincount(bus, minlength=bins_plane)
    score += alpha_plane/(counts[bus] + delta)
    us_coverage.append(numpy.count_nonzero(counts)/bins_plane)
    if boost > 0.0:
        score += boost/(numpy.minimum(us[:, 1], 1.0 - us[:, 1]) + 1.0/bins_plane)
    for i in range(k):
        mask = gs[:, i]
        if numpy.any(mask):
            bfs = binarize(fs[mask, i], bins_phase, 2.0*numpy.pi, True)
            key = bus[mask]*bins_phase + bfs
            counts = numpy.bincount(key, minlength=bins_plane*bins_phase)
            score[mask] += alpha_phase/(counts[key] + delta)
            fs_coverage.append(numpy.count_nonzero(counts)/(bins_plane*bins_phase))
        else:
            fs_coverage.append(0.0)
    if power != 1.0:
        score = score**power
    total = numpy.sum(score)
    if not numpy.isfinite(total) or total <= 0.0:
        probabilities = numpy.full(n, 1.0/n, dtype=float64)
    else:
        probabilities = score/total
    if uniform > 0.0:
        probabilities = (1.0 - uniform)*probabilities + uniform/n
    probabilities /= numpy.sum(probabilities)
    statistics = {
        "us_coverage": numpy.asarray(us_coverage),
        "fs_coverage": numpy.asarray(fs_coverage)
    }
    return probabilities, statistics


def select(
    domain,
    nsamples:int,
    bins_plane:int=16,
    bins_phase:int=32,
    threshold:float=1.0E-3,
    alpha_plane:float=1.0,
    alpha_phase:float=0.5,
    boost:float=0.50,
    delta:float=1.0,
    uniform:float=0.05,
    power:float=1.0
) -> tuple[NDArray[int], NDArray[float64], NDArray[float64], dict]:
    """
    Select boundary cells using angular undercoverage weights

    Exploration steps:

    1. Convert sparse occupied cell ids into cartesian cell centers
    2. Transform each center into angular data:
       - the plane amplitudes
       - the mixing coordinate
       - the in-plane phase with phase-validity mask
    3. Bin the mixing coordinate.
       Measure how strongly each mixing region is already represented by the current marked boundary cells
    4. Boost cells close to the edges of the mixing interval
    5. For each plane, bin the phase only where the corresponding amplitude is large enough
    6. Normalize the resulting scores into sampling probabilities
    7. Draw samples cell ids with replacement according to these probabilities

    Parameters
    ----------
    domain: Domain
        input domain
    nsamples: int
        number of cell ids to draw
    bins_plane: int
        number of bins for the mixing coordinate in [0, 1]
    bins_phase: int
        number of bins for each phase_i in [0, 2*pi) in (plane, phase) histograms
    threshold:  float, default=1.0E-3
        phase threshold
    alpha_plane: float, default=1.0
        plane weight factor
    alpha_phase: float, default=0.5
        phase weight factor
    boost: float, default=0.5
        weight boost near edges
    delta: float, default=1.0
         inverse-density score regularization
    uniform: float, default=0.05
        uniform sampling fraction to use
    power: float, default=1.0
        score power

    Returns
    -------
    Tuple[NDArray[int64], NDArray[float64], NDArray[float64], dict]
        selected cell ids
        selected cell centers
        probabilities (all cells)
        coverage statistics

    """
    keys = numpy.asarray(domain.list)
    points = domain.transform(keys)
    probabilities, statistics = weights(
        points,
        bins_plane=bins_plane,
        bins_phase=bins_phase,
        threshold=threshold,
        alpha_plane=alpha_plane,
        alpha_phase=alpha_phase,
        boost=boost,
        delta=delta,
        uniform=uniform,
        power=power
    )
    sample = numpy.random.choice(len(keys), size=nsamples, replace=True, p=probabilities)
    return keys[sample], points[sample], probabilities, statistics
