"""
Fixed point
-----------

Periodic fixed point search and related functions

"""
from typing import Callable
from typing import Optional
from typing import Tuple

from numpy import bool_
from numpy import complex128
from numpy import float64
from numpy import int64
from numpy.typing import NDArray
import numpy

from numba import njit

from domain.adapters import vectorize
from domain.cbm import edges
from domain.cbm import psolve
from domain.cbm import tiles


Mapping = Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]


@njit
def trajectory(
    length:int,
    mapping:Mapping,
    state:NDArray[float64],
    parameters:NDArray[float64],
    orbit:bool=True,
) -> NDArray[float64]:
    """
    Generate a mapping trajectory

    Parameters
    ----------
    length: int
        trajectory length
    mapping: Mapping
        mapping
    state: NDArray[float64]
        initial state
    parameters: NDArray[float64]
        additional mapping parameters
    orbit: bool, default=True
        flag to return all iterations

    Returns
    -------
    NDArray[float64]
        trajectory

    """
    local = numpy.ascontiguousarray(state)
    if orbit:
        table = numpy.empty((length, *state.shape), dtype=float64)
        for i in range(length):
            local = mapping(local, parameters)
            table[i] = local
        return table
    for _ in range(length):
        local = mapping(local, parameters)
    return local.reshape(1, *local.shape)


def power(
    order:int,
    mapping:Mapping,
) -> Mapping:
    """
    Generate a mapping power

    Parameters
    ----------
    order: int
        mapping power
    mapping: Mapping
        mapping

    Returns
    -------
    Mapping

    """
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64],
    ) -> NDArray[float64]:
        local = state
        for _ in range(order):
            local = mapping(local, parameters)
        return local
    return closure


def problem(
    mapping:Mapping,
    order:int=1,
    roots:Optional[NDArray[float64]]=None,
    powers:Optional[NDArray[int64]]=None,
    alpha:float=1.0E-4,
    epsilon:float=1.0E-16,
) -> Mapping:
    """
    Generate a periodic fixed-point residual with optional deflation

    The input mapping must accept vector states shaped ``(dimension, count)``
    Use :func:`domain.adapters.vectorize` to adapt a scalar-state Domain mapping

    Parameters
    ----------
    mapping: Mapping
        vector mapping
    order: int, default=1
        fixed-point order
    roots: Optional[NDArray[float64]], default=None
        known roots to deflate
    powers: Optional[NDArray[int64]], default=None
        root multiplicities
    alpha: float, default=1.0E-4
        deflation offset
    epsilon: float, default=1.0E-16
        deflation stabilization

    Returns
    -------
    Mapping
        vector residual mapping

    """
    @njit
    def weight(
        state:NDArray[float64],
        known:NDArray[float64],
        multiplicities:NDArray[int64],
        offset:float,
        regularization:float,
    ) -> NDArray[float64]:
        dimension, length = state.shape
        factors = numpy.ones(length, dtype=float64)
        for i in range(len(known)):
            for j in range(length):
                square = 0.0
                for k in range(dimension):
                    delta = state[k, j] - known[i, k]
                    square += delta*delta
                factors[j] *= (offset + (square + regularization)**(-0.5*multiplicities[i]))
        return factors
    if roots is not None:
        known = numpy.asarray(roots, dtype=float64)
        multiplicities = (numpy.ones(len(known), dtype=int64) if powers is None else numpy.asarray(powers, dtype=int64))
        @njit
        def closure(
            state:NDArray[float64],
            parameters:NDArray[float64],
        ) -> NDArray[float64]:
            local = state
            for _ in range(order):
                local = mapping(local, parameters)
            return (weight(state, known, multiplicities, alpha, epsilon)*(local - state))
        return closure
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64],
    ) -> NDArray[float64]:
        local = state
        for _ in range(order):
            local = mapping(local, parameters)
        return local - state
    return closure


def chain(
    order:int,
    mapping:Mapping,
) -> Mapping:
    """
    Generate a periodic-chain callable

    Parameters
    ----------
    order: int
        prime period
    mapping: Mapping
        mapping

    Returns
    -------
    Mapping

    """
    @njit
    def closure(
        state:NDArray[float64],
        parameters:NDArray[float64],
    ) -> NDArray[float64]:
        return trajectory(order, mapping, state, parameters)
    return closure


def prime(
    mapping:Mapping,
    order:int=1,
    rtol:float=1.0E-12,
    atol:float=1.0E-12,
) -> Callable[[NDArray[float64], NDArray[float64]], bool]:
    """
    Generate a prime-period fixed-point test

    Parameters
    ----------
    mapping: Mapping
        mapping
    order: int, default=1
        fixed-point order
    rtol: float, default=1.0E-12
        relative tolerance
    atol: float, default=1.0E-12
        absolute tolerance

    Returns
    -------
    Callable[[NDArray[float64], NDArray[float64]], bool]

    """
    generate = chain(order, mapping)
    def closure(
        point:NDArray[float64],
        parameters:NDArray[float64],
    ) -> bool:
        orbit = generate(point, parameters)
        close = numpy.isclose(
            orbit,
            point,
            rtol=rtol,
            atol=atol,
        ).all(axis=-1)
        return bool(numpy.sum(close) == 1)
    return closure


def exact(
    order:int,
    mapping:Mapping,
    points:NDArray[float64],
    parameters:NDArray[float64],
    tolerance:float=1.0E-9,
) -> NDArray[bool_]:
    """
    Test candidate points for exact prime period

    Parameters
    ----------
    order: int
        fixed-point order
    mapping: Mapping
        mapping
    points: NDArray[float64]
        candidate points
    parameters: NDArray[float64]
        additional mapping parameters
    tolerance: float, default=1.0E-9
        comparison tolerance

    Returns
    -------
    NDArray[bool_]
        prime-period mask

    """
    points = numpy.asarray(points, dtype=float64)
    test = prime(
        mapping,
        order=order,
        rtol=tolerance,
        atol=tolerance,
    )
    return numpy.asarray([test(point, parameters) for point in points], dtype=bool_)


def canonize(
    orbit:NDArray[float64],
    tolerance:float=1.0E-9,
    reverse:bool=True,
) -> NDArray[float64]:
    """
    Select a canonical starting point for a periodic chain

    Parameters
    ----------
    orbit: NDArray[float64]
        periodic chain
    tolerance: float, default=1.0E-9
        comparison tolerance
    reverse: bool, default=True
        flag to include reversed rotations

    Returns
    -------
    NDArray[float64]
        canonical chain

    """
    length, _ = orbit.shape
    rotations = numpy.empty(((1 + int(reverse))*length, *orbit.shape), dtype=float64)
    local = numpy.copy(orbit)
    for i in range(length):
        rotations[i] = local
        local = numpy.roll(local, -1, axis=0)
    if reverse:
        local = numpy.flip(orbit, axis=0)
        for i in range(length):
            rotations[i + length] = local
            local = numpy.roll(local, -1, axis=0)
    size, *_ = rotations.shape
    flat = rotations.reshape(size, -1)
    keys = numpy.round(flat/tolerance).astype(int64)
    indices = numpy.lexsort(keys.T[::-1])
    return rotations[indices[0]]


def unique(
    order:int,
    mapping:Mapping,
    points:NDArray[float64],
    parameters:NDArray[float64],
    tolerance:float=1.0E-9,
    reverse:bool=True,
) -> NDArray[bool_]:
    """
    Create a unique-periodic-chain mask

    Parameters
    ----------
    order: int
        fixed-point order
    mapping: Mapping
        mapping
    points: NDArray[float64]
        fixed points
    parameters: NDArray[float64]
        additional mapping parameters
    tolerance: float, default=1.0E-9
        duplicate tolerance
    reverse: bool, default=True
        flag to identify reversed chains

    Returns
    -------
    NDArray[bool_]
        unique-chain mask

    """
    points = numpy.asarray(points, dtype=float64)
    if not len(points):
        return numpy.empty(0, dtype=bool_)
    generate = chain(order, mapping)
    starts = numpy.stack([canonize(generate(point, parameters), tolerance=tolerance, reverse=reverse).reshape(-1) for point in points])
    keep = numpy.ones(len(starts), dtype=bool_)
    threshold = tolerance*tolerance
    for i in range(len(starts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(starts)):
            delta = starts[i] - starts[j]
            if numpy.dot(delta, delta) <= threshold:
                keep[j] = False
    return keep


def search(
    order:int,
    mapping:Mapping,
    bounds:NDArray[float64],
    pieces:NDArray[int64],
    parameters:NDArray[float64],
    overlap:float=0.10,
    tolerance:float=1.0E-12,
    filtering:float=1.0E-9,
    duplicate:float=1.0E-9,
    ncube:int=32,
    nball:int=64,
    max_iterations:int=128,
    max_expansions:int=16,
    reverse:bool=True,
    parallel:bool=False,
    roots:Optional[NDArray[float64]]=None,
    powers:Optional[NDArray[int64]]=None,
    alpha:float=1.0E-4,
    epsilon:float=1.0E-16,
) -> NDArray[float64]:
    """
    Search and filter prime-period fixed points using CBM

    Parameters
    ----------
    order: int
        fixed-point order
    mapping: Mapping
        scalar-state Domain mapping
    bounds: NDArray[float64]
        bounds shaped ``(dimension, 2)``
    pieces: NDArray[int64]
        number of tiles along each dimension
    parameters: NDArray[float64]
        additional mapping parameters
    overlap: float, default=0.10
        tile overlap fraction
    tolerance: float, default=1.0E-12
        CBM residual tolerance
    filtering: float, default=1.0E-9
        prime-period comparison tolerance
    duplicate: float, default=1.0E-9
        periodic-chain duplicate tolerance
    ncube: int, default=32
        grid points per tile dimension
    nball: int, default=64
        random sphere points
    max_iterations: int, default=128
        maximum characteristic-bisection iterations
    max_expansions: int, default=16
        maximum initial-polyhedron expansions
    reverse: bool, default=True
        flag to identify reversed chains
    parallel: bool, default=False
        vector mapping parallel-evaluation flag
    roots: Optional[NDArray[float64]], default=None
        known roots to deflate
    powers: Optional[NDArray[int64]], default=None
        root multiplicities
    alpha: float, default=1.0E-4
        deflation offset
    epsilon: float, default=1.0E-16
        deflation stabilization

    Returns
    -------
    NDArray[float64]
        one representative per unique prime-period chain

    """
    bounds = numpy.asarray(bounds, dtype=float64)
    pieces = numpy.asarray(pieces, dtype=int64)
    vector = vectorize(mapping, parallel=parallel)
    residual = problem(
        vector,
        order=order,
        roots=roots,
        powers=powers,
        alpha=alpha,
        epsilon=epsilon,
    )
    boxes = tiles(bounds, pieces, overlap=overlap)
    tests, points, _ = psolve(
        edges(len(bounds)),
        boxes,
        residual,
        parameters,
        tolerance=tolerance,
        ncube=ncube,
        nball=nball,
        max_iterations=max_iterations,
        max_expansions=max_expansions,
    )
    points = points[tests.astype(bool_)]
    if not len(points):
        return numpy.empty((0, len(bounds)), dtype=float64)
    points = points[
        exact(order, mapping, points, parameters, tolerance=filtering)]
    if not len(points):
        return numpy.empty((0, len(bounds)), dtype=float64)
    return points[unique(order, mapping, points, parameters, tolerance=duplicate, reverse=reverse)]


def monodromy(
    order:int,
    mapping:Mapping,
    difference:float=1.0E-6,
) -> Mapping:
    """
    Generate a finite-difference monodromy-matrix callable.

    Parameters
    ----------
    order: int
        fixed-point order
    mapping: Mapping
        mapping
    difference: float, default=1.0E-6
        central finite-difference step

    Returns
    -------
    Mapping

    """
    fixed = power(order, mapping)
    def closure(
        point:NDArray[float64],
        parameters:NDArray[float64],
    ) -> NDArray[float64]:
        point = numpy.asarray(point, dtype=float64)
        dimension = len(point)
        matrix = numpy.empty((dimension, dimension), dtype=float64)
        for i in range(dimension):
            step = difference*max(1.0, abs(point[i]))
            delta = numpy.zeros(dimension, dtype=float64)
            delta[i] = step
            matrix[:, i] = (fixed(point + delta, parameters) - fixed(point - delta, parameters))/(2.0*step)
        return matrix
    return closure


def combine(
    values:NDArray[complex128],
    vectors:NDArray[complex128],
    tolerance:float=1.0E-9,
) -> Tuple[NDArray[complex128], NDArray[complex128]]:
    """
    Combine reciprocal monodromy eigenvalues and eigenvectors

    Parameters
    ----------
    values: NDArray[complex128]
        eigenvalues
    vectors: NDArray[complex128]
        eigenvectors stored in columns
    tolerance: float, default=1.0E-9
        reciprocal pairing tolerance

    Returns
    -------
    Tuple[NDArray[complex128], NDArray[complex128]]
        grouped eigenvalues and eigenvectors

    """
    values = numpy.asarray(values, dtype=complex128)
    vectors = numpy.asarray(vectors, dtype=complex128)
    available = list(range(len(values)))
    groups = []
    while available:
        i = available.pop(0)
        errors = numpy.asarray([
            abs(values[i]*values[j] - 1.0)
            for j in available
        ])
        local = int(numpy.argmin(errors))
        j = available.pop(local)
        groups.append((i, j))
    indices = numpy.asarray(groups, dtype=int64)
    return values[indices], vectors.T[indices]


def classify(
    pairs:NDArray[complex128],
    tolerance:float=1.0E-9,
) -> NDArray[bool_]:
    """
    Classify reciprocal eigenvalue pairs

    Parameters
    ----------
    pairs: NDArray[complex128]
        reciprocal eigenvalue pairs
    tolerance: float, default=1.0E-9
        unit-circle tolerance

    Returns
    -------
    NDArray[bool_]
        True for elliptic pairs and False for hyperbolic pairs

    """
    return numpy.all(numpy.abs(numpy.abs(pairs) - 1.0) < tolerance, axis=-1)


def identify(
    pairs:NDArray[complex128],
    tolerance:float=1.0E-9,
) -> str:
    """
    Generate a fixed-point stability label

    Parameters
    ----------
    pairs: NDArray[complex128]
        reciprocal eigenvalue pairs
    tolerance: float, default=1.0E-9
        unit-circle tolerance

    Returns
    -------
    str
        label such as ``H``, ``H-E``, ``H-H``, or ``H-H-H``

    """
    kinds = ["E" if elliptic else "H" for elliptic in classify(pairs, tolerance=tolerance)]
    kinds.sort(key=lambda kind: kind == "E")
    return "-".join(kinds)


def manifold(
    values:NDArray[complex128],
    tolerance:float=1.0E-9,
) -> NDArray[bool_]:
    """
    Select stable eigenvalues

    Parameters
    ----------
    values: NDArray[complex128]
        grouped eigenvalues
    tolerance: float, default=1.0E-9
        unit-circle tolerance

    Returns
    -------
    NDArray[bool_]
        stable eigenvalue mask

    """
    return numpy.abs(values) - 1.0 < tolerance
