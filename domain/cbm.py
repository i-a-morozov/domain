"""
CBM
---

CBM fixed point estimation
M. N. Vrahatis, An Efficient Method for Locating and Computing Periodic Orbits of Nonlinear Mappings, 1995

Code adapted from xsuite/xnlbd
D. E. Veres et al., Xnlbd: a New Python Package for the Analysis of Non-linear Beam Dynamics Phenomena

I.M., 2025-2026

"""
from typing import Tuple
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from numba import njit
from numba import prange
from numba.typed import List as NDList


@njit(cache=False)
def solve(
    edges: NDArray[np.int64],
    bounds: NDArray[np.float64],
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
    tolerance: float = 1.0E-12,
    ncube: int = 32,
    nball: int = 64,
    max_iterations: int = 128,
    max_expansions: int = 16
) -> Tuple[bool, NDArray[np.float64], int]:
    """
    Estimate fixed point

    Parameters
    ----------
    edges : NDArray[np.int64]
        hypercube edges
    bounds : NDArray[np.float64]
        [..., [min, max], ...] bounds for each dimension
    function : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        fixed point function
    parameters : NDArray[np.float64]
        additional function parameters
    tolerance : float, default=1.0E-12
        fixed point estimation target tolerance
    ncube : int, default=32
        number of grid points per dimension
    nball : int, default=64
        number of sphere random points
    max_iterations : int, default=128
        maximum number of generalised-bisection steps
    max_expansions : int, default=16
        maximum number of sphere-radius expansions (initial polyhedron construction)

    Returns
    -------
    test : bool
        True if a fixed-point estimate meeting the tolerance was found
    point : NDArray[np.float64]
        the fixed-point estimate
    count : np.int64
        number of bisection steps performed

    """
    polygon = proper_polygon(ncube, bounds, nball, function, parameters, max_expansions)
    if np.isnan(np.sum(polygon)):
        return False, np.full(len(bounds), np.nan, dtype=np.float64), 0
    test, point = check(polygon, tolerance, function, parameters)
    iterations = 0
    while (not test) and (iterations < max_iterations):
        changed, polygon = bisection(edges, polygon, function, parameters)
        test, point = check(polygon, tolerance, function, parameters)
        iterations += 1
        if (not changed) and (not test):
            break
    if not test:
        point = np.full(len(bounds), np.nan, dtype=np.float64)
    return test, point, iterations


@njit(parallel=True, cache=False)
def psolve(
    edges: NDArray[np.int64],
    tiles: NDArray[np.float64],
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
    tolerance: float = 1.0E-12,
    ncube: int = 32,
    nball: int = 64,
    max_iterations: int = 128,
    max_expansions: int = 16
) -> Tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.int64]]:
    """
    Estimate fixed point

    Parameters
    ----------
    edges : NDArray[np.int64]
        hypercube edges
    tiles : NDArray[np.float64]
        array of bounds
    function : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        fixed point function
    parameters : NDArray[np.float64]
        additional function parameters
    tolerance : float, default=1.0E-12
        fixed point estimation target tolerance
    ncube : int, default=32
        number of grid points per dimension
    nball : int, default=64
        number of sphere random points
    max_iterations : int, default=128
        maximum number of generalised-bisection steps
    max_expansions : int, default=16
        maximum number of sphere-radius expansions (initial polyhedron construction)

    Returns
    -------
    tests : NDArray[np.uint8]
        True if a fixed-point estimate meeting the tolerance was found
    points : NDArray[np.float64]
        the fixed-point estimate
    counts : NDArray[np.int64]
        number of bisection steps performed

    """
    m, n, _ = tiles.shape
    tests = np.zeros(m, dtype=np.uint8)
    points = np.full((m, n), np.nan, dtype=np.float64)
    counts = np.zeros(m, dtype=np.int64)
    for i in prange(m):
        test, point, count = solve(
            edges,
            tiles[i],
            function,
            parameters,
            tolerance=tolerance,
            ncube=ncube,
            nball=nball,
            max_iterations=max_iterations,
            max_expansions=max_expansions
        )
        if test:
            tests[i] = 1
            for j in range(n):
                points[i, j] = point[j]
        counts[i] = count
    return tests, points, counts


@njit
def tiles(
    bounds: NDArray[np.float64],
    pieces: NDArray[np.int64],
    overlap: float = 0.10
) -> NDArray[np.float64]:
    """
    Split the hyperbox into overlapping tiles

    Parameters
    ----------
    bounds : NDArray[np.float64]
        [..., [min, max], ...] bounds for each dimension
    pieces : NDArray[np.int64]
        number of tiles per dimension
    overlap : float, default=0.10
        fraction of tile length per axis to extend on both sides

    Returns
    -------
    tiles : NDArray[np.float64]

    """
    dimension = len(bounds)
    total = 1
    for i in range(dimension):
        total *= pieces[i]
    tiles = np.empty((total, dimension, 2), dtype=np.float64)
    stride = np.empty(dimension, dtype=np.int64)
    stride[dimension - 1] = 1
    for i in range(dimension - 2, -1, -1):
        stride[i] = stride[i + 1]*pieces[i + 1]
    for i in range(total):
        for j in range(dimension):
            idx = (i // stride[j]) % pieces[j]
            lb = bounds[j, 0]
            ub = bounds[j, 1]
            length = ub - lb
            step = length/pieces[j]
            a = lb + idx*step
            b = a + step
            pad = overlap*step
            aa = a - pad
            bb = b + pad
            if aa < lb:
                aa = lb
            if bb > ub:
                bb = ub
            tiles[i, j, 0] = aa
            tiles[i, j, 1] = bb
    return tiles


@njit
def binary_coefficients(
    n: int,
    size: int
) -> NDArray[np.int64]:
    """
    Return the binary coefficients (LSB-first) of a non-negative integer up to some length

    Parameters
    ----------
    n: int, non-negative
        non-negative integer
    size : int
        number of bits to return

    Returns
    -------
    coefficients : NDArray[np.int64] (length, )
        array of binary coefficients (LSB-first)

    """
    bits = np.empty(size, dtype=np.int64)
    m = n
    for idx in range(size):
        bits[idx] = m & 1
        m >>= 1
    return bits


@njit
def complete_matrix(
    dimension: int
) -> NDArray[np.int64]:
    """
    Construct the complete sign matrix for the characteristic n-polyhedron

    Parameters
    ----------
    dimension : int, positive
        dimension

    Returns
    -------
    matrix : NDArray[np.int64] (2**dimension, dimension)
        characteristic n-polyhedron sign matrix

    """
    nvertices = 1 << dimension
    matrix = np.empty((nvertices, dimension), dtype=np.int64)
    for i in range(nvertices):
        bits = binary_coefficients(i, dimension)
        for j in range(dimension):
            matrix[i, j] = 2*bits[dimension - 1 - j] - 1
    return matrix


@njit
def neighbour_indices(
    idx: int,
    dimension: int
) -> NDArray[np.int64]:
    """
    Return the vertex indices (1-based) that are neighbours of a given vertex

    Parameters
    ----------
    idx : int, positive
        1-based vertex index
    dimension : int, positive
        dimension

    Returns
    -------
    neighbors : NDArray[np.int64], (dimension, )
        neighbour indices (1-based)

    """
    neighbors = np.empty(dimension, dtype=np.int64)
    for i in range(dimension):
        neighbors[i] = 1 + ((idx - 1) ^ (1 << (dimension - 1 - i)))
    return neighbors


@njit
def edges(
    dimension: int
) -> NDArray[np.int64]:
    """
    Return unique hypercube edges (1-based vertex indices)

    Parameters
    ----------
    dimension : int, positive
        dimension

    Returns
    -------
    edges : NDArray[np.int64], (dimension * 2**(dimension - 1), 2)
        hypercube edges, each row [u, v] with u < v (1-based)

    """
    total = 1 << dimension
    nedges = dimension*(1 << (dimension - 1))
    edges = np.empty((nedges, 2), dtype=np.int64)
    idx = 0
    for u in range(1, total + 1):
        neighbours = neighbour_indices(u, dimension)
        for k in range(dimension):
            v = neighbours[k]
            if u < v:
                edges[idx, 0] = u
                edges[idx, 1] = v
                idx += 1
    return edges


@njit
def diagonal(
    polygon: NDArray[np.float64]
) -> NDArray[np.int64]:
    """
    Find the longest antipodal diagonal of a characteristic n-polyhedron

    Parameters
    ----------
    polygon :  NDArray[np.float64], (2**n, n)
        coordinates of the hypercube vertices (rows are vertices)

    Returns
    -------
    (i, j) :  NDArray[np.int64]
        pair of 1-based vertex indices (i, j)

    """
    vertices, n = polygon.shape
    mask = (1 << n) - 1
    vertex_i = 1
    vertex_j = 1
    distance_max = -1.0
    for i in range(vertices):
        j = (~i) & mask
        if j <= i:
            continue
        distance = 0.0
        for k in range(n):
            delta = polygon[i, k] - polygon[j, k]
            distance += delta * delta
        if distance > distance_max:
            distance_max = distance
            vertex_i = i + 1
            vertex_j = j + 1
    return np.array((vertex_i, vertex_j), dtype=np.int64)


@njit
def sign(
    x: NDArray[np.float64]
) -> NDArray[np.int64]:
    """
    Return elementwise signs of the input array

    Parameters
    ----------
    x : NDArray[np.float64]
        input array

    Returns
    -------
    NDArray[np.int64]
        signs

    """
    flat = x.ravel()
    mask = np.empty(flat.size, dtype=np.int64)
    for i in range(flat.size):
        mask[i] = 1 if flat[i] >= 0.0 else -1
    return mask.reshape(x.shape)


@njit
def grid(
    count: int,
    bounds: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], np.float64]:
    """
    Generate an n-D Cartesian flattened grid and compute the diagonal step radius

    Parameters
    ----------
    count : int, positive
        number of evenly spaced points per dimension
    bounds : NDArray[np.float64]
        [..., [min, max], ...] bounds for each dimension

    Returns
    -------
    points : NDArray[np.float64], (n, count**n)
        grid points
    radius : np.float64
        diagonal step radius

    """
    dimension = len(bounds)
    starts = np.empty(dimension, dtype=np.float64)
    steps  = np.empty(dimension, dtype=np.float64)
    for i in range(dimension):
        lb, ub = bounds[i]
        starts[i] = lb
        steps[i] = (ub - lb)/(count - 1)
    radius = np.float64(0.0)
    for i in range(dimension):
        radius += steps[i] * steps[i]
    radius = np.sqrt(radius)
    powers = np.empty(dimension + 1, dtype=np.int64)
    powers[0] = 1
    for i in range(1, dimension + 1):
        powers[i] = powers[i - 1]*count
    total = powers[dimension]
    points = np.empty((dimension, total), dtype=np.float64)
    for i in range(dimension):
        axis = (dimension > 1)*(i + (i < 2)*(1 - 2*i))
        block = powers[dimension - axis - 1]
        groups = powers[axis]
        column = 0
        for _ in range(groups):
            value = starts[i]
            for _ in range(count):
                for j in range(block):
                    points[i, column + j] = value
                column += block
                value += steps[i]
    return points, radius


@njit
def change(
    signs: NDArray[np.int64],
    shape: NDArray[np.int64],
) -> NDList[NDArray[np.int64]]:
    """
    Detect sign-change locations along all spatial axes for each component

    Parameters
    ----------
    signs : NDArray[np.int64]
        flat grid signs
    shape : Tuple[int, ...]
        original shape

    Returns
    -------
    changes : NDList[NDArray[np.int64]]
        1D array of flattened indices where sign changes occur for each component

    """
    dimension = len(shape)
    changes = NDList()
    total = 1
    for i in range(dimension):
        total *= shape[i]
    axis = np.empty(dimension, dtype=np.int64)
    sizes = np.empty(dimension, dtype=np.int64)
    strides = np.empty(dimension, dtype=np.int64)
    dimensions = np.empty(dimension, dtype=np.int64)
    for i in range(dimension):
        axis[i] = (dimension > 1)*(i + (i < 2)*(1 - 2*i))
        sizes[i] = shape[axis[i]]
    increment = 1
    for i in range(dimension - 1, -1, -1):
        strides[i] = increment
        increment *= sizes[i]
    for i in range(dimension):
        dimensions[i] = strides[axis[i]]
    for component in range(dimension):
        data = signs[component]
        mask = np.zeros(total, dtype=np.uint8)
        for i in range(dimension):
            stride = dimensions[i]
            length = shape[i]
            for start in range(0, total, stride*length):
                current = start + stride
                for _ in range(1, length):
                    previous = current - stride
                    for offset in range(stride):
                        if data[previous + offset] != data[current + offset]:
                            mask[previous + offset] = 1
                    current += stride
        count = 0
        for i in range(total):
            if mask[i] != 0:
                count += 1
        idxs = np.empty(count, dtype=np.int64)
        count = 0
        for i in range(total):
            if mask[i] != 0:
                idxs[count] = i
                count += 1
        changes.append(idxs)
    return changes


@njit
def intersections(
    grid: NDArray[np.float64],
    changes: NDList[NDArray[np.int64]]
) -> NDArray[np.float64]:
    """
    Return the coordinate on the grid at which the all components change sign

    Parameters
    ----------
    grid : NDArray[np.float64]
        grid points
    changes : List[NDArray[np.int64]]
        1D array of flattened indices where sign changes occur for each component

    Returns
    -------
    NDArray[np.float64]
        array of coordinates of intersection point in each dimension

    """
    n, total = grid.shape
    counts = np.zeros(total, dtype=np.int64)
    for i in range(n):
        change = changes[i]
        for j in range(len(change)):
            idx = change[j]
            counts[idx] += 1
    ni = 0
    for i in range(total):
        if counts[i] == n:
            ni += 1
    if ni == 0:
        return np.empty(0, dtype=np.float64)
    idxs = np.empty(ni, dtype=np.int64)
    count = 0
    for i in range(total):
        if counts[i] == n:
            idxs[count] = i
            count += 1
    point = np.empty(n, dtype=np.float64)
    factor = 1.0/ni
    for i in range(n):
        s = 0.0
        for j in range(ni):
            s += grid[i, idxs[j]]
        point[i] = factor*s
    return point


@njit
def sphere_polygon(
    count: int,
    origin: NDArray[np.float64],
    radius: float,
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Generate an initial proper polygon by sampling points on an n-dimensional sphere

    Parameters
    ----------
    count : int
        number of sample points
    origin : NDArray[np.float64]
        origin
    radius : float
        radius
    function : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        fixed point function
    parameters : NDArray[np.float64]
        additional function parameters

    Returns
    -------
    NDArray[np.float64], (2**n, n)
        proper polygon

    """
    dimension = len(origin)
    points = np.empty((dimension, count), dtype=np.float64)
    buffer = np.empty(dimension, dtype=np.float64)
    for j in range(count):
        norm = 0.0
        for i in range(dimension):
            rnd = np.random.normal()
            buffer[i] = rnd
            norm += rnd*rnd
        scale = radius / np.sqrt(norm)
        for i in range(dimension):
            points[i, j] = origin[i] + scale*buffer[i]
    values = sign(function(points, parameters))
    matrix = complete_matrix(dimension)
    vertices = np.nan*np.empty(matrix.shape)
    for i in range(2**dimension):
        for j in range(count):
            if np.all(matrix[i, :] == values[:, j]):
                vertices[i, :] = points[:, j]
                break
    return vertices


@njit
def proper_polygon(
    count: int,
    bounds: NDArray[np.float64],
    ns: int,
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
    limit: int = 64,
    factor: float=0.5*(1.0 + 5.0**0.5)
) -> NDArray[np.float64]:
    """
    Construct a proper n-polyhedron

    Parameters
    ----------
    count : int, positive
        number of evenly spaced points per dimension
    bounds : NDArray[np.float64]
        [..., [min, max], ...] bounds for each dimension
    ns : int, positive
        number of random samples on the sphere
    function : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        fixed point function
    parameters : NDArray[np.float64]
        additional function parameters
    limit : int, default=64
        maximum expansion attempts
    factor : float, default=0.5*(1.0 + 5.0**0.5)
        expansion factor

    Returns
    -------
    NDArray[np.float64], (2**n, n)
        proper polygon

    """
    dimension = len(bounds)
    points, radius = grid(count, bounds)
    shape = np.empty(dimension, dtype=np.int64)
    for i in range(dimension):
        shape[i] = count
    changes = change(sign(function(points, parameters)), shape)
    origin = intersections(points, changes)
    if origin.size == 0:
        return np.full((2**dimension, dimension), np.nan)
    vertices = sphere_polygon(ns, origin, radius, function, parameters)
    i = 0
    while i < limit:
        retry = False
        for vertex in range(len(vertices)):
            if np.isnan(vertices[vertex, 0]):
                retry = True
                break
        if not retry:
            break
        radius = radius*factor
        vertices = sphere_polygon(ns, origin, radius, function, parameters)
        i += 1
    return vertices


@njit
def check(
    polygon: NDArray[np.float64],
    tolerance: float,
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64],
) -> Tuple[bool, NDArray[np.float64]]:
    """
    Check accuracy (midpoint of the longest diagonal)

    Parameters
    ----------
    polygon : NDArray[np.float64]
        proper polygon
    tolerance : float
        tolerance
    function : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        fixed point function
    parameters : NDArray[np.float64]
        additional function parameters

    Returns
    -------
    Tuple[bool, NDArray[np.float64]]

    """
    i, j = diagonal(polygon) - 1
    point = 0.5*(polygon[i] + polygon[j]).reshape(-1, 1)
    value = function(point, parameters).ravel()
    success = True
    for i in range(len(value)):
        if np.abs(value[i]) > tolerance:
            success = False
            break
    return success, point.ravel()


@njit
def bisection(
    edges: NDArray[np.int64],
    initial: NDArray[np.float64],
    function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    parameters: NDArray[np.float64]
) -> Tuple[bool, NDArray[np.float64]]:
    """
    Perform one characteristic bisection step

    Parameters
    ----------
    edges : NDArray[np.int64]
        hypercube edges
    initial : NDArray[np.float64]
        initial characteristic polyhedron
    function : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        fixed point function
    parameters : NDArray[np.float64]
        additional function parameters

    Returns
    -------
    changed : bool
        True if at least one vertex was replaced
    polygon : NDArray[np.float64], (2**n, n)
        updated characteristic  polyhedron
    """
    nvertices, dimension = initial.shape
    polygon = np.copy(initial)
    cached_sign = np.zeros((nvertices, dimension), dtype=np.int8)
    cached_work = np.zeros(nvertices, dtype=np.uint8)
    buffer_i = np.empty(dimension, dtype=np.int8)
    buffer_j = np.empty(dimension, dtype=np.int8)
    buffer_k = np.empty(dimension, dtype=np.int8)
    x_center = np.empty(dimension, dtype=np.float64)
    x_single  = np.empty((dimension, 1), dtype=np.float64)
    def copy_sign(idx: int, out: NDArray[np.int8]) -> None:
        for i in range(dimension):
            out[i] = cached_sign[idx, i]
    def evaluate(x: NDArray[np.float64], out: NDArray[np.int8]) -> None:
        for i in range(dimension):
            x_single[i, 0] = x[i]
        values = function(x_single, parameters)
        for i in range(dimension):
            out[i] = 1 if values[i, 0] >= 0.0 else -1
    def cached(idx: int) -> None:
        if cached_work[idx] == 0:
            for i in range(dimension):
                x_single[i, 0] = polygon[idx, i]
            values = function(x_single, parameters)
            for i in range(dimension):
                cached_sign[idx, i] = 1 if values[i, 0] >= 0.0 else -1
            cached_work[idx] = 1
    replaced = 0
    for e in range(len(edges)):
        i = edges[e, 0] - 1
        j = edges[e, 1] - 1
        ci = cached_work[i] == 1
        cj = cached_work[j] == 1
        if not ci and not cj:
            x = np.empty((dimension, 2), dtype=np.float64)
            for k in range(dimension):
                x[k, 0] = polygon[i, k]
                x[k, 1] = polygon[j, k]
            v = function(x, parameters)
            for k in range(dimension):
                cached_sign[i, k] = 1 if v[k, 0] >= 0.0 else -1
                cached_sign[j, k] = 1 if v[k, 1] >= 0.0 else -1
            cached_work[i] = 1
            cached_work[j] = 1
        else:
            if not ci:
                cached(i)
            if not cj:
                cached(j)
        for k in range(dimension):
            x_center[k] = 0.5*(polygon[i, k] + polygon[j, k])
        evaluate(x_center, buffer_k)
        copy_sign(i, buffer_i)
        match = True
        for k in range(dimension):
            if buffer_i[k] != buffer_k[k]:
                match = False
                break
        if match:
            for k in range(dimension):
                polygon[i, k] = x_center[k]
            cached_work[i] = 0
            replaced += 1
            continue
        copy_sign(j, buffer_j)
        match = True
        for k in range(dimension):
            if buffer_j[k] != buffer_k[k]:
                match = False
                break
        if match:
            for k in range(dimension):
                polygon[j, k] = x_center[k]
            cached_work[j] = 0
            replaced += 1
            continue
        reflections = 0
        while reflections < 2:
            u_idx = -1
            for v in range(nvertices):
                if v == i or v == j:
                    continue
                cached(v)
                equal = True
                for k in range(dimension):
                    if cached_sign[v, k] != buffer_k[k]:
                        equal = False
                        break
                if equal:
                    u_idx = v
                    break
            if u_idx == -1:
                break
            for k in range(dimension):
                x_center[k] = 2.0*x_center[k] - polygon[u_idx, k]
            evaluate(x_center, buffer_k)
            copy_sign(i, buffer_i)
            equal = True
            for k in range(dimension):
                if buffer_i[k] != buffer_k[k]:
                    equal = False
                    break
            if equal:
                for k in range(dimension):
                    polygon[i, k] = x_center[k]
                cached_work[i] = 0
                replaced += 1
                break
            copy_sign(j, buffer_j)
            equal = True
            for k in range(dimension):
                if buffer_j[k] != buffer_k[k]:
                    equal = False
                    break
            if equal:
                for k in range(dimension):
                    polygon[j, k] = x_center[k]
                cached_work[j] = 0
                replaced += 1
                break
            reflections += 1
    return (replaced > 0), polygon
