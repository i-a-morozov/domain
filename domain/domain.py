"""
Domain class

"""
from typing import Iterable
from typing import Tuple

from dataclasses import dataclass

from numpy import bool_
from numpy import int64
from numpy import float64
from numpy.typing import NDArray
import numpy

from numba import njit
from numba import prange

from domain.volume import rays

@dataclass
class Domain:
    """
    Phase space domain specification and sparse occupancy tracking

    The domain is discretized into a uniform Cartesian grid (Fortran-style / column-major linearization)
    Occupancy is stored sparsely as a sorted, unique 1D array of flattened cell indices

    Parameters
    ----------
    lb : NDArray[float64]
        lower bounds per dimension, shape (dimension, )
    ub : NDArray[float64]
        upper bounds per dimension, shape (dimension, )
    cell : float
        uniform cell size along all dimensions

    Attributes
    ----------
    origin : NDArray[float64]
        grid origin (lower corner used for binning)
    counts : NDArray[int64]
        number of cells along each dimension (Fortran order)
    strides : NDArray[int64]
        fortran-style multi-index factors
    total : int
        total number of cells in the full grid
    keys : NDArray[int64]
        sorted, unique flattened indices of filled cells

    Properties
    ----------
    dimension : int
        phase-space dimension
    ratio : float
        fill ratio (number of filled cells / total)
    size : int
        number of filled cells
    list : NDArray[int64]
        alias for `keys`
    construct : NDArray[float64]
        centers of filled cells, shape (size, dimension)

    Methods
    -------
    index(points)
        map points to flattened grid cell indices
    convert(ids)
        convert ids to multi-index 
    outer(ids)
        check whether given id corresponds to a boundary cell
    transform(ids)
        map flattened grid cell indices to cell-center coordinates
    insert(ids)
        Insert a sequence of ids into the sparse set (kept sorted/unique)
    update(points)
        insert points by binning to ids first
    volume(n, m, center)
        compute equivalent hypersphere radius and hypervolume via ray quadrature
    boundary(n, m, center)
        return first-hit occupied cell ids for rays (one id per ray, -1 if none)

    """
    lb: NDArray[float64]
    ub: NDArray[float64]
    cell: float
    origin: NDArray[float64] = None
    counts: NDArray[int64] = None
    strides: NDArray[int64] = None
    total: int = 0
    keys: NDArray[int64] = None

    def __post_init__(self):
        lb = numpy.asarray(self.lb, dtype=float64)
        ub = numpy.asarray(self.ub, dtype=float64)
        self.origin, self.counts, self.strides, self.total = box(lb, ub, float(self.cell))
        self.keys = numpy.zeros((0, ), dtype=int64)

    @property
    def dimension(self) -> int:
        return self.origin.size

    @property
    def ratio(self) -> float:
        return len(self.keys)/self.total

    def index(self, points:NDArray[float64]) -> NDArray[int64]:        
        points = numpy.ascontiguousarray(points, dtype=float64)
        return index(points, self.origin, self.counts, self.strides, float(self.cell))

    def convert(self, keys:NDArray[int64]) -> NDArray[float64]:
        keys = numpy.ascontiguousarray(keys, dtype=int64)
        return convert(keys, self.counts, self.strides)

    def outer(self, keys:NDArray[int64]) -> NDArray[bool_]:
        keys = numpy.ascontiguousarray(keys, dtype=int64)
        return outer(keys, self.counts, self.strides)

    def transform(self, keys:NDArray[int64]) -> NDArray[float64]:
        keys = numpy.ascontiguousarray(keys, dtype=int64)
        return transform(keys, self.origin, self.counts, self.strides, float(self.cell))

    def insert(self, keys:Iterable[int]) -> None:
        self.keys = numpy.union1d(self.keys, keys)

    def update(self, points:NDArray[float64]) -> None:
        keys = self.index(points)
        self.insert(keys)

    @property
    def size(self) -> int:
        return len(self.keys)

    def __len__(self) -> int:
        return self.size
    
    @property
    def list(self) -> NDArray[int64]:
        return self.keys

    @property
    def construct(self) -> NDArray[float64]:
        return self.transform(self.list)

    def volume(self, n:int, m:int, center:NDArray[float64], directions:NDArray[float64], factors:NDArray[float64]) -> float:
        return volume(self.dimension, n, m, self.origin, self.counts, self.strides, self.cell, center, directions, factors, self.list)
    
    def boundary(self, n:int, m:int, center:NDArray[float64], directions:NDArray[float64]) -> Tuple[NDArray[int64], NDArray[float64], NDArray[float64]]:
        return boundary(self.dimension, n, m, self.origin, self.counts, self.strides, self.cell, center, directions, self.list)

@njit
def cumprod(
    counts:NDArray[int64]
) -> NDArray[int64]:
    """
    Compute cumulative product

    Parameters
    ----------
    counts:NDArray[int64]
        input array

    Returns
    -------
    NDArray[int64]
    
    """
    dimension = len(counts)
    strides = numpy.empty(dimension, dtype=int64)
    stride = 1
    for i in range(dimension):
        strides[i] = stride
        stride *= counts[i]
    return strides


@njit
def box(
    lb:NDArray[float64],
    ub:NDArray[float64],
    cell:float
) -> Tuple[NDArray[float64], NDArray[int64], NDArray[int64], int64]:
    """
    Define hyperbox from given lower/upper bounds and cell size

    Parameters
    ----------
    lb: NDArray[float64]
        low bounds
    ub: NDArray[float64]
        upper bounds    
    cell: float
        bin cell size

    Returns
    -------
    Tuple[NDArray[float64], NDArray[int64], NDArray[int64], int64]
        box origin (lower corner)
        number of cells per dimension
        multi-index factors (Fortran ordering)
        total number of elements
    
    """
    dimension = len(lb)
    origin = numpy.empty(dimension, dtype=float64)
    counts = numpy.empty(dimension, dtype=int64)
    for i in range(dimension):
        origin[i] = lb[i] - 0.5*cell
        extent = ub[i] - origin[i]
        if extent < 0.0:
            extent = 0.0
        n = numpy.ceil(extent/cell)
        counts[i] = n if n >= 1 else 1
    strides = cumprod(counts)
    total = 1
    for i in range(dimension):
        total *= counts[i]
    return origin, counts, strides, total


@njit(parallel=True)
def grid(
    origin:NDArray[float64],
    counts:NDArray[int64],
    strides:NDArray[int64],
    cell:float
) -> NDArray[float64]:
    """
    Generate full grid (center of cells)

    Parameters
    ----------
    origin: NDArray[float64]
        box origin (lower corner)
    counts: NDArray[int64]
        number of cells per dimension
    strides: NDArray[int64]
        multi-index factors (Fortran ordering)
    cell: float
        cell size

    Returns
    -------
    NDArray[float64], (total, dimensions)
    
    """
    dimension = len(counts)
    total = 1
    for i in range(dimension):
        total *= counts[i]
    centers = numpy.empty((total, dimension), dtype=float64)
    for i in prange(total):
        for j in range(dimension):
            ij = (i // strides[j]) % counts[j]
            centers[i, j] = origin[j] + (ij + 0.5)*cell
    return centers


@njit(parallel=True)
def index(points:NDArray[float64],
          origin:NDArray[float64],
          counts:NDArray[int64],
          strides:NDArray[int64],
          cell:float
) -> NDArray[int64]:
    """
    Index points (indexing starts with one)

    Parameters
    ----------
    points:NDArray[float64]
        input points
    origin: NDArray[float64]
        box origin (lower corner)
    counts: NDArray[int64]
        number of cells per dimension
    strides: NDArray[int64]
        multi-index factors (Fortran ordering)
    cell: float
        cell size

    Returns
    -------
    NDArray[int64]
    
    """
    length, dimension = points.shape
    ids = numpy.empty(length, dtype=int64)
    factor = 1.0/cell
    for i in prange(length):
        idx = 0
        member = True
        for j in range(dimension):
            key = numpy.floor(factor*(points[i, j] - origin[j]))
            if key < 0 or key >= counts[j]:
                member = False
                break
            idx += key*strides[j]
        ids[i] = idx if member else -1
    return ids


@njit(parallel=True)
def insert(
    mask:NDArray[bool_],
    ids:NDArray[int64]
) -> None:
    """
    Insert mask (inplace)

    Parameters
    ----------
    mask: NDArray[numpy.bool]
        mask to update
    ids: NDArray[int64]
        ids to update

    Returns
    -------
    None
    
    """
    for i in prange(len(ids)):
        mask[ids[i]] = True


@njit(parallel=True)
def convert(
    ids:NDArray[int64],
    counts:NDArray[int64],
    strides:NDArray[int64]
) -> NDArray[int64]:
    """
    Convert ids to multi-index

    Parameters
    ----------
    ids: NDArray[int64]
        ids to transform
    counts: NDArray[int64]
        number of cells per dimension
    strides: NDArray[int64]
        multi-index factors (Fortran ordering)

    Returns
    -------
    NDArray[int64]    

    """
    length = len(ids)
    dimension = len(counts)
    out = numpy.empty((length, dimension), dtype=int64)
    for i in prange(length):
        idx = ids[i]
        for j in range(dimension):
            out[i, j] = (idx // strides[j]) % counts[j]
    return out


@njit(parallel=True)
def outer(
    ids:NDArray[int64],
    counts:NDArray[int64],
    strides:NDArray[int64],
) -> NDArray[bool_]:
    """
    Check whether given id corresponds to a boundary cell

    Parameters
    ----------
    ids: NDArray[int64]
        ids to check
    counts: NDArray[int64]
        number of cells per dimension
    strides: NDArray[int64]
        multi-index factors (Fortran ordering)

    Returns
    -------
    NDArray[bool_]
    
    """
    length = len(ids)
    dimension = len(counts)
    mask = numpy.empty(length, dtype=bool_)
    for i in prange(length):
        idx = ids[i]
        flag = False
        for j in range(dimension):
            k = (idx // strides[j]) % counts[j]
            if k == 0 or k == (counts[j] - 1):
                flag = True
                break
        mask[i] = flag
    return mask


@njit(parallel=True)
def transform(ids:NDArray[int64],
              origin:NDArray[float64],
              counts:NDArray[int64],
              strides:NDArray[int64],
              cell:float
) -> NDArray[float64]:
    """
    Transform ids to grid coordinates

    Parameters
    ----------
    ids: NDArray[int64]
        ids to transform
    origin: NDArray[float64]
        box origin (lower corner)
    counts: NDArray[int64]
        number of cells per dimension
    strides: NDArray[int64]
        multi-index factors (Fortran ordering)
    cell: float
        cell size

    Returns
    -------
    NDArray[float64]
    
    """
    length = len(ids)
    dimension = len(origin)
    out = numpy.empty((length, dimension), dtype=float64)
    for i in prange(length):
        idx = ids[i]
        for j in range(dimension):
            out[i, j] = origin[j] + cell*((idx // strides[j]) % counts[j] + 0.5)
    return out


@njit
def prefix(
    cs:NDArray[int64]
) -> NDArray[int64]:
    out = numpy.empty(cs.size + 1, dtype=int64)
    total = 0
    out[0] = 0
    for i in range(cs.size):
        total += cs[i]
        out[i + 1] = total
    return out


@njit(parallel=True)
def size(
    mask:NDArray[bool_],
    block:int
) -> NDArray[int64]:
    n = mask.size
    nb = (n + block - 1) // block
    counts = numpy.zeros(nb, dtype=int64)
    for b in prange(nb):
        start = b*block
        end = min(n, start + block)
        c = 0
        for i in range(start, end):
            if mask[i]:
                c += 1
        counts[b] = c
    return counts


@njit(parallel=True)
def fill(
    mask:NDArray[bool_],
    block:int,
    offsets:NDArray[int64],
    out:NDArray[int64]
) -> None:
    n = mask.size
    nb = offsets.size - 1
    for b in prange(nb):
        start = b*block
        end = min(n, start + block)
        pos = offsets[b]
        for i in range(start, end):
            if mask[i]:
                out[pos] = i
                pos += 1


@njit
def nonzero(
    mask:NDArray[bool_],
    block:int=10**6
) -> NDArray[int64]:
    """
    Return indices where mask is True
    Count the number of True elements, allocate and fill corresponding array

    Parameters
    ----------
    mask: NDArray[numpy.bool]
        mask
    block: int, default=10**6
        scan block size

    Returns
    -------
    NDArray[int64]

    """
    counts = size(mask, block)
    offsets = prefix(counts)
    out = numpy.empty(offsets[-1], dtype=int64)
    fill(mask, block, offsets, out)
    return out


def construct(mask:NDArray[bool_],
              origin:NDArray[float64],
              counts:NDArray[int64],
              strides:NDArray[int64],
              cell:float
) -> NDArray[float64]:
    """
    Construct True points
    
    Parameters
    ----------
    mask: NDArray[numpy.bool]
        mask    
    origin: NDArray[float64]
        box origin (lower corner)
    counts: NDArray[int64]
        number of cells per dimension
    strides: NDArray[int64]
        multi-index factors (Fortran ordering)
    cell: float
        cell size

    Returns
    -------
    NDArray[float64]
    
    """
    return transform(nonzero(mask), origin, counts, strides, cell)


@njit
def member(
    keys:NDArray[int64],
    value:int64
) -> bool:
    low, high = 0, keys.size
    while low < high:
        center = (low + high) // 2
        key = keys[center]
        if key < value:
            low = center + 1
        else:
            high = center
    return (low < keys.size) and (keys[low] == value)


@njit
def position(indices, strides):
    position = 0
    for i in range(len(indices)):
        position += indices[i]*strides[i]
    return position


@njit
def intersection(
    origin:NDArray[float64],
    counts:NDArray[int64],
    strides:NDArray[int64],
    cell:float,
    start:NDArray[float64],
    direction: NDArray[float64],
    keys:NDArray[int64],
    radius:float
) -> Tuple[int64, float64, NDArray[float64]]:
    """
    Ray traversal through the grid to the first occupied cell (DDA stepping)

    Parameters
    ----------
    origin : NDArray[float64]
        grid origin (lower corner), shape (dimension, ) 
    counts : NDArray[int64]
        number of cells per dimension, shape (dimension, )
    strides : NDArray[int64]
        fortran-style stride factors, shape (dimension, )
    cell : float
        cell size
    start : NDArray[float64]
        ray start point, shape (dimension, )
    direction : NDArray[float64]
        ray direction (assumed normalized), shape (dimension, )
    keys : NDArray[int64]
        sorted, unique occupied cell ids.
    radius : float
        maximum traversal distance along the ray

    Returns
    -------
    Tuple[int64, float64, NDArray[float64]]

    """
    dimension = len(origin)
    point = numpy.empty(dimension, dtype=float64)
    idx = numpy.empty(dimension, dtype=int64)
    for i in range(dimension):
        j = numpy.floor((start[i] - origin[i]) / cell)
        if j < 0:
            j = 0
        if j >= counts[i]:
            j = counts[i] - 1
        idx[i] = int64(j)
    key = position(idx, strides)
    if member(keys, key):
        for k in range(dimension):
            point[k] = start[k]
        return key, 0.0, point
    steps = numpy.empty(dimension, dtype=int64)
    limit = numpy.empty(dimension, dtype=float64)
    delta = numpy.empty(dimension, dtype=float64)
    for i in range(dimension):
        di = direction[i]
        if di > 0.0:
            steps[i] = 1
            limit[i] = ((origin[i] + (idx[i] + 1)*cell) - start[i])/di
            delta[i] = cell / di
        elif di < 0.0:
            steps[i] = -1
            limit[i] = ((origin[i] + idx[i]*cell) - start[i])/di
            delta[i] = -cell/di
        else:
            steps[i] = 0
            limit[i] = numpy.inf
            delta[i] = numpy.inf
    while True:
        axis = 0
        current = limit[0]
        for i in range(1, dimension):
            if limit[i] < current:
                current = limit[i]
                axis = i
        total = current
        if total > radius:
            for k in range(dimension):
                point[k] = start[k] + radius*direction[k]
            return int64(-1), radius, point
        idx[axis] += steps[axis]
        if idx[axis] < 0 or idx[axis] >= counts[axis]:
            for k in range(dimension):
                point[k] = start[k] + total*direction[k]
            return int64(-1), total, point
        limit[axis] += delta[axis]
        key = position(idx, strides)
        if member(keys, key):
            for k in range(dimension):
                point[k] = start[k] + total*direction[k]
            return key, total, point


@njit(parallel=True)
def volume(
    dimension:int,
    n:int,
    m:int,
    origin:NDArray[float64],
    counts:NDArray[int64],
    strides:NDArray[int64],
    cell:float,
    center:NDArray[float64],
    directions:NDArray[float64],
    factors:NDArray[float64],    
    keys:NDArray[int64]
) -> Tuple[float64, float64]:
    """
    Compute equivalent hypersphere radius and hypervolume via ray-grid intersections

    Parameters
    ----------
    dimension : int
        phase space dimension
    n : int
        number of mixing-angle bins
    m : int
        number of in-plane angle bins
    origin : NDArray[float64]
        grid origin (lower corner), shape (dimension, )
    counts : NDArray[int64]
        number of cells per dimension, shape (dimension, )
    strides : NDArray[int64]
        fortran-style stride factors, shape (dimension, )
    cell : float
        cell size
    center : NDArray[float64]
        ray origin, shape (dimension, )
    directions : NDArray[float64]
        ray directions
    factors : NDArray[float64]
        Jacobian factors
    keys : NDArray[int64]
        sorted, unique occupied cell ids

    Returns
    -------
    Tuple[float64, float64]

    """
    k = dimension // 2
    n_psi = n**(k - 1)
    n_phi = m**k
    d_psi = 0.5*numpy.pi/n
    d_phi = 2.0*numpy.pi/m
    corner = origin + counts*cell
    square = 0.0
    for i in range(dimension):
        dc = corner[i] - center[i]
        do = center[i] - origin[i]
        square += dc*dc if dc*dc > do*do else do*do
    limit = numpy.sqrt(square)
    table = numpy.zeros(n_psi, dtype=float64)
    for i_psi in prange(n_psi):
        shift = i_psi * n_phi
        local = 0.0
        for i_phi in range(n_phi):
            _, r, _ = intersection(origin, counts, strides, cell, center, directions[shift + i_phi], keys, limit)
            local += r**dimension
        table[i_psi] = factors[i_psi]*local
    total = 0.0
    for i_psi in range(n_psi):
        total += table[i_psi]
    integral = total*(d_psi**(k - 1))*(d_phi**k)
    scale = numpy.pi**k
    factor = 1.0
    for i in range(1, k):
        factor *= i
    area = 2.0*scale/factor
    unit = scale/(factor*k)
    radius = (integral/area)**(1.0/dimension)
    return radius, unit*(integral/area)


@njit(parallel=True)
def boundary(
    dimension:int,
    n:int,
    m:int,
    origin:NDArray[float64],
    counts:NDArray[int64],
    stride:NDArray[int64],
    cell:float,
    center:NDArray[float64],
    directions:NDArray[float64],
    keys:NDArray[int64]
) -> Tuple[NDArray[int64], NDArray[float64], NDArray[float64]]:
    """
    Compute first-hit occupied cell ids for all rays (boundary)

    Parameters
    ----------
    dimension : int
        phase space dimension
    n : int
        number of mixing-angle bins
    m : int
        number of in-plane angle bins
    origin : NDArray[float64]
        grid origin (lower corner), shape (dimension, )
    counts : NDArray[int64]
        number of cells per dimension, shape (dimension, )
    stride : NDArray[int64]
        fortran-style stride factors, shape (dimension, )
    cell : float
        cell size
    center : NDArray[float64]
        ray origin, shape (dimension, )
    directions : NDArray[float64]
        ray directions
    keys : NDArray[int64]
        sorted, unique occupied cell ids

    Returns
    -------
    Tuple[NDArray[int64], NDArray[float64], NDArray[float64]]

    """
    corner = origin + counts*cell
    square = 0.0
    for i in range(dimension):
        dc = corner[i] - center[i]
        do = center[i] - origin[i]
        square += dc*dc if dc*dc > do*do else do*do
    limit = numpy.sqrt(square)
    count = len(directions)
    boundary = numpy.empty(count, dtype=int64)
    radii = numpy.empty(count, dtype=float64)
    points = numpy.empty((count, dimension), dtype=float64)
    for i in prange(count):
        index, radius, point = intersection(origin, counts, stride, cell, center, directions[i], keys, limit)
        boundary[i] = index
        radii[i] = radius
        points[i] = point
    return boundary, radii, points
