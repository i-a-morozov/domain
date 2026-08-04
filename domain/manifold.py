"""
Hyperbolic manifold
-------------------

Hyperbolic-manifold bases, initial conditions, and cloud construction

"""
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

from numpy import bool_
from numpy import complex128
from numpy import float64
from numpy import int64
from numpy.typing import NDArray
import numpy

from domain.fp import Mapping
from domain.fp import chain
from domain.fp import classify
from domain.fp import combine
from domain.fp import identify
from domain.fp import monodromy
from domain.fp import unit


def basis(
    values:NDArray[complex128],
    vectors:NDArray[complex128],
    direction:Literal["S", "U"],
    *,
    tolerance:float=1.0E-9,
) -> NDArray[float64]:
    """
    Build a stable or unstable hyperbolic-manifold basis

    Parameters
    ----------
    values: NDArray[complex128]
        grouped reciprocal eigenvalues
    vectors: NDArray[complex128]
        grouped eigenvectors
    direction: Literal["S", "U"]
        stable or unstable direction
    tolerance: float, default=1.0E-9
        unit-circle tolerance

    Returns
    -------
    NDArray[float64]
        orthonormal basis stored in columns

    """
    dimension = values.size
    elliptic = classify(values, tolerance=tolerance)
    stable = unit(values, tolerance=tolerance)
    select = stable[:, 0].astype(int64)^int(direction == "S")
    columns = numpy.take_along_axis(
        vectors,
        select[:, None, None],
        axis=1,
    ).squeeze(axis=1)
    columns = columns[numpy.logical_not(elliptic)]
    if len(columns) == 0:
        return numpy.zeros((dimension, 0), dtype=float64)
    output, _ = numpy.linalg.qr(numpy.real(columns).T, mode="reduced")
    return numpy.asarray(output, dtype=float64)


def sample_line(
    point:NDArray[float64],
    vectors:NDArray[float64],
    scale:float,
    count:int,
    seed:Optional[int]=None,
) -> NDArray[float64]:
    """
    Sample a one-dimensional manifold line

    Parameters
    ----------
    point: NDArray[float64]
        fixed point
    vectors: NDArray[float64]
        one-dimensional orthonormal subspace basis
    scale: float
        line half-length
    count: int
        number of points
    seed: Optional[int], default=None
        random seed

    Returns
    -------
    NDArray[float64]
        points sampled along the line

    """
    generator = numpy.random.default_rng(seed)
    values = generator.uniform(-1.0, 1.0, size=count)
    return point[None, :] + scale*values[:, None]*vectors[:, 0][None, :]


def sample_ball(
    point:NDArray[float64],
    vectors:NDArray[float64],
    scale:float,
    count:int,
    seed:Optional[int]=None,
    surface:bool=False,
) -> NDArray[float64]:
    """
    Sample a solid K-ball or its bounding sphere

    Parameters
    ----------
    point: NDArray[float64]
        fixed point
    vectors: NDArray[float64]
        orthonormal subspace basis
    scale: float
        ball radius
    count: int
        number of points
    seed: Optional[int], default=None
        random seed
    surface: bool, default=False
        flag to sample the bounding sphere

    Returns
    -------
    NDArray[float64]

    """
    generator = numpy.random.default_rng(seed)
    _, size = vectors.shape
    directions = generator.normal(size=(count, size))
    norms = numpy.linalg.norm(directions, axis=1, keepdims=True)
    while numpy.any(norms == 0.0):
        select = norms[:, 0] == 0.0
        directions[select] = generator.normal(size=(numpy.sum(select), size))
        norms = numpy.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions/norms
    radii = (numpy.ones((count, 1), dtype=float64) if surface else generator.random((count, 1))**(1.0/size))
    return point[None, :] + scale*((directions*radii) @ vectors.T)


def sample(
    point:NDArray[float64],
    vectors:NDArray[float64],
    scale:float=1.0E-3,
    nline:int=2**8,
    nball:int=2**8,
    seed:Optional[int]=None,
    surface:bool=False,
) -> NDArray[float64]:
    """
    Sample initials in a hyperbolic-manifold subspace

    Parameters
    ----------
    point: NDArray[float64]
        fixed point
    vectors: NDArray[float64]
        orthonormal subspace basis stored in columns
    scale: float, default=1.0E-3
        line half-length or ball radius
    nline: int, default=2**8
        number of points for a one-dimensional subspace
    nball: int, default=2**8
        number of points for a higher-dimensional subspace
    seed: Optional[int], default=None
        random seed
    surface: bool, default=False
        flag to sample the bounding sphere of a higher-dimensional subspace

    Returns
    -------
    NDArray[float64]
        sampled initial conditions

    """
    dimension, size = vectors.shape
    if size == 0:
        return numpy.zeros((0, dimension), dtype=float64)
    if size == 1:
        return sample_line(point, vectors, scale, nline, seed=seed)
    return sample_ball(
        point,
        vectors,
        scale,
        nball,
        seed=seed,
        surface=surface,
    )


def initials(
    point:NDArray[float64],
    matrix:NDArray[float64],
    scale:float=1.0E-3,
    nline:int=2**8,
    nball:int=2**8,
    tolerance:float=1.0E-9,
    seed:Optional[int]=None,
    surface:bool=False,
) -> Tuple[str, NDArray[float64], NDArray[float64]]:
    """
    Classify a fixed point and generate stable and unstable initials

    Parameters
    ----------
    point: NDArray[float64]
        fixed point
    matrix: NDArray[float64]
        monodromy matrix evaluated at the fixed point
    scale: float, default=1.0E-3
        line half-length or ball radius
    nline: int, default=2**8
        number of points for a one-dimensional manifold
    nball: int, default=2**8
        number of points for a higher-dimensional manifold
    tolerance: float, default=1.0E-9
        eigenvalue comparison tolerance
    seed: Optional[int], default=None
        random seed
    surface: bool, default=False
        flag to sample bounding spheres instead of solid balls

    Returns
    -------
    Tuple[str, NDArray[float64], NDArray[float64]]
        stability label, stable initials, and unstable initials

    """
    values, vectors = numpy.linalg.eig(matrix)
    values, vectors = combine(values, vectors, tolerance=tolerance)
    stable = basis(values, vectors, "S", tolerance=tolerance)
    unstable = basis(values, vectors, "U", tolerance=tolerance)
    generator = numpy.random.default_rng(seed)
    ss, su = generator.integers(0, numpy.iinfo(int64).max, size=1+1)
    return (
        identify(values, tolerance=tolerance),
        sample(point, stable, scale, nline, nball, seed=int(ss), surface=surface),
        sample(point, unstable, scale, nline, nball, seed=int(su), surface=surface)
    )


def downsample(
    cloud:NDArray[float64],
    size:float=1.0E-3,
    total:Optional[int]=None,
    shuffle:bool=False,
    seed:Optional[int]=None,
) -> NDArray[float64]:
    """
    Downsample a cloud by retaining one point per Cartesian cell

    Parameters
    ----------
    cloud: NDArray[float64]
        point cloud shaped ``(count, dimension)``
    size: float, default=1.0E-3
        Cartesian cell size
    total: Optional[int], default=None
        maximum number of retained points
    shuffle: bool, default=False
        flag to randomize point selection within occupied cells
    seed: Optional[int], default=None
        random seed

    Returns
    -------
    NDArray[float64]
        downsampled point cloud

    """
    if not len(cloud):
        return cloud
    keys = numpy.floor(cloud/size).astype(int64)
    keys = keys - keys.min(axis=0, keepdims=True)
    ranges = 1 + keys.max(axis=0)
    strides = numpy.ones(keys.shape[1], dtype=int64)
    for i in range(1, len(strides)):
        strides[i] = strides[i - 1]*ranges[i - 1]
    identifiers = (keys*strides[None, :]).sum(axis=1)
    indices = numpy.arange(len(cloud), dtype=int64)
    generator = numpy.random.default_rng(seed)
    if shuffle:
        order = generator.permutation(len(cloud))
        identifiers, indices = identifiers[order], indices[order]
    order = numpy.argsort(identifiers, kind="stable")
    identifiers, indices = identifiers[order], indices[order]
    select = numpy.concatenate([numpy.asarray([True]), identifiers[1:] != identifiers[:-1]])
    keep = indices[select]
    if total is not None and len(keep) > total:
        keep = generator.permutation(keep)[:total]
    return cloud[keep]


def perturbation(
    count:int,
    radius:float,
    cloud:NDArray[float64],
    seed:Optional[int]=None,
) -> NDArray[float64]:
    """
    Perturb every cloud point inside a full-dimensional ball

    Parameters
    ----------
    count: int
        number of perturbations per cloud point
    radius: float
        perturbation-ball radius
    cloud: NDArray[float64]
        point cloud shaped ``(length, dimension)``
    seed: Optional[int], default=None
        random seed

    Returns
    -------
    NDArray[float64]
        perturbed point cloud shaped ``(length*count, dimension)``

    """
    length, dimension = cloud.shape
    if not length:
        return numpy.empty((0, dimension), dtype=float64)
    vectors = numpy.eye(dimension, dtype=float64)
    generator = numpy.random.default_rng(seed)
    seeds = generator.integers(0, numpy.iinfo(int64).max, size=length)
    values = [sample_ball(point, vectors, radius, count, seed=int(local)) for point, local in zip(cloud, seeds)]
    return numpy.asarray(values).reshape(-1, dimension)


def mask(
    data:NDArray[float64],
    cut:int,
    radius:float,
    strict:bool=True,
) -> NDArray[bool_]:
    """
    Mask orbits escaping after a given iteration

    Parameters
    ----------
    data: NDArray[float64]
        orbits shaped ``(count, length, dimension)``
    cut: int
        threshold iteration
    radius: float
        escape radius
    strict: bool, default=True
        flag to exclude orbits escaping before the threshold

    Returns
    -------
    NDArray[bool_]

    """
    _, length, _ = data.shape
    cut = int(numpy.clip(cut, 0, length))
    nan = numpy.isnan(data)
    square = numpy.sum(data*data, axis=-1)
    selected = (numpy.any(nan[:, cut:, :], axis=(1, 2)) | numpy.any(square[:, cut:] > radius*radius, axis=1))
    if strict:
        before = (numpy.any(nan[:, :cut, :], axis=(1, 2)) | numpy.any(square[:, :cut] > radius*radius, axis=1))
        selected = selected & numpy.logical_not(before)
    return selected


def propagate(
    count:int,
    mapping:Mapping,
    points:NDArray[float64],
    parameters:NDArray[float64],
) -> NDArray[float64]:
    """
    Propagate a set of initial conditions

    The input mapping must accept vector states shaped ``(dimension, count)``.

    Parameters
    ----------
    count: int
        number of mapping iterations to generate
    mapping: Mapping
        vectorized phase-space mapping
    points: NDArray[float64]
        initial conditions shaped ``(number, dimension)``
    parameters: NDArray[float64]
        mapping parameters

    Returns
    -------
    NDArray[float64]
        orbits shaped ``(number, count, dimension)``

    """
    if not len(points):
        return numpy.empty((0, count, points.shape[1]), dtype=float64)
    generate = chain(count, mapping)
    local = numpy.ascontiguousarray(points.T)
    return generate(local, parameters).transpose(2, 0, 1)


def construct(
    orders:List[int],
    points:NDArray[float64],
    forward:Mapping,
    inverse:Mapping,
    parameters:NDArray[float64],
    seed:Optional[int]=None,
    generate:bool=False,
    stable:bool=True,
    unstable:bool=True,
    scale:float=1.0E-3,
    nline:int=8,
    nball:int=16,
    cut:int=4096,
    count:int=8192,
    radius:float=1.0,
    strict:bool=False,
    full:bool=False,
    reduce:bool=False,
    size:float=1.0E-3,
    total:int=10**9,
    shuffle:bool=False,
    difference:float=1.0E-6,
    tolerance:float=1.0E-9,
    surface:bool=False,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    Construct stable and unstable hyperbolic-manifold clouds

    Unstable initials are propagated with the forward mapping and stable
    initials with the inverse mapping. Complete periodic chains are generated
    before sampling unless ``generate`` is false

    The forward and inverse mappings must accept vector states shaped
    ``(dimension, count)``. Reduction is applied independently to each output
    cloud, and ``total`` is the maximum retained by each cloud

    Parameters
    ----------
    orders: List[int]
        period associated with each input point
    points: NDArray[float64]
        periodic-chain representatives shaped ``(number, dimension)``
    forward: Mapping
        vectorized forward phase-space mapping
    inverse: Mapping
        vectorized inverse phase-space mapping
    parameters: NDArray[float64]
        mapping parameters
    seed: Optional[int], default=None
        random seed
    generate: bool, default=False
        flag to generate complete periodic chains from the input representatives
    stable: bool, default=True
        flag to construct the stable-manifold cloud
    unstable: bool, default=True
        flag to construct the unstable-manifold cloud
    scale: float, default=1.0E-3
        initial line half-length or ball radius
    nline: int, default=8
        number of initials for each one-dimensional manifold
    nball: int, default=16
        number of initials for each higher-dimensional manifold
    cut: int, default=4096
        iteration after which an orbit must escape
    count: int, default=8192
        number of mapping iterations to generate per initial condition
    radius: float, default=1.0
        escape radius and output-cloud radial bound
    strict: bool, default=False
        flag to exclude orbits escaping before ``cut``
    full: bool, default=False
        flag to also retain orbits that do not escape before ``count``
    reduce: bool, default=False
        flag to downsample each output cloud
    size: float, default=1.0E-3
        Cartesian cell size used for downsampling
    total: int, default=10**9
        maximum number of points retained in each output cloud
    shuffle: bool, default=False
        flag to randomize point selection during downsampling
    difference: float, default=1.0E-6
        finite-difference step used to compute monodromy matrices
    tolerance: float, default=1.0E-9
        eigenvalue comparison tolerance
    surface: bool, default=False
        flag to sample bounding spheres instead of solid balls

    Returns
    -------
    Tuple[NDArray[float64], NDArray[float64]]
        stable and unstable manifold clouds

    """
    points = numpy.asarray(points, dtype=float64)
    if not len(points):
        dimension = points.shape[1] if points.ndim == 2 else 0
        empty = numpy.empty((0, dimension), dtype=float64)
        return empty, numpy.copy(empty)
    dimension = points.shape[1]
    if not stable and not unstable:
        empty = numpy.empty((0, dimension), dtype=float64)
        return empty, numpy.copy(empty)
    grouped = {}
    for order, point in zip(orders, points):
        grouped.setdefault(order, []).append(point)
    if generate:
        for order in grouped:
            generate_chain = chain(order, forward)
            local = numpy.ascontiguousarray(numpy.asarray(grouped[order]).T)
            chains = generate_chain(local, parameters).transpose(2, 0, 1)
            grouped[order] = chains.reshape(-1, dimension)
    generator = numpy.random.default_rng(seed)
    unstable_samples = []
    stable_samples = []
    for order, local in grouped.items():
        matrix = monodromy(order, forward, difference=difference)
        for point in local:
            values, vectors = numpy.linalg.eig(matrix(point, parameters))
            values, vectors = combine(values, vectors, tolerance=tolerance)
            unstable_seed, stable_seed = generator.integers(0, numpy.iinfo(int64).max, size=2)
            if unstable:
                unstable_basis = basis(values, vectors, "U", tolerance=tolerance)
                unstable_samples.append(sample(point, unstable_basis, scale, nline, nball, seed=int(unstable_seed), surface=surface))
            if stable:
                stable_basis = basis(values, vectors, "S", tolerance=tolerance)
                stable_samples.append(sample(point, stable_basis, scale, nline, nball, seed=int(stable_seed), surface=surface))
    unstable_samples = [value for value in unstable_samples if len(value)]
    stable_samples = [value for value in stable_samples if len(value)]
    unstable_points = (numpy.concatenate(unstable_samples) if unstable_samples else numpy.empty((0, dimension), dtype=float64))
    stable_points = (numpy.concatenate(stable_samples) if stable_samples else numpy.empty((0, dimension), dtype=float64))
    unstable_orbits = propagate(count, forward, unstable_points, parameters)
    stable_orbits = propagate(count, inverse, stable_points, parameters)
    def finalize(orbits:NDArray[float64]) -> NDArray[float64]:
        if not len(orbits):
            return numpy.empty((0, dimension), dtype=float64)
        with numpy.errstate(over="ignore", invalid="ignore"):
            selected = mask(orbits, cut, radius, strict=strict)
            if full:
                finite = numpy.isfinite(orbits).all(axis=(1, 2))
                square = numpy.sum(orbits*orbits, axis=-1)
                bounded = numpy.all(square <= radius*radius, axis=1)
                selected = selected | (finite & bounded)
            orbits = orbits[selected]
            cloud = orbits.reshape(-1, dimension)
            cloud = cloud[numpy.isfinite(cloud).all(axis=1)]
            cloud = cloud[numpy.linalg.norm(cloud, axis=1) < radius]
        if reduce:
            cloud = downsample(cloud, size=size, total=total, shuffle=shuffle, seed=seed)
        return cloud
    return finalize(stable_orbits), finalize(unstable_orbits)
