"""
Version and domain loop macro

"""
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import List
from typing import Sequence

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import numpy
from numpy import float64
from numpy.typing import NDArray

from numba import njit

from domain.escape import Escape
from domain.da import da
from domain.domain import Domain
from domain.domain import array
from domain.sample import filter
from domain.sample import mask
from domain.sample import sample
from domain.sample import select
from domain.scan import orbit
from domain.scan import scan
from domain.volume import directions
from domain.volume import rays
from domain.volume import mean
from domain.projection import Geometry

__version__ = '0.1.5'


@dataclass
class Configuration:
    """
    Domain configuration

    Parameters
    ----------
    lb: NDArray[float64]
        domain lower bounds
    ub: NDArray[float64]
        domain upper bounds
    dl: float | NDArray[float64]
        scalar domain cell size or one cell size per dimension
    size: int
        total number of mapping iterations
    ndirections: int, default=8
        number of initial random directions
    nsamples: int, default=1024
        number of marked cells to select for expansion
    npoints: int, default=8
        number of random points to generate for each selected cell
    nrounds: int, default=128
        maximum number of expansion rounds
    nepochs: int, default=32
        number of outer Monte Carlo realizations
    center: NDArray[float64], default=(0, 0, 0, 0)
        domain center point (origin)
    cut: float, default=2.0
        selection cut radius
    threshold: float, default=5.0
        threshold radius used in orbit computation
    termination: float, default=0.95
        ray saturation termination parameter
    scale: float, default=3.0
        perturbation scaling factor
    lds: List[float], default=(1.0, )
        cell-size multipliers relative to `dl`
    bins_plane: int, default=16
        number of bins for the mixing coordinate
    bins_phase: int, default=32
        number of bins for in-plane phases
    phase_threshold: float, default=1.0E-3
        phase threshold used in weighted selection
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
    seed: Optional[int], default=None
        random seed offset for initial random directions
    batch: int, default=512
        maximum number of full orbits held in memory at once
    projection: tuple, default=()
        omitted original-coordinate indices and half-widths in base dl units
    periodic: tuple, default=()
        original-coordinate indices and periods
        retained axes store one period starting at lb[index]
        omitted axes use shortest periodic distance
    random: bool, default=False
        random reference directions (projection always enables random directions)
    epsilon: float, default=1.0E-6
        numerical tolerance added to section half-widths

    """
    lb: NDArray[float64]
    ub: NDArray[float64]
    dl: float | NDArray[float64]
    size: int
    ndirections: int = 8
    nsamples: int = 1024
    npoints: int = 8
    nrounds: int = 128
    nepochs: int = 32
    center: NDArray[float64] = field(default_factory=lambda: numpy.zeros(4, dtype=float64))
    cut: float = 2.0
    threshold: float = 5.0
    termination: float = 0.95
    scale: float = 3.0
    lds: List[float] = field(default_factory=lambda: (1.0, ))
    bins_plane: int = 16
    bins_phase: int = 32
    phase_threshold: float = 1.0E-3
    alpha_plane: float = 1.0
    alpha_phase: float = 0.5
    boost: float = 0.5
    delta: float = 1.0
    uniform: float = 0.05
    power: float = 1.0
    seed: Optional[int] = None
    batch: int = 512
    projection: tuple = ()
    periodic: tuple = ()
    random: bool = False
    epsilon: float = 1.0E-6

    def __post_init__(self) -> None:
        self.lb = numpy.asarray(self.lb, dtype=float64)
        self.ub = numpy.asarray(self.ub, dtype=float64)
        self.center = numpy.asarray(self.center, dtype=float64)
        self.dl = array(self.dl, self.dimension)
        self.batch = int(self.batch)

    @property
    def dimension(self) -> int:
        return len(self.center)

    @property
    def dr(self) -> float:
        return float(numpy.linalg.norm(self.dl))

    @property
    def cells(self) -> list[NDArray[float64]]:
        return [float(ld)*self.dl for ld in self.lds]


@dataclass
class Result:
    """
    Domain construction result

    Parameters
    ----------
    data: list
        convergence data containers
    costs: Optional[list]
        cost data containers
    rads: list
        current radius estimates for each round
    cells: list[Domain]
        resulting boundary domains
    container: Optional[Domain]
        optional full domain container
    projection, periodic, coordinates:
        section, original-index periodic metadata and stored-column map
    references, origins:
        reduced-space reference rays and origins for the new construction modes
        reference directions are reused across rounds, levels and epochs
    dimension:
        dimension of the nonperiodic ray family

    """
    data: list
    costs: Optional[list]
    rads: list
    cells: list[Domain]
    container: Optional[Domain]
    projection: tuple = ()
    periodic: tuple = ()
    coordinates: tuple = ()
    references: list = field(default_factory=list)
    origins: Optional[NDArray[float64]] = None
    dimension: Optional[int] = None


def batches(
    initial:NDArray[float64],
    generator:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    parameters:NDArray[float64],
    configuration:Configuration,
    escaping:bool=False,
    cut:Optional[float]=None, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None
) -> Iterator[NDArray[float64]]:
    """
    Stream orbit points using the generator's escape criterion

    Initials and generator are always full-map dimensional
    Projection/periodic options override the configuration
    Filter full states before section selection and periodic wrapping
    In projection mode include selected trajectory initials as well as iterates

    """
    geometry = Geometry(configuration, projection, periodic)
    radius = configuration.cut if cut is None else float(cut)
    for start in range(0, len(initial), configuration.batch):
        local = numpy.ascontiguousarray(initial[start:start + configuration.batch])
        buffer = numpy.empty((len(local), configuration.size, configuration.dimension), dtype=float64)
        scan(local, buffer, generator, parameters)
        if escaping:
            lost = mask(buffer, configuration.threshold, escape=escape, parameters=parameters)
            buffer = buffer[lost]
            local = local[lost]
        if len(buffer):
            if geometry.projection:
                points = filter(local, radius, escape=escape, parameters=parameters)
                points = geometry.project(points)
                if len(points):
                    yield points
            points = filter(buffer.reshape(-1, configuration.dimension), radius, escape=escape, parameters=parameters)
            if geometry.active:
                points = geometry.project(points)
            if len(points):
                yield points


def collect(
    initial:NDArray[float64],
    generator:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    parameters:NDArray[float64],
    configuration:Configuration,
    escaping:bool=False,
    cut:Optional[float]=None, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None
) -> NDArray[float64]:
    """
    Collect streamed points
    
    """
    options = {} if escape is None else {'escape': escape}
    geometry = Geometry(configuration, projection, periodic)
    if geometry.active or projection is not None or periodic is not None:
        options.update(projection=geometry.projection, periodic=geometry.periodic)
    chunks = list(batches(initial, generator, parameters, configuration, escaping=escaping, cut=cut, **options))
    if not chunks:
        return numpy.empty((0, geometry.dimension), dtype=float64)
    return numpy.vstack(chunks)


def project(
    initial:NDArray[float64],
    generator:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    parameters:NDArray[float64],
    configuration:Configuration,
    domains:Sequence[Domain],
    escaping:bool=False,
    cut:Optional[float]=None, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None
) -> int:
    """
    Update domains from streamed points
    
    """
    options = {} if escape is None else {'escape': escape}
    geometry = Geometry(configuration, projection, periodic)
    if geometry.active or projection is not None or periodic is not None:
        options.update(projection=geometry.projection, periodic=geometry.periodic)
    count = 0
    for points in batches(initial, generator, parameters, configuration, escaping=escaping, cut=cut, **options):
        for domain in domains:
            domain.update(points)
        count += len(points)
    return count


def grow(
    configuration:Configuration,
    parameters:NDArray[float64],
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    domain:Domain,
    epochs:int=64,
    limit:int=64_000_000,
    verbose:bool=True, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None
) -> Domain:
    """
    Grow an escape-based transport domain

    Parameters
    ----------
    configuration: Configuration
        sampling, orbit, escape, and batching parameters
    parameters: NDArray[float64]
        additional parameters passed to ``mapping``
    mapping: Callable
        forward or inverse state mapping
    domain: Domain
        seeded domain to update in place
    epochs: int, default=64
        maximum number of growth epochs
    limit: int, default=64_000_000
        stop once the number of marked cells exceeds this value
    verbose: bool, default=True
        print the epoch and current domain size

    Returns
    -------
    Domain

    """
    options = {} if escape is None else {'escape': escape}
    generator = orbit(configuration.size, configuration.threshold, mapping, **options)
    geometry = Geometry(configuration, projection, periodic)
    if geometry.active or projection is not None or periodic is not None:
        options.update(projection=geometry.projection, periodic=geometry.periodic)
    for epoch in range(epochs):
        if domain.size == 0:
            break
        previous = domain.size
        indices = numpy.random.choice(domain.list, configuration.nsamples)
        centers = domain.transform(indices)
        initial = sample(configuration.npoints, configuration.scale*domain.cell, centers)
        if geometry.active:
            initial = geometry.lift(initial)
        batches = (len(initial) + configuration.batch - 1)//configuration.batch
        if verbose:
            print(
                f'{epoch + 1:02d} start'
                f' {previous:12d}'
                f' {len(initial):8d} initials'
                f' {batches:4d} batches',
                flush=True)
        project(initial, generator, parameters, configuration, (domain, ), escaping=True, cut=configuration.threshold, **options)
        if verbose:
            print(
                f'{epoch + 1:02d} done '
                f' {domain.size:12d}'
                f' +{domain.size - previous:d}',
                flush=True
            )
        if domain.size > limit:
            break
    return domain


def grow_indicator(
    configuration:Configuration,
    parameters:NDArray[float64],
    factory:Callable,
    forward:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    inverse:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    threshold:float,
    domain:Domain,
    epochs:int=64,
    limit:int=64_000_000,
    verbose:bool=True, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None
) -> Domain:
    """
    Grow an indicator-based transport domain

    Parameters
    ----------
    configuration: Configuration
        sampling, orbit, radius, and batching parameters
    parameters: NDArray[float64]
        additional parameters passed to the mappings and indicators
    factory: Callable
        indicator factory called as ``factory(size, forward, inverse)``
    forward: Callable
        forward state mapping
    inverse: Callable
        inverse state mapping
    threshold: float
        indicator selection threshold
    domain: Domain
        seeded domain to update in place
    epochs: int, default=16
        maximum number of growth epochs
    limit: int, default=128_000_000
        stop once the number of marked cells exceeds this value
    verbose: bool, default=True
        print the epoch and current domain size

    Returns
    -------
    Domain

    """
    metric_forward_inverse = factory(configuration.size, forward, inverse)
    metric_inverse_forward = factory(configuration.size, inverse, forward)
    options = {} if escape is None else {'escape': escape}
    orbit_forward = orbit(configuration.size, configuration.threshold, forward, **options)
    orbit_inverse = orbit(configuration.size, configuration.threshold, inverse, **options)
    geometry = Geometry(configuration, projection, periodic)
    if geometry.active or projection is not None or periodic is not None:
        options.update(projection=geometry.projection, periodic=geometry.periodic)
    for epoch in range(epochs):
        if domain.size == 0:
            break
        previous = domain.size
        indices = numpy.random.choice(domain.list, configuration.nsamples)
        centers = domain.transform(indices)
        initial = sample(configuration.npoints, configuration.scale*domain.cell, centers)
        if geometry.active:
            initial = geometry.lift(initial)
        if verbose:
            print(
                f'{epoch + 1:02d} start'
                f' {previous:12d}'
                f' {len(initial):8d} initials',
                flush=True
            )
        values_forward_inverse = numpy.empty(len(initial), dtype=float64)
        values_inverse_forward = numpy.empty(len(initial), dtype=float64)
        scan(initial, values_forward_inverse, metric_forward_inverse, parameters)
        scan(initial, values_inverse_forward, metric_inverse_forward, parameters)
        selected = initial[
            ~numpy.isfinite(values_forward_inverse)
            | ~numpy.isfinite(values_inverse_forward)
            | (values_forward_inverse > threshold)
            | (values_inverse_forward > threshold)
        ]
        batches = (len(selected) + configuration.batch - 1)//configuration.batch
        if verbose:
            print(
                f'   selected {len(selected):8d}'
                f' {batches:4d} batches',
                flush=True,
            )
        project(selected, orbit_forward, parameters, configuration, (domain, ), cut=configuration.threshold, **options)
        project(selected, orbit_inverse, parameters, configuration, (domain, ), cut=configuration.threshold, **options)
        if verbose:
            print(
                f'{epoch + 1:02d} done '
                f' {domain.size:12d}'
                f' +{domain.size - previous:d}',
                flush=True,
            )
        if domain.size > limit:
            break
    return domain


def compute(
    configuration:Configuration,
    parameters:NDArray[float64],
    pairs:Sequence[int | tuple[int, int]],
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    objective:Callable[[NDArray[float64], NDArray[float64]], bool],
    cost:Optional[Callable[[NDArray[float64], NDArray[float64]], int]]=None,
    full:bool=False,
    complexity:bool=True,
    verbose:bool=True,
    initial:Optional[NDArray[float64]]=None, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None,
    random:Optional[bool]=None,
    boundary_origins:Optional[NDArray[float64]]=None,
) -> Result:
    """
    Run domain construction loop

    Parameters
    ----------
    configuration: Configuration
        domain construction configuration
    parameters: NDArray[float64]
        additional parameters passed to `mapping`, `objective`, and `cost`
    pairs: Sequence[tuple[int, int]]
        ray-count pairs used in the boundary saturation loop
    mapping: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        mapping mapping
    objective: Callable[[NDArray[float64], NDArray[float64]], bool]
        stability objective used in the initial DA search
    cost: Optional[Callable[[NDArray[float64], NDArray[float64]], int]], default=None
        optional cost function
    full: bool, default=True
        flag to construct and update the full domain container
    complexity: bool, default=True
        flag to compute cost data when `cost` is provided
    verbose: bool, default=False
        verbose output flag
    initial: Optional[NDArray[float64]], default=None
        points used to seed the domains directly

    Returns
    -------
    Result

    """
    geometry = Geometry(configuration, projection, periodic)
    if projection is not None or periodic is not None:
        configuration = replace(configuration, projection=geometry.projection, periodic=geometry.periodic)
    random = configuration.random if random is None else random
    if geometry.active or random or boundary_origins is not None:
        from domain.construction import compute_geometry
        return compute_geometry(
            configuration, 
            parameters, 
            pairs, 
            mapping, 
            objective, 
            cost,
            full,
            complexity,
            verbose,
            initial,
            escape,
            geometry,
            random,
            boundary_origins
        )
    table = []
    costs = [] if (complexity and cost is not None) else None
    rads = []
    cells = []
    container = Domain(configuration.lb, configuration.ub, configuration.dl) if full else None
    options = {} if escape is None else {'escape': escape}
    generator = orbit(configuration.size, configuration.threshold, mapping, **options)
    seeds = None if initial is None else initial
    for epoch in range(configuration.nepochs):
        if verbose:
            print(epoch)
            print()
        domains = [Domain(configuration.lb, configuration.ub, cell) for cell in configuration.cells]
        targets = domains if container is None else [*domains, container]
        if seeds is None:
            seed = None if configuration.seed is None else configuration.seed + epoch
            ds = directions(configuration.dimension, configuration.ndirections, random=True, seed=seed)
            rb, xb = da(configuration.dimension, configuration.dr, configuration.threshold, configuration.center, ds, objective, parameters, unstable=True)
            seed_options = {} if escape is None else {'escape': escape, 'escaping': True}
            point_count = project(xb, generator, parameters, configuration, targets, **seed_options)
            initial_cost = None
            if costs is not None:
                out = numpy.zeros(configuration.ndirections, dtype=numpy.int64)
                scan(xb, out, cost, parameters)
                cn = int(configuration.size*numpy.sum((rb/configuration.dr) - 1))
                cm = int(2*numpy.sum(out))
                initial_cost = [cn, cm]
            if verbose:
                print(ds.shape)
                print((len(xb)*configuration.size, configuration.dimension))
                print((point_count, configuration.dimension))
                print()
        else:
            point_count = len(seeds)
            for target in targets:
                target.update(seeds)
            initial_cost = [0, 0] if costs is not None else None
            if verbose:
                print('initial', seeds.shape)
                print()
        for domain in domains:
            if verbose:
                print((domain.size, domain.total))
        if verbose and domains:
            print()
        local_data = []
        local_cost = [] if costs is not None else None
        local_rads = []
        while domains:
            domain, *_ = domains
            if domain.size == 0:
                cells.append(domains.pop(0))
                table.append(list(local_data))
                rads.append(list(local_rads))
                if costs is not None:
                    costs.append([*initial_cost, list(local_cost)])
                continue
            cell = domain.cell
            for pair in pairs:
                ds, _ = rays(domain.dimension, *pair)
                for i in range(configuration.nrounds):
                    indices, centers, probabilities, statistics = select(
                        domain,
                        configuration.nsamples,
                        bins_plane=configuration.bins_plane,
                        bins_phase=configuration.bins_phase,
                        threshold=configuration.phase_threshold,
                        alpha_plane=configuration.alpha_plane,
                        alpha_phase=configuration.alpha_phase,
                        boost=configuration.boost,
                        delta=configuration.delta,
                        uniform=configuration.uniform,
                        power=configuration.power,
                    )
                    initial = sample(configuration.npoints, configuration.scale*cell, centers)
                    targets = domains if container is None else [*domains, container]
                    project(initial, generator, parameters, configuration, targets, escaping=True, **options)
                    domain, *_ = domains
                    keys, rs, xs = domain.boundary(*pair, configuration.center, ds)
                    rs = rs[keys != -1]
                    xs = xs[keys != -1]
                    flag = int(numpy.sum(keys == -1))
                    radius = 0.0 if len(rs) == 0 else float(mean(configuration.dimension, rs))
                    boundary = Domain(configuration.lb, configuration.ub, cell)
                    keys = numpy.unique(keys[keys != -1])
                    boundary.insert(keys)
                    domains = [boundary] + domains[1:]
                    domain, *_ = domains
                    local_data.append(numpy.asarray([flag, domain.size, len(ds)]))
                    local_rads.append(radius)
                    if local_cost is not None:
                        out = numpy.zeros(len(initial), dtype=numpy.int64)
                        scan(initial, out, cost, parameters)
                        local_cost.append(out)
                    if verbose:
                        total = 0 if container is None else container.size
                        print(f'{i + 1:02d}', f'{domain.size:12d}', f'{flag:12d}', f'{100*flag/len(ds):12.2f}', f'{total:12d}', radius)
                    if flag <= (1.0 - configuration.termination)*len(ds):
                        break
            if verbose:
                print()
            cells.append(domains.pop(0))
            table.append(list(local_data))
            rads.append(list(local_rads))
            if costs is not None:
                costs.append([*initial_cost, list(local_cost)])
    return Result(table, costs, rads, cells, container)


def compute_indicator(
    configuration:Configuration,
    parameters:NDArray[float64],
    pairs:Sequence[int | tuple[int, int]],
    factory:Callable[[int, Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]], Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]], Callable[[NDArray[float64], NDArray[float64]], float64]],
    forward:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    inverse:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    threshold:float,
    cost:Optional[Callable[[NDArray[float64], NDArray[float64]], int]]=None,
    full:bool=False,
    complexity:bool=True,
    verbose:bool=True,
    initial:Optional[NDArray[float64]]=None, *,
    escape:Optional[Escape]=None,
    projection=None,
    periodic=None,
    random:Optional[bool]=None,
    boundary_origins:Optional[NDArray[float64]]=None
) -> Result:
    """
    Run domain construction loop using a scalar indicator threshold

    Parameters
    ----------
    configuration: Configuration
        domain construction configuration
    parameters: NDArray[float64]
        additional parameters passed to mappings, indicator, and cost
    pairs: Sequence[tuple[int, int]]
        ray-count pairs used in the boundary saturation loop
    factory: Callable
        indicator factory called as `factory(configuration.size, forward, inverse)`
    forward: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        forward mapping
    inverse: Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]]
        inverse mapping
    threshold: float
        indicator threshold used to classify escaping initials
    cost: Optional[Callable[[NDArray[float64], NDArray[float64]], int]], default=None
        optional cost function
    full: bool, default=True
        flag to construct and update the full domain container
    complexity: bool, default=True
        flag to compute cost data when `cost` is provided
    verbose: bool, default=False
        verbose output flag
    initial: Optional[NDArray[float64]], default=None
        points used to seed the domains directly; when given, the initial

    Returns
    -------
    Result

    """
    metric = factory(configuration.size, forward, inverse)
    options = {} if escape is None else {'escape': escape}
    generator = orbit(configuration.size, configuration.threshold, forward, **options)

    @njit
    def objective(state:NDArray[float64], parameters:NDArray[float64]) -> bool:
        return metric(state, parameters) <= threshold

    geometry = Geometry(configuration, projection, periodic)
    if projection is not None or periodic is not None:
        configuration = replace(configuration, projection=geometry.projection, periodic=geometry.periodic)
    random = configuration.random if random is None else random
    if geometry.active or random or boundary_origins is not None:
        from domain.construction import compute_geometry
        return compute_geometry(
            configuration,
            parameters,
            pairs,
            forward,
            objective,
            cost,
            full,
            complexity,
            verbose,
            initial,
            escape,
            geometry,
            random,
            boundary_origins,
            metric=metric,
            indicator_threshold=threshold
    )
    table = []
    costs = [] if (complexity and cost is not None) else None
    rads = []
    cells = []
    container = Domain(configuration.lb, configuration.ub, configuration.dl) if full else None
    seeds = None if initial is None else initial
    for epoch in range(configuration.nepochs):
        if verbose:
            print(epoch)
            print()
        domains = [Domain(configuration.lb, configuration.ub, cell) for cell in configuration.cells]
        targets = domains if container is None else [*domains, container]
        if seeds is None:
            seed = None if configuration.seed is None else configuration.seed + epoch
            ds = directions(configuration.dimension, configuration.ndirections, random=True, seed=seed)
            rb, xb = da(configuration.dimension, configuration.dr, configuration.threshold, configuration.center, ds, objective, parameters, unstable=True)
            values = numpy.zeros(configuration.ndirections, dtype=float64)
            scan(xb, values, metric, parameters)
            escaped = xb[~numpy.isfinite(values) | (values > threshold)]
            point_count = project(escaped, generator, parameters, configuration, targets, **options)
            initial_cost = None
            if costs is not None:
                out = numpy.zeros(configuration.ndirections, dtype=numpy.int64)
                scan(xb, out, cost, parameters)
                cn = int(configuration.size*numpy.sum((rb/configuration.dr) - 1))
                cm = int(2*numpy.sum(out))
                initial_cost = [cn, cm]
            if verbose:
                print(ds.shape)
                print(escaped.shape)
                print((point_count, configuration.dimension))
                print()
        else:
            point_count = len(seeds)
            for target in targets:
                target.update(seeds)
            initial_cost = [0, 0] if costs is not None else None
            if verbose:
                print('initial', seeds.shape)
                print()
        for domain in domains:
            if verbose:
                print((domain.size, domain.total))
        if verbose and domains:
            print()
        local_data = []
        local_cost = [] if costs is not None else None
        local_rads = []
        while domains:
            domain, *_ = domains
            if domain.size == 0:
                cells.append(domains.pop(0))
                table.append(list(local_data))
                rads.append(list(local_rads))
                if costs is not None:
                    costs.append([*initial_cost, list(local_cost)])
                continue
            cell = domain.cell
            for pair in pairs:
                ds, _ = rays(domain.dimension, *pair)
                for i in range(configuration.nrounds):
                    indices, centers, probabilities, statistics = select(
                        domain,
                        configuration.nsamples,
                        bins_plane=configuration.bins_plane,
                        bins_phase=configuration.bins_phase,
                        threshold=configuration.phase_threshold,
                        alpha_plane=configuration.alpha_plane,
                        alpha_phase=configuration.alpha_phase,
                        boost=configuration.boost,
                        delta=configuration.delta,
                        uniform=configuration.uniform,
                        power=configuration.power,
                    )
                    initial = sample(configuration.npoints, configuration.scale*cell, centers)
                    values = numpy.zeros(len(initial), dtype=float64)
                    scan(initial, values, metric, parameters)
                    escaped = initial[~numpy.isfinite(values) | (values > threshold)]
                    targets = domains if container is None else [*domains, container]
                    project(escaped, generator, parameters, configuration, targets, **options)
                    domain, *_ = domains
                    keys, rs, xs = domain.boundary(*pair, configuration.center, ds)
                    rs = rs[keys != -1]
                    xs = xs[keys != -1]
                    flag = int(numpy.sum(keys == -1))
                    radius = 0.0 if len(rs) == 0 else float(mean(configuration.dimension, rs))
                    boundary = Domain(configuration.lb, configuration.ub, cell)
                    keys = numpy.unique(keys[keys != -1])
                    boundary.insert(keys)
                    domains = [boundary] + domains[1:]
                    domain, *_ = domains
                    local_data.append(numpy.asarray([flag, domain.size, len(ds)]))
                    local_rads.append(radius)
                    if local_cost is not None:
                        out = numpy.zeros(len(initial), dtype=numpy.int64)
                        scan(initial, out, cost, parameters)
                        local_cost.append(out)
                    if verbose:
                        total = 0 if container is None else container.size
                        print(f'{i + 1:02d}', f'{domain.size:12d}', f'{flag:12d}', f'{100*flag/len(ds):12.2f}', f'{total:12d}', radius)
                    if flag <= (1.0 - configuration.termination)*len(ds):
                        break
            if verbose:
                print()
            cells.append(domains.pop(0))
            table.append(list(local_data))
            rads.append(list(local_rads))
            if costs is not None:
                costs.append([*initial_cost, list(local_cost)])
    return Result(table, costs, rads, cells, container)


__all__ = [
    '__version__',
    'Domain',
    'Configuration',
    'Result',
    'collect',
    'project',
    'compute',
    'compute_indicator',
    'grow',
    'grow_indicator'
]
