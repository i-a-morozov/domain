"""
Version and domain loop macro

"""
from typing import Callable
from typing import Optional
from typing import List

from dataclasses import dataclass
from dataclasses import field

import numpy
from numpy import float64
from numpy.typing import NDArray

from domain.da import da
from domain.domain import Domain
from domain.sample import filter
from domain.sample import mask
from domain.sample import sample
from domain.sample import select
from domain.scan import orbit
from domain.scan import scan
from domain.volume import directions
from domain.volume import rays

__version__ = '0.1.1'


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
    dl: float
        domain cell side length
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

    """
    lb: NDArray[float64]
    ub: NDArray[float64]
    dl: float
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

    def __post_init__(self) -> None:
        self.lb = numpy.asarray(self.lb, dtype=float64)
        self.ub = numpy.asarray(self.ub, dtype=float64)
        self.center = numpy.asarray(self.center, dtype=float64)

    @property
    def dimension(self) -> int:
        return len(self.center)

    @property
    def dr(self) -> float:
        return float(self.dl)*self.dimension**0.5

    @property
    def cells(self) -> list[float]:
        return [float(ld)*float(self.dl) for ld in self.lds]


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
    cells: list[Domain]
        resulting boundary domains
    container: Optional[Domain]
        optional full domain container

    """
    data: list
    costs: Optional[list]
    cells: list[Domain]
    container: Optional[Domain]


def compute(
    configuration:Configuration,
    parameters:NDArray[float64],
    pairs:List[tuple[int, int]],
    mapping:Callable[[NDArray[float64], NDArray[float64]], NDArray[float64]],
    objective:Callable[[NDArray[float64], NDArray[float64]], bool],
    cost:Optional[Callable[[NDArray[float64], NDArray[float64]], int]]=None,
    full:bool=False,
    complexity:bool=True,
    verbose:bool=True
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

    Returns
    -------
    Result

    """
    table = []
    costs = [] if (complexity and cost is not None) else None
    cells = []
    container = Domain(configuration.lb, configuration.ub, configuration.dl) if full else None
    generator = orbit(configuration.size, configuration.threshold, mapping)
    for epoch in range(configuration.nepochs):
        if verbose:
            print(epoch)
            print()
        seed = None if configuration.seed is None else configuration.seed + epoch
        ds = directions(configuration.dimension, configuration.ndirections, random=True, seed=seed)
        rb, xb = da(configuration.dimension, configuration.dr, configuration.threshold, configuration.center, ds, objective, parameters, unstable=True)
        buffer = numpy.zeros((configuration.ndirections, configuration.size, configuration.dimension), dtype=float64)
        scan(xb, buffer, generator, parameters)
        initial_cost = None
        if costs is not None:
            out = numpy.zeros(configuration.ndirections, dtype=numpy.int64)
            scan(xb, out, cost, parameters)
            cn = int(configuration.size*numpy.sum((rb/configuration.dr) - 1))
            cm = int(2*numpy.sum(out))
            initial_cost = [cn, cm]
        points = filter(numpy.vstack(buffer), configuration.cut)
        if verbose:
            print(ds.shape)
            print(numpy.vstack(buffer).shape)
            print(points.shape)
            print()
        buffer = numpy.empty((configuration.nsamples*configuration.npoints, configuration.size, configuration.dimension), dtype=float64)
        domains = []
        for cell in configuration.cells:
            domain = Domain(configuration.lb, configuration.ub, cell)
            domain.update(points)
            domains.append(domain)
            if verbose:
                print((domain.size, domain.total))
        if verbose and domains:
            print()
        if container is not None:
            container.update(points)
        local_data = []
        local_cost = [] if costs is not None else None
        while domains:
            domain = domains[0]
            if domain.size == 0:
                cells.append(domains.pop(0))
                table.append(list(local_data))
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
                    scan(initial, buffer, generator, parameters)
                    points = buffer[mask(buffer, configuration.threshold)].reshape(-1, configuration.dimension)
                    points = filter(points, configuration.cut)
                    for item in domains:
                        item.update(points)
                    if container is not None:
                        container.update(points)
                    domain = domains[0]
                    keys, rs, xs = domain.boundary(*pair, configuration.center, ds)
                    rs = rs[keys != -1]
                    xs = xs[keys != -1]
                    flag = int(numpy.sum(keys == -1))
                    radius = 0.0 if len(rs) == 0 else float(numpy.mean(rs**configuration.dimension)**(1/configuration.dimension))
                    boundary = Domain(configuration.lb, configuration.ub, cell)
                    keys = numpy.unique(keys[keys != -1])
                    boundary.insert(keys)
                    domains = [boundary] + domains[1:]
                    domain = domains[0]
                    local_data.append(numpy.asarray([flag, domain.size, len(ds)]))
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
            if costs is not None:
                costs.append([*initial_cost, list(local_cost)])
    return Result(table, costs, cells, container)


__all__ = [
    '__version__',
    'Domain',
    'Configuration',
    'Result',
    'compute',
]
