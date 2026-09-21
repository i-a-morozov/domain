"""
Construction
------------

Construction with projected storage and/or random reference directions

"""
import numpy as np

from domain.da import da
from domain.sample import sample, select
from domain.scan import orbit, scan
from domain.volume import directions, rays, mean
from domain import Result, project

def reference(geometry, stages, random, seed):
    dimension = len(geometry.axes)
    result = []
    for stage, specification in enumerate(stages):
        if random:
            if np.isscalar(specification):
                count = int(specification)
            else:
                n, m = specification
                k = dimension//2
                count = int(n**(k - 1)*m**k)
            local_seed = None if seed is None else np.random.SeedSequence([seed, stage, 1])
            ds = directions(dimension, count, seed=local_seed)
        else:
            ds, _ = rays(dimension, *specification)
        result.append(geometry.embed_directions(ds))
    return result


def boundary(domain, origins, directions):
    keys, radii, missed = [], [], []
    for origin in origins:
        local_keys, local_radii, points = domain.boundary(1, 1, origin, directions)
        keys.append(local_keys)
        radii.append(local_radii)
        missed.append(np.count_nonzero(local_keys < 0))
    return np.concatenate(keys), np.concatenate(radii), np.asarray(missed)


def retain(boundary, cloud, geometry, origins, missed, count):
    if not geometry.storage_periodic:
        return
    centers = cloud.construct
    keep = np.zeros(len(centers), dtype=bool)
    for origin in origins[missed == count]:
        local = np.ones(len(centers), dtype=bool)
        for axis, period in geometry.storage_periodic:
            distance = (centers[:, axis] - origin[axis] + period/2) % period - period/2
            local &= np.abs(distance) <= 0.5*cloud.cell[axis] + geometry.configuration.projection_epsilon
        keep |= local
    boundary.insert(cloud.keys[keep])


def compute_geometry(
    configuration, 
    parameters, 
    stages, 
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
    boundary_origins,
    metric=None,
    indicator_threshold=None
):
    random = bool(random or geometry.projection)
    references = reference(geometry, stages, random, configuration.seed)
    origins = geometry.origins(boundary_origins)
    dimension = len(geometry.axes)
    options = dict(escape=escape, projection=geometry.projection, periodic=geometry.periodic)
    generator = orbit(configuration.size, configuration.threshold, mapping, escape=escape)
    container = geometry.domain(configuration.dl) if full else None
    costs = [] if complexity and cost is not None else None
    result = Result(
        [], 
        costs, 
        [], 
        [], 
        container,
        projection=geometry.projection,
        periodic=geometry.periodic,
        coordinates=tuple(int(i) for i in geometry.keep),
        references=references, 
        origins=origins,
        dimension=dimension
    )

    def deposit(initials, targets):
        if metric is not None:
            values = np.empty(len(initials), dtype=np.float64)
            scan(initials, values, metric, parameters)
            initials = initials[~np.isfinite(values) | (values > indicator_threshold)]
        return project(initials, generator, parameters, configuration, targets, escaping=metric is None, **options)

    for epoch in range(configuration.nepochs):
        domains = [geometry.domain(cell) for cell in configuration.cells]
        targets = domains if container is None else [*domains, container]
        seed = None if configuration.seed is None else configuration.seed + epoch
        rng = np.random.default_rng(seed)
        initial_cost = [0, 0]
        if initial is None:
            seed_ds = geometry.embed_directions(directions(dimension, configuration.ndirections, seed=seed))
            full_ds = geometry.lift_directions(seed_ds)
            seed_radii, seed_points = [], []
            for origin in origins:
                rb, xb = da(configuration.dimension, geometry.dr, configuration.threshold, geometry.lift(origin[None])[0], full_ds, objective, parameters, unstable=True)
                seed_radii.append(rb)
                seed_points.append(xb)
            rb, xb = np.concatenate(seed_radii), np.vstack(seed_points)
            deposit(xb, targets)
            if costs is not None:
                counts = np.zeros(len(xb), dtype=np.int64)
                scan(xb, counts, cost, parameters)
                initial_cost = [int(configuration.size*np.sum(np.maximum(rb/geometry.dr - 1, 0))), int(2*np.sum(counts))]
        else:
            seeds = geometry.project(geometry.lift(initial))
            for target in targets:
                target.update(seeds)
        data, rads, local_cost = [], [], []
        if verbose:
            print(f'epoch {epoch + 1:02d}: {[domain.size for domain in domains]} initial cells', flush=True)
        for level, requested_cell in enumerate(configuration.cells):
            domain = domains[level]
            for ds in references:
                for round_index in range(configuration.nrounds):
                    if not domain.size:
                        break
                    if geometry.active or domain.dimension < 4 or domain.dimension % 2:
                        centers = domain.transform(rng.choice(domain.keys, configuration.nsamples))
                    else:
                        _, centers, _, _ = select(
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
                            power=configuration.power
                        )
                    initials = geometry.lift(sample(configuration.npoints, configuration.scale*domain.cell, centers))
                    targets = domains[level:] if container is None else [*domains[level:], container]
                    deposit(initials, targets)
                    keys, radii, missed = boundary(domain, origins, ds)
                    hit = keys >= 0
                    radius = float(mean(dimension, radii[hit])) if np.any(hit) else 0.0
                    shell = geometry.domain(requested_cell)
                    shell.insert(np.unique(keys[hit]))
                    retain(shell, domain, geometry, origins, missed, len(ds))
                    if shell.size:
                        domain = shell
                    domains[level] = domain
                    data.append(np.asarray([int(missed.sum()), domain.size, len(origins)*len(ds)]))
                    rads.append(radius)
                    if costs is not None:
                        counts = np.zeros(len(initials), dtype=np.int64)
                        scan(initials, counts, cost, parameters)
                        local_cost.append(counts)
                    if verbose:
                        print(f'{round_index + 1:03d} {domain.size:12d} {missed.sum():12d} worst level missed: {100*missed.max()/len(ds):.2f}% {radius:.6f}', flush=True)
                    if np.all(missed <= (1 - configuration.termination)*len(ds)):
                        break
            result.cells.append(domain)
            result.data.append(list(data))
            result.rads.append(list(rads))
            if costs is not None:
                costs.append([*initial_cost, list(local_cost)])
    return result
