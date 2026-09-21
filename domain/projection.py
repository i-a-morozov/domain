"""
Projection
----------

Finite-width sections, lifting, and periodic coordinate utilities

"""
from itertools import product

import numpy as np

from domain.domain import Domain


def specification(entries, dimension, name):
    entries = () if entries is None else tuple(entries)
    result = []
    for index, value in entries:
        result.append((int(index), float(value)))
    return tuple(sorted(result))


class Geometry:
    """
    Full-map configuration with a possibly reduced storage

    """
    def __init__(self, configuration, projection=None, periodic=None):
        self.configuration = configuration
        self.projection = specification(configuration.projection if projection is None else projection, configuration.dimension, 'projection')
        self.periodic = specification(configuration.periodic if periodic is None else periodic, configuration.dimension, 'periodic')
        excluded = {i for i, _ in self.projection}
        self.keep = np.array([i for i in range(configuration.dimension) if i not in excluded], dtype=int)
        self.dimension = len(self.keep)
        self.periods = dict(self.periodic)
        self.storage_periodic = tuple((j, self.periods[i]) for j, i in enumerate(self.keep) if i in self.periods)
        self.center = configuration.center[self.keep].copy()
        self.axes = np.array([j for j, i in enumerate(self.keep) if i not in self.periods], dtype=int)

    @property
    def active(self):
        return bool(self.projection or self.periodic)

    @property
    def dr(self):
        return float(np.linalg.norm(self.configuration.dl[self.keep[self.axes]]))

    def domain(self, cell):
        config = self.configuration
        result = Domain(config.lb[self.keep], config.ub[self.keep], np.asarray(cell)[self.keep], periodic=self.storage_periodic)
        result.coordinates = tuple(int(i) for i in self.keep)
        return result

    def wrap(self, points):
        out = np.array(points, dtype=np.float64, copy=True)
        for index, period in self.periodic:
            lower = self.configuration.lb[index]
            out[..., index] = lower + (out[..., index] - lower) % period
        return np.ascontiguousarray(out)

    def lift(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.dimension)
        lifted = np.tile(self.configuration.center, (len(points), 1))
        lifted[:, self.keep] = points
        for stored_axis, period in self.storage_periodic:
            index = self.keep[stored_axis]
            lower = self.configuration.lb[index]
            lifted[:, index] = lower + (lifted[:, index] - lower) % period
        return np.ascontiguousarray(lifted)

    def lift_directions(self, directions):
        out = np.zeros((len(directions), self.configuration.dimension), dtype=np.float64)
        out[:, self.keep] = directions
        return out

    def project(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, self.configuration.dimension)
        keep = np.isfinite(points).all(axis=1)
        for index, width in self.projection:
            distance = points[:, index] - self.configuration.center[index]
            if index in self.periods:
                period = self.periods[index]
                distance = (distance + period/2) % period - period/2
            keep &= np.abs(distance) < width*self.configuration.dl[index] + self.configuration.projection_epsilon
        return np.ascontiguousarray(self.wrap(points[keep])[:, self.keep])

    def embed_directions(self, directions):
        out = np.zeros((len(directions), self.dimension), dtype=np.float64)
        out[:, self.axes] = directions
        return out

    def origins(self, supplied=None):
        if supplied is not None:
            values = np.asarray(supplied, dtype=np.float64)
            if values.ndim == 1:
                values = values[None, :]
            if values.ndim != 2 or values.shape[1] != self.dimension or not len(values):
                raise ValueError('boundary origins must be a nonempty array of stored-coordinate points')
            return self.lift(values)[:, self.keep]
        if not self.storage_periodic:
            return self.center[None].copy()
        grid = self.domain(self.configuration.dl)
        levels = [grid.lb[j] + np.arange(grid.counts[j])*grid.cell[j] for j, _ in self.storage_periodic]
        origins = np.tile(self.center, (int(np.prod([len(level) for level in levels])), 1))
        for origin, values in zip(origins, product(*levels)):
            for (j, _), value in zip(self.storage_periodic, values):
                origin[j] = value
        return origins
