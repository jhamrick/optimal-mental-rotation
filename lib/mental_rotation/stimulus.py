import numpy as np
import json
from copy import copy
from path import path
from . import DTYPE


class Stimulus2D(object):

    def __init__(self, vertices, sort=True):
        if vertices.ndim != 2:
            raise ValueError("vertex array must be 2D")
        if vertices.shape[1] != 2:
            raise ValueError("vertex array must be n-by-2")

        # order them by angle, so they plot nicely
        if sort:
            r = np.arctan2(vertices[:, 1], vertices[:, 0])
            idx = np.argsort(r)
            self._v = np.array(vertices[idx], dtype=DTYPE, copy=True)
        else:
            self._v = np.array(vertices, dtype=DTYPE, copy=True)

        self.operations = []

    @property
    def vertices(self):
        v = self._v.copy()
        for op, arg in self.operations:
            getattr(self, "_%s" % op)(v, arg)
        return v

    def __getstate__(self):
        state = {}
        state['vertices'] = self._v.copy().tolist()
        state['operations'] = copy(self.operations)
        return state

    def __setstate__(self, state):
        self._v = np.array(state['vertices'], dtype=DTYPE)
        self.operations = state['operations']

    @staticmethod
    def _rotate(v, theta):
        theta_rad = np.radians(theta)
        M = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]], dtype=DTYPE)
        v[:] = np.dot(v, M.T)

    def rotate(self, theta):
        """Rotate the stimulus by `theta` degrees."""
        self.operations.append(["rotate", float(theta)])

    @staticmethod
    def _flip(v, axis):
        M = np.array([
            [axis[0]**2 - axis[1]**2, 2*axis[0]*axis[1]],
            [2*axis[0]*axis[1], axis[1]**2 - axis[0]**2]], dtype=DTYPE)
        M /= axis[0]**2 + axis[1]**2
        v[:] = np.dot(v, M.T)

    def flip(self, axis):
        """Flip the stimulus across the vector given by `axis`."""
        self.operations.append(["flip", [float(x) for x in axis]])

    @classmethod
    def random(cls, npoints):
        """Make a random shape with `npoints` vertices."""

        # if npoints is a number, then we generate that many vertices,
        # otherwise it should be a 2-tuple specifying an interval of
        # possible values
        try:
            lower, upper = npoints
        except TypeError:
            pass
        else:
            npoints = np.random.randint(lower, upper+1)

        # pick random points
        X_ = np.random.rand(npoints, 2)
        # normalize points
        X = X_ - np.mean(X_, axis=0)
        # normalize the shape's size, so the furthest point is distance 1
        # away from the origin
        X = 0.9 * X / np.max(np.sqrt(np.sum(X ** 2, axis=1)))

        # create the stimulus
        stim = cls(X)
        return stim

    def save(self, filename, force=False):
        filename = path(filename)
        if filename.exists() and not force:
            raise IOError("'%s' already exists", filename.relpath())
        state = self.__getstate__()
        with open(filename, "w") as fh:
            json.dump(state, fh)

    @classmethod
    def load(cls, filename):
        filename = path(filename)
        with open(filename, "r") as fh:
            state = json.load(fh)
        stim = cls.__new__(cls)
        stim.__setstate__(state)
        return stim

    def __eq__(self, other):
        if (self._v != other._v).any():
            return False
        if self.operations != other.operations:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def plot(self, ax, **kwargs):
        v = self.vertices
        X = np.empty((v.shape[0] + 1, 2))
        X[:-1] = v
        X[-1] = v[0]

        ax.plot(
            X[:, 0], X[:, 1],
            color='k',
            linewidth=2,
            **kwargs)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.axis([-1, 1, -1, 1])
        ax.set_aspect('equal')
        ax.axis('off')

    def copy_from_state(self):
        state = self.__getstate__()
        cls = type(self)
        stim = cls.__new__(cls)
        stim.__setstate__(state)
        return stim

    def copy_from_vertices(self):
        stim = type(self)(self.vertices, sort=False)
        return stim

    def copy_from_initial(self):
        stim = type(self)(self._v.copy(), sort=False)
        return stim
