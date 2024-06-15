from dataclasses import dataclass

import numpy as np
from scipy import signal

from .types import *



@dataclass
class Coeff:
    b: FloatArray
    a: FloatArray
    _sos: FloatArray|None = None

    @property
    def sos(self) -> FloatArray:
        s = self._sos
        if s is None:
            s = self._sos = signal.tf2sos(self.b, self.a)
        return s



HS_COEFF = Coeff(
    b = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
    a = np.array([1.0, -1.69065929318241, 0.73248077421585]),
)
HP_COEFF = Coeff(
    b = np.array([1.0, -2.0, 1.0]),
    a = np.array([1.0, -1.99004745483398, 0.99007225036621]),
)

# BS-1771 coefficients for decimated 320 samples/s
# (128 samples per 400ms block)
MOMENTARY_HP_COEFF = Coeff(
    b = np.array([1.0, 1.0]),
    a = np.array([1.0, -0.9921767002])
)


class Filter:
    sos_zi: FloatArray
    coeff: Coeff
    num_channels: int
    def __init__(self, coeff: Coeff, num_channels: int = 1):
        self.coeff = coeff
        zi = signal.sosfilt_zi(coeff.sos)
        zi[...] = 0
        self.sos_zi = np.repeat(np.expand_dims(zi, axis=1), num_channels, axis=1)
        self.num_channels = num_channels

    def _sos(self, x: FloatArray) -> FloatArray:
        zi = self.sos_zi

        n_dim = x.ndim
        if n_dim == 1:
            assert self.num_channels == 1
            x = np.reshape(x, (1, *x.shape))
            axis = 1
        else:
            assert n_dim == 2
            assert x.shape[0] == self.num_channels
            axis = 1

        y, zi = signal.sosfilt(self.coeff.sos, x, axis=axis, zi=zi)
        self.sos_zi = zi
        return y

    def __call__(self, x: FloatArray) -> FloatArray:
        return self._sos(x)


class FilterGroup:
    num_channels: int
    def __init__(self, num_channels: int = 1):
        self.num_channels = num_channels
        self.pre_filter = Filter(HS_COEFF, num_channels)
        self.lp_filter = Filter(HP_COEFF, num_channels)


    def __call__(self, x: FloatArray) -> FloatArray:
        y = self.pre_filter(x)
        return self.lp_filter(y)
