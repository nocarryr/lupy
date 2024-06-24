from __future__ import annotations
from typing import TypeVar, Generic

from dataclasses import dataclass

import numpy as np
from scipy import signal

from lupy.types import FloatArray

from .types import *

T = TypeVar('T')


@dataclass
class Coeff:
    """Digital filter coefficients
    """
    b: FloatArray   #: Numerator of the filter
    a: FloatArray   #: Denominator of the filter
    _sos: FloatArray|None = None

    @property
    def sos(self) -> FloatArray:
        """Array of second-order sections calculated from the filter's transfer
        function form
        """
        s = self._sos
        if s is None:
            s = self._sos = signal.tf2sos(self.b, self.a)
        return s



HS_COEFF = Coeff(
    b = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
    a = np.array([1.0, -1.69065929318241, 0.73248077421585]),
)
"""Stage 1 (high-shelf) of the pre-filter defined in :term:`BS 1770` table 1"""

HP_COEFF = Coeff(
    b = np.array([1.0, -2.0, 1.0]),
    a = np.array([1.0, -1.99004745483398, 0.99007225036621]),
)
"""Stage 2 (high-pass) of the pre-filter defined in :term:`BS 1770` table 2"""

# BS-1771 coefficients for decimated 320 samples/s
# (128 samples per 400ms block)
MOMENTARY_HP_COEFF = Coeff(
    b = np.array([1.0, 1.0]),
    a = np.array([1.0, -0.9921767002])
)

# BS-1770 coefficients for 4-phase interpolating FIR (48k upsampled 4x)
TP_WIN: Float1dArray = np.array([
    # phase 1
    0.0017089843750, 0.0109863281250, -0.0196533203125, 0.0332031250000,
    -0.0594482421875, 0.1373291015625, 0.9721679687500, -0.1022949218750,
    0.0476074218750, -0.0266113281250, 0.0148925781250, -0.0083007812500,
    # phase 2
    -0.0291748046875, 0.0292968750000, -0.0517578125000, 0.0891113281250,
    -0.1665039062500, 0.4650878906250, 0.7797851562500, -0.2003173828125,
    0.1015625000000, -0.0582275390625, 0.0330810546875, -0.0189208984375,
    # phase 3
    -0.0189208984375, 0.0330810546875, -0.0582275390625, 0.1015625000000,
    -0.2003173828125, 0.7797851562500, 0.4650878906250, -0.1665039062500,
    0.0891113281250, -0.0517578125000, 0.0292968750000, -0.0291748046875,
    # phase 4
    -0.0083007812500, 0.0148925781250, -0.0266113281250, 0.0476074218750,
    -0.1022949218750, 0.9721679687500, 0.1373291015625, -0.0594482421875,
    0.0332031250000, -0.0196533203125, 0.0109863281250, 0.0017089843750,
])


class BaseFilter(Generic[T]):
    """
    """

    coeff: T
    """The filter coefficients"""

    num_channels: int
    """Number of audio channels to filter"""

    def __init__(self, coeff: T, num_channels: int = 1) -> None:
        self.coeff = coeff
        self.num_channels = num_channels

    def __call__(self, x: Float2dArray) -> Float2dArray:
        """Apply the filter defined by :attr:`coeff` and return the result

        Arguments:
            x: The input data with shape ``(num_channels, n)`` where *n* is the
                length of the input data for each channel

        """
        raise NotImplementedError


class TruePeakFilter(BaseFilter[Float1dArray]):
    """4x Oversampling filter with interpolating FIR window
    """
    upsample_factor: int
    """Upsampling factor (currently only 4 is supported)"""
    def __init__(
        self,
        coeff: Float1dArray = TP_WIN,
        num_channels: int = 1,
        upsample_factor: int = 1
    ) -> None:
        super().__init__(coeff=coeff, num_channels=num_channels)
        self.upsample_factor = upsample_factor

    def __call__(self, x: Float2dArray) -> Float2dArray:
        return signal.resample_poly(
            x,
            self.upsample_factor,
            self.upsample_factor,
            axis=1,
            window=TP_WIN
        )


class Filter(BaseFilter[Coeff]):
    """Multi-channel filter that tracks the filter conditions between calls

    The filter (defined by :attr:`coeff`) is applied by calling a :class:`Filter`
    instance directly.
    """
    sos_zi: FloatArray
    """The filter conditions"""

    def __init__(self, coeff: Coeff, num_channels: int = 1) -> None:
        super().__init__(coeff=coeff, num_channels=num_channels)
        zi = signal.sosfilt_zi(coeff.sos)
        zi[...] = 0
        self.sos_zi = np.repeat(np.expand_dims(zi, axis=1), num_channels, axis=1)

    def _sos(self, x: Float2dArray) -> Float2dArray:
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

    def __call__(self, x: Float2dArray) -> Float2dArray:
        return self._sos(x)


class FilterGroup:
    """Apply multiple :class:`filters <Filter>` in series

    Arguments:
        *coeff: The :class:`coefficients <Coeff>` to construct each :class:`Filter`
        num_channels: Number of channels to filter. This will also be set on
            the constructed :class:`filters <Filter>`

    """

    num_channels: int
    """Number of audio channels to filter"""

    def __init__(self, *coeff: Coeff, num_channels: int = 1):
        self.num_channels = num_channels
        self._filters = [Filter(c, num_channels) for c in coeff]

    def __call__(self, x: Float2dArray) -> Float2dArray:
        """Apply the filters in series and return the result

        Arguments:
            x: The input data with shape ``(num_channels, n)`` where *n* is the
                length of the input data for each channel

        """
        y = x
        for filt in self._filters:
            y = filt(y)
        return y
