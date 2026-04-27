from __future__ import annotations
from typing import TypeVar, Generic, Literal, cast
from abc import ABC, abstractmethod
import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from dataclasses import dataclass

import numpy as np
from scipy import signal


from .types import *
from .signalutils.sosfilt import sosfilt, validate_sos
from .signalutils.resample import ResamplePoly, calc_tp_fir_win
from .typeutils import (
    ensure_1d_array, ensure_2d_array, is_3d_array, is_1d_array, is_float64_array,
)

T = TypeVar('T')


@dataclass
class Coeff:
    """Digital filter coefficients
    """
    b: Float1dArray             #: Numerator of the filter
    a: Float1dArray             #: Denominator of the filter
    sample_rate: int = 48000    #: Sample rate of the filter
    _sos: SosCoeff|None = None

    @classmethod
    def from_sos(cls, sos: SosCoeff, sample_rate: int = 48000) -> Self:
        """Create a :class:`Coeff` instance from second-order sections

        This is the inverse of :attr:`sos` property.
        """
        b, a = signal.sos2tf(sos)
        assert is_1d_array(b)
        assert is_1d_array(a)
        assert is_float64_array(b)
        assert is_float64_array(a)
        return cls(b=b, a=a, _sos=sos, sample_rate=sample_rate)

    @property
    def sos(self) -> SosCoeff:
        """Array of second-order sections calculated from the filter's transfer
        function form
        """
        if self._sos is None:
            s = ensure_2d_array(signal.tf2sos(self.b, self.a))
            assert s.shape[1] == 6
            s = validate_sos(s)
            self._sos = s
        return self._sos

    def combine(self, other: Self) -> Self:
        """Return a new :class:`Coeff` instance is a combination of this and
        another :class:`Coeff` instance

        Raises:
            ValueError: If the sample rates of the two :class:`Coeff` instances
                do not match

        """
        if self.sample_rate != other.sample_rate:
            raise ValueError(
                "Cannot combine Coeff instances with different sample rates"
            )
        sos1 = self.sos
        sos2 = other.sos
        combined_sos = np.vstack([sos1, sos2])
        num_sections, _ = combined_sos.shape
        assert combined_sos.shape == (num_sections, 6)
        combined_sos = cast(SosCoeff, combined_sos)
        return self.__class__.from_sos(
            sos=combined_sos,
            sample_rate=self.sample_rate,
        )

    def as_sample_rate(self, sample_rate: int) -> Self:
        """Return a new :class:`Coeff` instance with the coefficients converted
        to the specified sample rate
        """
        if sample_rate == self.sample_rate:
            return self
        # https://github.com/klangfreund/LUFSMeter/blob/783a59d78c31e52b3a50b52d9afaadf3118f7536/filters/SecondOrderIIRFilter.cpp#L47-L115
        b, a = self.b, self.a
        KoverQ = (2. - 2. * a[2]) / (a[2] - a[1] + 1.)
        K = np.sqrt((a[1] + a[2] + 1.) / (a[2] - a[1] + 1.))
        Q = K / KoverQ
        arctanK = np.atan(K)
        VB = (b[0] - b[2])/(1. - a[2])
        VH = (b[0] - b[1] + b[2])/(a[2] - a[1] + 1.)
        VL = (b[0] + b[1] + b[2])/(a[1] + a[2] + 1.)

        K = np.tan(arctanK * self.sample_rate / sample_rate)
        commonFactor = 1. / (1. + K/Q + K*K)
        b0 = (VH + VB*K/Q + VL*K*K)*commonFactor
        b1 = 2.*(VL*K*K - VH)*commonFactor
        b2 = (VH - VB*K/Q + VL*K*K)*commonFactor
        a1 = 2.*(K*K - 1.)*commonFactor
        a2 = (1. - K/Q + K*K)*commonFactor

        return self.__class__(
            b=ensure_1d_array(np.array([b0, b1, b2])),
            a=ensure_1d_array(np.array([1., a1, a2])),
            sample_rate=sample_rate,
        )


def design_HPF2(fc, Q, fs):
    """Calculates filter coefficients for a 2nd-order highpass filter

    Parameters
    ----------
    fc : float
        Cutoff frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    fs : float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    wc = 2 * np.pi * fc / fs
    c = 1.0 / np.tan(wc / 2.0)
    phi = c*c
    K = c / Q
    a0 = phi + K + 1.0

    b = [phi / a0, -2.0 * phi / a0, phi / a0]
    a = [1, 2.0 * (1.0 - phi) / a0, (phi - K + 1.0) / a0]
    return ensure_1d_array(np.asarray(b)), ensure_1d_array(np.asarray(a))


def design_highshelf(fc, Q, gain, fs):
    """Calculates filter coefficients for a High Shelf filter.

    Parameters
    ----------
    fc : float
        Center frequency of the filter in Hz
    Q : float
        Quality factor of the filter
    gain : float
        Linear gain for the shelved frequencies
    fs :  float
        Sample rate in Hz

    Returns
    -------
    b : ndarray
        "b" (feedforward) coefficients of the filter
    a : ndarray
        "a" (feedback) coefficients of the filter
    """
    A = np.sqrt(gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin(wc)
    wC = np.cos(wc)
    beta = np.sqrt(A) / Q

    a0 = ((A+1.0) - ((A-1.0) * wC) + (beta*wS))

    b = np.zeros(3)
    a = np.zeros(3)
    b[0] = A*((A+1.0) + ((A-1.0)*wC) + (beta*wS)) / a0
    b[1] = -2.0*A * ((A-1.0) + ((A+1.0)*wC)) / a0
    b[2] = A*((A+1.0) + ((A-1.0)*wC) - (beta*wS)) / a0

    a[0] = 1
    a[1] = 2.0 * ((A-1.0) - ((A+1.0)*wC)) / a0
    a[2] = ((A+1.0) - ((A-1.0)*wC)-(beta*wS)) / a0
    return b, a




@dataclass
class KFiltCoeff(Coeff):
    fc: float|None = None
    gain: float|None = None
    q: float|None = None
    is_high_shelf: bool|None = None

    @classmethod
    def design(cls, fc: float, gain: float, q: float, is_high_shelf: bool, sample_rate: int) -> Self:
        if is_high_shelf:
            b, a = design_highshelf(fc, q, gain, sample_rate)
        else:
            b, a = design_HPF2(fc, q, sample_rate)
        # f0, G, Q = fc, gain, q
        # K = np.tan(np.pi * f0 / sample_rate)
        # Vh = np.power(10.0, G / 20.0)
        # Vb = np.power(Vh, 0.4996667741545416)

        # pb = np.array([0., 0., 0.])
        # pa = np.array([1., 0., 0.])
        # rb = np.array([1.0, -2.0, 1.0])
        # ra = np.array([1., 0., 0.])
        # # if not is_high_shelf:
        # #     ra[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
        # #     ra[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
        # #     return cls(
        # #         b=np.array([Vh * (1.0 + K / Q + K * K), 2.0 * Vh * (K * K - 1.0), Vh * (1.0 - K / Q + K * K)]),
        # #         a=np.array([1., ra[1], ra[2]]),
        # #         sample_rate=sample_rate,
        # #     )

        # a0 = 1.0 + K / Q + K * K
        # pb[0] = (Vh + Vb * K / Q + K * K) / a0
        # pb[1] = 2.0 * (K * K - Vh) / a0
        # pb[2] = (Vh - Vb * K / Q + K * K) / a0
        # pa[1] = 2.0 * (K * K - 1.0) / a0
        # pa[2] = (1.0 - K / Q + K * K) / a0

        # if not is_high_shelf:
        #     rb[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
        #     rb[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
        #     b, a = rb, ra
        # else:
        #     b, a = pb, pa
        return cls(
            b=b,
            a=a,
            sample_rate=sample_rate,
            fc=fc,
            gain=gain,
            q=q,
            is_high_shelf=is_high_shelf,
        )

    def as_sample_rate(self, sample_rate: int) -> Self:
        assert self.fc is not None
        assert self.gain is not None
        assert self.q is not None
        assert self.is_high_shelf is not None
        return self.__class__.design(
            fc=self.fc,
            gain=self.gain,
            q=self.q,
            is_high_shelf=self.is_high_shelf,
            sample_rate=sample_rate,
        )
# class _HSCoeff(Coeff):
#     # def __init__(self):
#     #     super().__init__(
#     #         b = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
#     #         a = np.array([1.0, -1.69065929318241, 0.73248077421585]),
#     #         sample_rate=48000,
#     #     )

#     def as_sample_rate(self, sample_rate: int) -> Self:
#         if sample_rate == self.sample_rate:
#             return self

#         f0 = 1681.974450955533
#         G = 3.999843853973347
#         Q = 0.7071752369554196
#         K = np.tan(np.pi * f0 / sample_rate)
#         Vh = np.power(10.0, G / 20.0)
#         Vb = np.power(Vh, 0.4996667741545416)

#         pb = np.array([0., 0., 0.])
#         pa = np.array([1., 0., 0.])
#         rb = np.array([1.0, -2.0, 1.0])
#         ra = np.array([1., 0., 0.])

#         a0 = 1.0 + K / Q + K * K
#         b0 = (Vh + Vb * K / Q + K * K) / a0
#         b1 = 2.0 * (K * K - Vh) / a0
#         b2 = (Vh - Vb * K / Q + K * K) / a0
#         a1 = 2.0 * (K * K - 1.0) / a0
#         a2 = (1.0 - K / Q + K * K) / a0

#         return self.__class__(
#             b = np.array([b0, b1, b2]),
#             a = np.array([1., a1, a2]),
#             sample_rate=sample_rate,
#         )

# HS_COEFF = KFiltCoeff.design(
#     fc=1681.974450955533,
#     gain=3.999843853973347,
#     q=0.7071752369554196,
#     sample_rate=48000,
#     is_high_shelf=True,
# )

HS_COEFF = Coeff(
    b = ensure_1d_array(
        np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    ),
    a = ensure_1d_array(
        np.array([1.0, -1.69065929318241, 0.73248077421585]),
    ),
    sample_rate=48000,
)
"""Stage 1 (high-shelf) of the pre-filter defined in :term:`BS 1770` table 1"""

# HP_COEFF = KFiltCoeff.design(
#     fc=38.13547087602444,
#     gain=7.58475998141661,
#     q=0.5003270373238773,
#     sample_rate=48000,
#     is_high_shelf=False,
# )
HP_COEFF = Coeff(
    b = ensure_1d_array(
        np.array([1.0, -2.0, 1.0])
    ),
    a = ensure_1d_array(
        np.array([1.0, -1.99004745483398, 0.99007225036621])
    ),
    sample_rate=48000,
)
"""Stage 2 (high-pass) of the pre-filter defined in :term:`BS 1770` table 2"""

# BS-1771 coefficients for decimated 320 samples/s
# (128 samples per 400ms block)
MOMENTARY_HP_COEFF = Coeff(
    b = ensure_1d_array(
        np.array([1.0, 1.0])
    ),
    a = ensure_1d_array(
        np.array([1.0, -0.9921767002])
    )
)




def _check_filt_input(x: Float1dArray|Float2dArray) -> Float2dArray:
    """Ensure the input array is 2-dimensional, reshaping if necessary

    This is used for filters with possibly a single channel input.
    """
    if x.ndim == 1:
        assert x.shape[0] > 0
        return np.reshape(x, (1, x.shape[0]))
    assert x.shape[0] > 0
    return ensure_2d_array(x)


class BaseFilter(Generic[T, NumChannelsT], ABC):
    """
    """

    coeff: T
    """The filter coefficients"""

    num_channels: NumChannelsT
    """Number of audio channels to filter"""

    def __init__(self, coeff: T, num_channels: NumChannelsT) -> None:
        self.coeff = coeff
        self.num_channels = num_channels

    @abstractmethod
    def __call__(self, x: Float1dArray|Float2dArray) -> Float2dArray:
        """Apply the filter defined by :attr:`coeff` and return the result

        Arguments:
            x: The input data with shape ``(num_channels, n)`` where *n* is the
                length of the input data for each channel

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal filter conditions
        """

    def _check_arr_dims(self, x: Float1dArray|Float2dArray) -> Float2dArray:
        if x.ndim == 1:
            assert self.num_channels == 1
        return _check_filt_input(x)


class TruePeakFilter(BaseFilter[Float1dArray, NumChannelsT]):
    """4x Oversampling filter with interpolating FIR window
    """
    upsample_factor: int
    """Upsampling factor (currently only 4 is supported)"""
    def __init__(
        self,
        num_channels: NumChannelsT,
        upsample_factor: int = 4
    ) -> None:
        coeff = calc_tp_fir_win(upsample_factor)
        super().__init__(coeff=coeff, num_channels=num_channels)
        self.upsample_factor = upsample_factor
        self.resampler = ResamplePoly(
            up=upsample_factor,
            down=1,
            window=coeff,
            num_channels=num_channels,
        )

    def __call__(self, x: Float1dArray|Float2dArray) -> Float2dArray:
        x = self._check_arr_dims(x)
        return self.resampler.apply(x)

    def reset(self) -> None:
        pass


def _check_sos_zi(zi: AnyArray, num_channels: int) -> SosZI:
    assert is_3d_array(zi)
    assert zi.shape[1] == num_channels
    assert zi.shape[2] == 2
    assert zi.dtype == np.float64
    return cast(SosZI, zi)


class Filter(BaseFilter[Coeff, NumChannelsT]):
    """Multi-channel filter that tracks the filter conditions between calls

    The filter (defined by :attr:`coeff`) is applied by calling a :class:`Filter`
    instance directly.
    """
    sos_zi: SosZI
    """The filter conditions"""

    def __init__(self, coeff: Coeff, num_channels: NumChannelsT) -> None:
        super().__init__(coeff=coeff, num_channels=num_channels)
        zi = signal.sosfilt_zi(coeff.sos)
        zi[...] = 0
        sos_zi = np.repeat(np.expand_dims(zi, axis=1), num_channels, axis=1)

        # Make sos_zi contiguous along the section axis so that
        # :func:`.signalutils.sosfilt.sosfilt` can operate on it properly.
        axis = 1 # num_channels axis
        sos_zi = np.moveaxis(sos_zi, [0, axis + 1], [-2, -1])
        sos_zi = np.ascontiguousarray(sos_zi)
        sos_zi = np.moveaxis(sos_zi, [-2, -1], [0, axis + 1])
        self.sos_zi = _check_sos_zi(sos_zi, num_channels)

    def _sos(self, x: Float1dArray|Float2dArray) -> Float2dArray:
        zi = self.sos_zi
        x = self._check_arr_dims(x)

        y, zi = sosfilt(self.coeff.sos, x, axis=1, zi=zi)
        self.sos_zi = _check_sos_zi(zi, self.num_channels)
        return ensure_2d_array(y)

    def __call__(self, x: Float1dArray|Float2dArray) -> Float2dArray:
        return self._sos(x)

    def reset(self) -> None:
        self.sos_zi[...] = 0



class FilterGroup(Generic[NumChannelsT]):
    """Apply multiple :class:`filters <Filter>` in series

    Arguments:
        *coeff: The :class:`coefficients <Coeff>` to construct each :class:`Filter`
        num_channels: Number of channels to filter. This will also be set on
            the constructed :class:`filters <Filter>`

    """

    num_channels: NumChannelsT
    """Number of audio channels to filter"""

    def __init__(self, *coeff: Coeff, num_channels: NumChannelsT) -> None:
        self.num_channels = num_channels
        if len(coeff) > 1:
            combined = coeff[0]
            for c in coeff[1:]:
                combined = combined.combine(c)
            coeff = (combined,)
        self._filters = [Filter(c, num_channels) for c in coeff]

    def __call__(self, x: Float1dArray|Float2dArray) -> Float2dArray:
        """Apply the filters in series and return the result

        Arguments:
            x: The input data with shape ``(num_channels, n)`` where *n* is the
                length of the input data for each channel

        """
        if x.ndim == 1:
            assert self.num_channels == 1
        x = _check_filt_input(x)
        y = x
        for filt in self._filters:
            y = filt(y)
        return y

    def reset(self) -> None:
        """Reset the filter conditions for each filter in the group
        """
        for filt in self._filters:
            filt.reset()
