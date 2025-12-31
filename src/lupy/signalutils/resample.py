from __future__ import annotations
from typing import NamedTuple
import math

import numpy as np
from scipy.signal._upfirdn import _output_len, _UpFIRDn

from ..types import Float1dArray, Float2dArray
from ..typeutils import ensure_2d_array


class ResamplePolyParams(NamedTuple):
    """Parameters for polyphase resampling
    """
    up: int
    """Upsampling factor"""
    down: int
    """Downsampling factor"""
    n_in: int
    """Number of input samples"""
    n_out: int
    """Number of output samples"""
    h: np.ndarray
    """The padded FIR filter window"""
    result_slice: tuple[slice, ...]
    """Slice object to extract the valid output samples from
    :meth:`scipy.signal._upfirdn._UpFIRDn.apply_filter`
    """


# https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_signaltools.py#L3547-L3759
def calc_resample_poly_params(
    up: int,
    down: int,
    n_samples: int,
    window: Float1dArray
) -> ResamplePolyParams:
    """Calculate parameters for polyphase resampling

    Matches that of :func:`scipy.signal.resample_poly` but assumes window is
    already provided.

    Also assumes 2D input with the filter applied along the last axis.
    """
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_
    n_in = n_samples
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    assert window.ndim == 1
    half_len = (window.size - 1) // 2
    h = window * up
    # h *= up

    n_pre_pad = (down - half_len % down)
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in,
                      up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate((np.zeros(n_pre_pad, dtype=h.dtype), h,
                        np.zeros(n_post_pad, dtype=h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out
    result_slice = np.s_[:, n_pre_remove:n_pre_remove_end]
    return ResamplePolyParams(
        up,
        down,
        n_in,
        n_out,
        h,
        result_slice,
    )


class ResamplePoly:
    """Polyphase resampler using FIR window

    Precomputes parameters for efficient repeated resampling.

    Arguments:
        up: Upsampling factor
        down: Downsampling factor
        num_channels: Number of channels
        window: FIR filter window
        num_input_samples: Number of input samples. If not specified,
            defaults to 1024.
    """
    DEFAULT_INPUT_SAMPLES = 1024
    def __init__(
        self,
        up: int,
        down: int,
        num_channels: int,
        window: Float1dArray,
        num_input_samples: int = DEFAULT_INPUT_SAMPLES,
    ) -> None:
        self._up = up
        self._down = down
        self._num_channels = num_channels
        self._num_input_samples = num_input_samples
        self._window = window
        self.params = calc_resample_poly_params(
            up,
            down,
            num_input_samples,
            window,
        )

        # Create the _UpFIRDn instance as used in scipy:
        # https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_upfirdn.py#L214-L216
        #
        # instead of discarding it after each call to resample_poly
        # (which is quite wasteful).
        self._upfirdn = _UpFIRDn(
            h=self.params.h,
            x_dtype=np.float64,
            up=self.params.up,
            down=self.params.down,
        )

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return self._num_channels

    @property
    def num_input_samples(self) -> int:
        """Number of input samples per call to :meth:`apply`

        If this is changed, internal parameters are recalculated.
        """
        return self._num_input_samples
    @num_input_samples.setter
    def num_input_samples(self, value: int) -> None:
        if value == self._num_input_samples:
            return
        self._num_input_samples = value
        self.params = calc_resample_poly_params(
            up=self._up,
            down=self._down,
            n_samples=value,
            window=self._window,
        )
        self._upfirdn = _UpFIRDn(
            h=self.params.h,
            x_dtype=np.float64,
            up=self.params.up,
            down=self.params.down,
        )

    @property
    def num_output_samples(self) -> int:
        """Number of output samples"""
        return self.params.n_out

    def apply(self, x: Float2dArray) -> Float2dArray:
        """Resample the input array using polyphase filtering

        The input array must be 2D with shape ``(num_channels, num_input_samples)``.

        If the number of input samples differs from :attr:`num_input_samples`,
        internal parameters are recalculated.
        """
        num_samples = x.shape[-1]
        if num_samples != self._num_input_samples:
            self.num_input_samples = num_samples

        axis = 1
        y = self._upfirdn.apply_filter(
            x,
            axis=axis,
            mode='constant',
            cval=0
        )
        r = y[self.params.result_slice]
        return ensure_2d_array(r)



#https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_upfirdn.py#L45C1-L104C19
def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.

    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].

    Then the internal buffer will look like this::

       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)

    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = np.zeros(h_padlen, h.dtype)
    h_full[:len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full


def _check_mode(mode):
    mode = mode.lower()
    enum = mode_enum(mode)
    return enum


class _UpFIRDn:
    """Helper for resampling."""

    def __init__(self, h, x_dtype, up, down):
        h = np.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError('h must be 1-D with non-zero length')
        self._output_type = np.result_type(h.dtype, x_dtype, np.float32)
        h = np.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError('Both up and down must be >= 1')
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = np.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(self, x, axis=-1, mode='constant', cval=0):
        """Apply the prepared filter to the specified axis of N-D signal x."""
        output_len = _output_len(self._h_len_orig, x.shape[axis],
                                 self._up, self._down)
        # Explicit use of np.int64 for output_shape dtype avoids OverflowError
        # when allocating large array on platforms where intp is 32 bits.
        output_shape = np.asarray(x.shape, dtype=np.int64)
        output_shape[axis] = output_len
        out = np.zeros(output_shape, dtype=self._output_type, order='C')
        axis = axis % x.ndim
        mode = _check_mode(mode)
        _apply(np.asarray(x, self._output_type),
               self._h_trans_flip, out,
               self._up, self._down, axis, mode, cval)
        return out
