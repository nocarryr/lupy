from __future__ import annotations
from typing import NamedTuple
import math

import numpy as np
from scipy.signal._upfirdn_apply import _output_len, _apply, mode_enum

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
        self._in_dtype = np.dtype(np.float64)
        self._out_dtype = np.dtype(np.float64)
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
            x_dtype=self._in_dtype,
            out_dtype=self._out_dtype,
            up=self.params.up,
            down=self.params.down,
            input_shape=self.input_shape,
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
            x_dtype=self._in_dtype,
            out_dtype=self._out_dtype,
            up=self.params.up,
            down=self.params.down,
            input_shape=self.input_shape,
        )

    @property
    def num_output_samples(self) -> int:
        """Number of output samples"""
        return self.params.n_out

    @property
    def input_shape(self) -> tuple[int, int]:
        """Expected input shape for :meth:`apply`"""
        return (self.num_channels, self.num_input_samples)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Output shape for :meth:`apply`"""
        return (self.num_channels, self.num_output_samples)

    def apply(self, x: Float2dArray) -> Float2dArray:
        """Resample the input array using polyphase filtering

        The input array must be 2D with shape ``(num_channels, num_input_samples)``.

        If the number of input samples differs from :attr:`num_input_samples`,
        internal parameters are recalculated.
        """
        num_samples = x.shape[-1]
        if num_samples != self._num_input_samples:
            self.num_input_samples = num_samples

        y = self._upfirdn.apply_filter(x)
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




class _UpFIRDn:
    """Helper for resampling."""
    mode = mode_enum('constant')

    def __init__(self, h, x_dtype, out_dtype, up, down, input_shape):
        if h.ndim != 1 or h.size == 0:
            raise ValueError('h must be 1-D with non-zero length')
        self._x_dtype = x_dtype
        self._output_type = out_dtype
        assert h.dtype == self._output_type
        self._up = int(up)
        self._down = int(down)
        self._axis = 1
        self._input_shape = input_shape
        assert len(self._input_shape) == 2
        self._input_len = int(self._input_shape[self._axis])
        if self._up < 1 or self._down < 1:
            raise ValueError('Both up and down must be >= 1')
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = np.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)
        output_len = _output_len(
            self._h_len_orig, self._input_len,
            self._up, self._down
        )
        self._output_len = int(output_len)
        self._output_shape = (
            self._input_shape[0], self._output_len
        )
        self._out_array = np.zeros(self._output_shape, dtype=self._output_type, order='C')

    def apply_filter(self, x):
        """Apply the prepared filter to the specified axis of N-D signal x."""
        out = self._out_array
        out[...] = 0
        if x.dtype != self._output_type:
            _x = np.asarray(x, self._output_type)
        else:
            _x = x

        _apply(_x,
               self._h_trans_flip, out,
               self._up, self._down, self._axis, self.mode, 0)
        return out
