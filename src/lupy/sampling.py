from __future__ import annotations

from typing import TypeVar, NamedTuple
import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from abc import ABC, abstractmethod
from fractions import Fraction
import math
import threading

import numpy as np

from .types import *
from .typeutils import ensure_2d_array, is_float64_array
from .filters import FilterGroup, HS_COEFF, HP_COEFF

T = TypeVar('T')

__all__ = (
    'Sampler', 'TruePeakSampler',
    'ThreadSafeSampler', 'ThreadSafeTruePeakSampler',
)


class BufferShape(NamedTuple):
    total_samples: int
    """Total number of samples for the buffer"""

    block_size: int
    """The input block size"""

    num_blocks: int
    """Number of blocks (``total_samples // block_size``)"""

    pad_size: int
    """The padding (overlap) between each windowed :term:`gating block`"""

    gate_size: int
    """Total length in samples of each :term:`gating block`"""

    num_gate_blocks: int
    """Number of overlapping gating blocks that can be stored within
    :attr:`total_samples`
    """

    gate_step_size: int
    """The step size in samples between each overlapped :term:`gating block`
    """



class Slice:
    """Helper class to manage slicing of overlapping array chunks

    This can be used to slice overlapping or non-overlapping chunks
    from an array, wrapping around the end of the array as needed.

    For non-overlapping slices, set :attr:`overlap` to zero.

    Arguments:
        step: Length of each sliced array chunk
        overlap: Number of elements to repeat for each sliced array chunk
        max_index: Maximum index value before wrapping to zero
        index_: Initial index value (default 0)


    .. note::

        The naming of :attr:`step` and :attr:`overlap` is somewhat
        counter-intuitive.  :attr:`step` refers to the length of each
        sliced chunk (what would typically be called "window size"), while
        :attr:`overlap` refers to the number of elements to repeat between
        chunks (what would typically be called "step size").

    """
    step: int
    """Length of each sliced array chunk
    (this would be better named "win_size")
    """

    overlap: int
    """Number of elements to repeat for each sliced array chunk
    (this would be better named "step")
    """

    max_index: int
    """Maximum :attr:`index` value before wrapping to zero when :attr:`overlap` is zero.

    .. note::

        This has no effect when :attr:`overlap` is non-zero, since the slice will
        wrap around the end of the array as needed regardless of the index value.
    """
    def __init__(
        self,
        step: int,
        max_index: int,
        index_: int = 0,
        overlap: int = 0
    ) -> None:
        self._index = index_
        self.step = step
        self.overlap = overlap
        self.max_index = max_index
        self._start_index: int|None = None
        self._end_index: int|None = None

    @property
    def index(self) -> int:
        """The current index value"""
        return self._index
    @index.setter
    def index(self, value: int):
        if value > self.max_index:
            value = 0
        if value == self._index:
            return
        self._index = value
        self._start_index = None
        self._end_index = None

    @property
    def start_index(self) -> int:
        """The starting index of the current slice"""
        ix = self._start_index
        if ix is None:
            if self.overlap != 0:
                ix = self._start_index = self.index * self.overlap
            else:
                ix = self._start_index = self.index * self.step
            if ix < 0:
                ix = 0
        return ix

    @property
    def end_index(self) -> int:
        """The ending index of the current slice"""
        ix = self._end_index
        if ix is None:
            ix = self._end_index = self.start_index + self.step
        return ix

    def increment(self, x: AnyArray, axis: int) -> None:
        """Increment the slice to the next position, wrapping around the end
        of the array as needed

        Arguments:
            x: The array being sliced
            axis: The axis along which to slice
        """
        if self.index == 0:
            self.index += 1
            return
        if self.overlap != 0:
            start_ix = self.start_index + self.overlap
        else:
            start_ix = self.start_index + self.step
        if start_ix >= x.shape[axis]:
            self.index = 0
        else:
            self._index += 1
            self._start_index = start_ix
            self._end_index = None

    def is_wrapped(self, x: AnyArray, axis: int) -> bool:
        """Whether the current slice wraps around the end of the array

        Arguments:
            x: The array being sliced
            axis: The axis along which to slice
        """
        return self.end_index > x.shape[axis]

    def indices(self, arr_len: int) -> IndexArray:
        """Get an index array for the current slice, wrapping around
        the end of the array as needed

        Arguments:
            arr_len: Length of the array being sliced
        """
        a = np.arange(self.step, dtype=np.intp) + self.start_index
        a[a>=arr_len] -= arr_len
        return a

    def calc_shape(self, x: AnyArray, axis: int) -> tuple[int, ...]:
        """Calculate the shape of the sliced array along the specified axis

        Arguments:
            x: The array being sliced
            axis: The axis along which to slice
        """
        ndim = x.ndim
        if axis == ndim - 1:
            new_shape = list(x.shape[:-1])
            new_shape.append(self.step)
        else:
            new_shape = list(x.shape)
            ax_size = self.step
            new_shape[axis] = ax_size
            del new_shape[axis + 1]
        return tuple(new_shape)

    def build_slice_array(self, x: AnyArray, axis: int) -> tuple[slice|IndexArray, ...]:
        """Build a tuple of slices/indices for slicing the array along
        the specified axis

        If the slice wraps around the end of the array, an index array
        will be used for that axis.  Otherwise, a standard slice will be used.

        Arguments:
            x: The array being sliced
            axis: The axis along which to slice
        """
        start_ix: int|None
        start_ix, end_ix = self.start_index, self.end_index
        if start_ix == 0:
            start_ix = None
        sl_arr: list[slice|IndexArray] = [
            slice(None, None, None) for _ in range(x.ndim)
        ]
        ax_slice: slice|IndexArray
        if self.is_wrapped(x, axis):
            ax_slice = self.indices(x.shape[axis])
        else:
            ax_slice = slice(start_ix, end_ix)
        sl_arr[axis] = ax_slice
        return tuple(sl_arr)

    def slice(self, x: AnyArray, axis: int) -> AnyArray:
        """Get the current slice of the array along the specified axis

        Arguments:
            x: The array being sliced
            axis: The axis along which to slice
        """
        sl_arr = self.build_slice_array(x, axis)
        new_shape = self.calc_shape(x, axis)
        return np.reshape(x[sl_arr], new_shape)

    def __repr__(self) -> str:
        return f'<Slice: {self}>'

    def __str__(self) -> str:
        return str(self.index)


def calc_buffer_length(sample_rate: int, block_size: int) -> BufferShape:
    """Calculate an appropriate :class:`BufferShape` for the given
    sample rate and block size

    The :attr:`~BufferShape.total_samples` of the result will be chosen to
    divide evenly with both the :attr:`~BufferShape.block_size` and
    :attr:`~BufferShape.pad_size`, allowing for input and output views of the
    same array through :func:`reshaping <numpy.reshape>`
    """
    fs = Fraction(sample_rate, 1)
    overlap = Fraction(3, 4)
    step = 1 - overlap
    step_samp = fs * step
    assert step_samp % 1 == 0

    gate_len = Fraction(4, 10)
    pad_len = Fraction(1, 10)
    assert (sample_rate * gate_len) % 1 == 0
    assert (sample_rate * pad_len) % 1 == 0
    gate_size = int(sample_rate * gate_len)
    pad_size = int(sample_rate * pad_len)

    bfr_len = math.lcm(pad_size, block_size)
    while bfr_len <= gate_size:
        bfr_len *= 2

    assert bfr_len % 1 == 0
    bfr_len = int(bfr_len)
    assert bfr_len % block_size == 0
    num_blocks = bfr_len // block_size
    bfr_t = bfr_len / fs

    x = (bfr_t - gate_len) / (gate_len * step)
    num_gb = int(np.round(float(x)+1))

    return BufferShape(
        total_samples=bfr_len,
        block_size=block_size,
        num_blocks=num_blocks,
        pad_size=pad_size,
        gate_size=gate_size,
        num_gate_blocks=num_gb,
        gate_step_size=int(step_samp),
    )



class BaseSampler(ABC):
    sample_rate: Fraction
    """The sample rate of the input data"""

    block_size: int
    """Sample length per call to :meth:`write`"""

    num_channels: int
    """Number of channels"""

    sample_array: Float2dArray
    """Flat array to store samples waiting to process"""

    write_view: Float3dArray
    """View of :attr:`sample_array` with shape
    ``(num_channels, block_size, sample_array.shape[1] // block_size)``
    """
    def __init__(self, block_size: int, num_channels: int, sample_rate: int = 48000) -> None:
        self.block_size = block_size
        self.sample_rate = Fraction(sample_rate, 1)
        self.num_channels = num_channels

        self.bfr_shape = self._calc_buffer_shape()
        bfr_len = self.bfr_shape.total_samples

        self.sample_array = np.zeros(
            (num_channels, bfr_len),
            dtype=np.float64,
        )
        self.write_view = np.reshape(
            self.sample_array,
            (num_channels, self.num_blocks, self.block_size)
        )
        self.write_slice = Slice(self.block_size, max_index=self.num_blocks-1)
        self.samples_available = 0

    @property
    def num_blocks(self) -> int:
        """Alias for :attr:`BufferShape.num_blocks`"""
        return self.bfr_shape.num_blocks

    @property
    def total_samples(self) -> int:
        """Alias for :attr:`BufferShape.total_samples`"""
        return self.bfr_shape.total_samples

    @abstractmethod
    def _calc_buffer_shape(self) -> BufferShape:
        """Calculate the :class:`BufferShape` for this sampler"""
        raise NotImplementedError

    def write(self, samples: Float2dArray|Float2dArray32, apply_filter: bool = True) -> None:
        """Store input data into the internal buffer.

        The input data must be of shape ``(num_channels, block_size)``
        """
        assert samples.shape == (self.num_channels, self.block_size)
        self._write(samples)

    def _write(self, samples: Float2dArray|Float2dArray32) -> None:
        sl = self.write_slice
        self.write_view[:,sl.index,:] = samples
        sl.index += 1
        self.samples_available += samples.shape[1]

    def can_write(self) -> bool:
        """Whether there is enough room on the internal buffer for at least
        one call to :meth:`write`
        """
        return self.samples_available <= self.total_samples - self.block_size

    @abstractmethod
    def read(self) -> Float2dArray:
        """Read samples from the internal buffer"""
        raise NotImplementedError

    @abstractmethod
    def can_read(self) -> bool:
        """Whether there are enough samples to read"""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear the internal buffer"""
        self._clear()

    def _clear(self) -> None:
        self.samples_available = 0
        self.write_slice.index = 0


class Sampler(BaseSampler):
    """Allows input data to be stored in chunks of a specified length
    and read out in windowed segments as needed for :term:`gating block`
    calculations.
    """

    gate_view: Float2dArray
    """Sliding window view of :attr:`~BaseSampler.sample_array` with 75% overlap
    and shape ``(num_channels, gate_size, sample_array.shape[1] // gate_size)``
    """

    filter: FilterGroup
    """A :class:`~.filters.FilterGroup` with both stages of the pre-filter
    defined in :term:`BS 1770`
    """
    def __init__(self, block_size: int, num_channels: int, sample_rate: int = 48000) -> None:
        super().__init__(block_size, num_channels, sample_rate)
        self.gate_view = self.sample_array.view()
        self.gate_slice = Slice(
            step=self.gate_size,
            overlap=self.pad_size,
            max_index=self.num_gate_blocks,
        )

        coeff = [HS_COEFF, HP_COEFF]
        if sample_rate != 48000:
            coeff = [c.as_sample_rate(sample_rate) for c in coeff]
        self.filter = FilterGroup(*coeff, num_channels=self.num_channels)

    @property
    def gate_size(self) -> int:
        """Length of one gated block in samples (400ms)"""
        return self.bfr_shape.gate_size

    @property
    def pad_size(self) -> int:
        """Overlap amount per gated block in samples (100ms)"""
        return self.bfr_shape.pad_size

    @property
    def num_gate_blocks(self) -> int:
        """Alias for :attr:`BufferShape.num_gate_blocks`"""
        return self.bfr_shape.num_gate_blocks

    def _calc_buffer_shape(self) -> BufferShape:
        return calc_buffer_length(int(self.sample_rate), self.block_size)

    def write(self, samples: Float2dArray|Float2dArray32, apply_filter: bool = True) -> None:
        """Store input data into the internal buffer, optionally applying the
        :attr:`pre-filter <filter>`

        The input data must be of shape ``(num_channels, block_size)``
        """
        assert samples.shape == (self.num_channels, self.block_size)
        if apply_filter:
            if not is_float64_array(samples):
                samples = samples.astype(np.float64)
            samples = self.filter(samples)

        super().write(samples)

    def can_read(self) -> bool:
        """Whether there are enough samples in the internal buffer for at least
        one call to :meth:`read`
        """
        return self.samples_available >= self.gate_size

    def read(self) -> Float2dArray:
        """Get the samples for one :term:`gating block`
        """
        return self._read()

    def _read(self) -> Float2dArray:
        sl = self.gate_slice
        r: FloatArray = sl.slice(self.gate_view, axis=1)
        sl.increment(self.gate_view, axis=1)
        self.samples_available -= self.pad_size
        return ensure_2d_array(r)

    def _clear(self) -> None:
        super()._clear()
        self.gate_slice.index = 0
        self.filter.reset()


class TruePeakSampler(BaseSampler):
    """A :class:`Sampler` subclass for use with true peak sampling

    This sampler writes in the same way as :class:`Sampler`, but reads
    are not overlapping.

    The length of each read is determined by :attr:`gate_duration`.

    """
    gate_view: Float3dArray
    """View of :attr:`~BaseSampler.sample_array` with shape
    ``(num_channels, num_gate_blocks, gate_size)``
    """
    gate_duration: Fraction
    """Duration of each read in seconds. Default is 400ms.

    The chosen duration must be divisible by the sample rate.
    Shorter durations (e.g., 100ms) may be used for faster updates and *should*
    not affect the accuracy of the true peak measurement (within reason).

    The durations tested and confirmed to be accurate are: ``100ms, 200ms, 400ms, 800ms``.
    """
    def __init__(
        self,
        block_size: int,
        num_channels: int,
        sample_rate: int = 48000,
        gate_duration: Fraction = Fraction(4, 10)
    ) -> None:
        self.gate_duration = gate_duration
        super().__init__(block_size, num_channels, sample_rate)
        self.gate_view = np.reshape(
            self.sample_array,
            (num_channels, self.num_gate_blocks, self.gate_size)
        )
        self.gate_slice = Slice(
            step=self.gate_size,
            overlap=0,
            max_index=self.num_gate_blocks-1,
        )

    @property
    def gate_size(self) -> int:
        """Length of each read in samples, depending on :attr:`gate_duration`"""
        return self.bfr_shape.gate_size

    @property
    def num_gate_blocks(self) -> int:
        """Number of :attr:`gate_size` blocks that can be stored in the internal buffer"""
        return self.bfr_shape.num_gate_blocks

    def _calc_buffer_shape(self) -> BufferShape:
        fs = self.sample_rate
        gate_time = self.gate_duration
        assert (fs * gate_time) % 1 == 0
        gate_samples = int(fs * gate_time)
        bfr_len = math.lcm(self.block_size, gate_samples)
        if bfr_len == self.block_size:
            bfr_len *= 2
        assert bfr_len > self.block_size
        num_blocks = bfr_len // self.block_size
        assert bfr_len % gate_samples == 0
        num_gate_blocks = bfr_len // gate_samples
        return BufferShape(
            total_samples=bfr_len,
            block_size=self.block_size,
            num_blocks=num_blocks,
            pad_size=0, # No overlap for true peak sampling
            gate_size=gate_samples,
            num_gate_blocks=num_gate_blocks,
            gate_step_size=gate_samples,
        )

    def can_read(self) -> bool:
        """Whether there are enough samples in the internal buffer for at least
        one call to :meth:`read`
        """
        return self.samples_available >= self.gate_size

    def read(self) -> Float2dArray:
        """Get next available samples.

        The result will be of shape ``(num_channels, gate_size)``.
        """
        return self._read()

    def _read(self) -> Float2dArray:
        sl = self.gate_slice
        r: FloatArray = self.gate_view[:, sl.index, :]
        sl.index += 1
        assert r.shape == (self.num_channels, self.gate_size)
        self.samples_available -= self.gate_size
        return ensure_2d_array(r)

    def _clear(self) -> None:
        super()._clear()
        self.gate_slice.index = 0


class LockContext:
    """A mixin for context manager support using a :class:`threading.RLock`
    """
    _lock: threading.RLock

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire the underlying lock

        See :meth:`threading.Lock.acquire` for argument details
        """
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        """Release the underlying lock

        See :meth:`threading.Lock.release` for argument details
        """
        self._lock.release()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()


class ThreadSafeSampler(Sampler, LockContext):
    """A :class:`Sampler` subclass for use with threaded reads and writes
    """
    def __init__(self, block_size: int, num_channels: int, sample_rate: int = 48000) -> None:
        super().__init__(block_size, num_channels, sample_rate)
        self._lock = threading.RLock()

    def _write(self, samples: Float2dArray|Float2dArray32) -> None:
        with self:
            super()._write(samples)

    def _read(self) -> Float2dArray:
        with self:
            return super()._read()

    def _clear(self) -> None:
        with self:
            super()._clear()

class ThreadSafeTruePeakSampler(TruePeakSampler, LockContext):
    """A :class:`TruePeakSampler` subclass for use with threaded reads and writes
    """
    def __init__(
        self,
        block_size: int,
        num_channels: int,
        sample_rate: int = 48000,
        gate_duration: Fraction = Fraction(4, 10)
    ) -> None:
        super().__init__(block_size, num_channels, sample_rate, gate_duration)
        self._lock = threading.RLock()

    def _write(self, samples: Float2dArray|Float2dArray32) -> None:
        with self:
            super()._write(samples)

    def _read(self) -> Float2dArray:
        with self:
            return super()._read()

    def _clear(self) -> None:
        with self:
            super()._clear()
