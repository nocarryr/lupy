from __future__ import annotations

from typing import TypeVar, NamedTuple
import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from fractions import Fraction
import threading

from lupy.types import FloatArray
import numpy as np

from .types import *
from .filters import FilterGroup, HS_COEFF, HP_COEFF

T = TypeVar('T')

__all__ = ('Sampler', 'ThreadSafeSampler')


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



def find_even_fractions(fr1: Fraction, fr2: Fraction, max_iter: int = 100):
    orig_fr1, orig_fr2 = fr1, fr2
    i = 0
    while i < max_iter:
        if fr1 == fr2:
            return fr1
        if fr1 < fr2:
            j = 0
            while fr1 < fr2:
                fr1 = fr1 + orig_fr1
                j += 1
                if j >= max_iter:
                    raise ValueError(f'max iter: {fr1=}, {fr2=}')
        else:
            j = 0
            while fr2 < fr1:
                fr2 = fr2 + orig_fr2
                j += 1
                if j >= max_iter:
                    raise ValueError(f'max iter: {fr1=}, {fr2=}')

        i += 1
    raise ValueError(f'no match found: {i=}, {fr1=}, {fr2=}')


class Slice:
    step: int
    """Length of each sliced array chunk
    (this would be better named "win_size")
    """

    overlap: int
    """Number of elements to repeat for each sliced array chunk
    (this would be better named "step")
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
        ix = self._start_index
        if ix is None:
            ix = self._start_index = self.index * self.overlap
            if ix < 0:
                ix = 0
        return ix

    @property
    def end_index(self) -> int:
        ix = self._end_index
        if ix is None:
            ix = self._end_index = self.start_index + self.step
        return ix

    def increment(self, x: AnyArray, axis: int) -> None:
        if self.index == 0:
            self.index += 1
            return
        start_ix = self.start_index + self.overlap
        if start_ix >= x.shape[axis]:
            self.index = 0
        else:
            self._index += 1
            self._start_index = start_ix
            self._end_index = None

    def is_wrapped(self, x: AnyArray, axis: int) -> bool:
        return self.end_index > x.shape[axis]

    def indices(self, arr_len: int) -> IndexArray:
        a = np.arange(self.step, dtype=np.intp) + self.start_index
        a[a>=arr_len] -= arr_len
        return a

    def _calc_shape(self, x: AnyArray, axis: int):
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

    def _build_slice_array(self, x: AnyArray, axis: int):
        start_ix, end_ix = self.start_index, self.end_index
        if start_ix == 0:
            start_ix = None
        sl_arr: list[slice|IndexArray] = [
            slice(None, None, None) for _ in range(x.ndim)
        ]
        if self.is_wrapped(x, axis):
            ax_slice = self.indices(x.shape[axis])
        else:
            ax_slice = slice(start_ix, end_ix)
        sl_arr[axis] = ax_slice
        return tuple(sl_arr)

    def slice(self, x: AnyArray, axis: int) -> AnyArray:
        sl_arr = self._build_slice_array(x, axis)
        new_shape = self._calc_shape(x, axis)
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
    one_sample = Fraction(1, sample_rate)
    fs = Fraction(sample_rate, 1)
    overlap = Fraction(3, 4)
    step = 1 - overlap
    step_samp = fs * step
    assert step_samp % 1 == 0

    block_len = one_sample * block_size
    gate_len = Fraction(4, 10)
    pad_len = Fraction(1, 10)
    assert (sample_rate * gate_len) % 1 == 0
    assert (sample_rate * pad_len) % 1 == 0
    gate_size = int(sample_rate * gate_len)
    pad_size = int(sample_rate * pad_len)

    max_fr = find_even_fractions(pad_len, block_len)

    bfr_len = sample_rate * max_fr
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



class Sampler:
    """Allows input data to be stored in chunks of a specified length
    and read out in windowed segments as needed for :term:`gating block`
    calculations.
    """
    sample_rate: Fraction
    """The sample rate of the input data"""

    num_channels: int
    """Number of channels"""

    sample_array: FloatArray
    """Flat array to store samples waiting to process"""

    write_view: FloatArray
    """View of :attr:`sample_array` with shape
    ``(num_channels, block_size, sample_array.shape[1] // block_size)``
    """

    gate_view: FloatArray
    """Sliding window view of :attr:`sample_array` with 75% overlap and shape
    ``(num_channels, gate_size, sample_array.shape[1] // gate_size)``
    """

    filter: FilterGroup
    """A :class:`~.filters.FilterGroup` with both stages of the pre-filter
    defined in :term:`BS 1770`
    """
    def __init__(self, block_size: int, num_channels: int, sample_rate: int = 48000) -> None:
        self.sample_rate = Fraction(sample_rate, 1)
        self.num_channels = num_channels

        bfr_shape = self.bfr_shape = calc_buffer_length(
            int(self.sample_rate), block_size,
        )
        bfr_len = bfr_shape.total_samples

        self.sample_array = np.zeros(
            (num_channels, bfr_len),
            dtype=np.float64,
        )

        self.write_view = np.reshape(
            self.sample_array,
            (num_channels, self.num_blocks, self.block_size)
        )
        self.gate_view = self.sample_array.view()

        self.write_slice = Slice(self.block_size, max_index=self.num_blocks-1)
        self.gate_slice = Slice(
            step=self.gate_size,
            overlap=self.pad_size,
            max_index=self.num_gate_blocks,
        )

        self.samples_available = 0
        coeff = [HS_COEFF, HP_COEFF]
        if sample_rate != 48000:
            coeff = [c.as_sample_rate(int(sample_rate)) for c in coeff]
        self.filter = FilterGroup(*coeff, num_channels=self.num_channels)

    @property
    def block_size(self) -> int:
        """Sample length per call to :meth:`write`
        """
        return self.bfr_shape.block_size

    @property
    def num_blocks(self) -> int:
        """Alias for :attr:`BufferShape.num_blocks`"""
        return self.bfr_shape.num_blocks

    @property
    def total_samples(self) -> int:
        """Alias for :attr:`BufferShape.total_samples`"""
        return self.bfr_shape.total_samples

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

    def write(self, samples: FloatArray, apply_filter: bool = True) -> None:
        """Store input data into the internal buffer, optionally appling the
        :attr:`pre-filter <filter>`

        The input data must be of shape ``(num_channels, block_size)``
        """
        assert samples.shape == (self.num_channels, self.block_size)
        if apply_filter:
            samples = self.filter(samples)

        self._write(samples)

    def _write(self, samples: FloatArray) -> None:
        sl = self.write_slice
        self.write_view[:,sl.index,:] = samples
        sl.index += 1
        self.samples_available += samples.shape[1]

    def can_write(self) -> bool:
        """Whether there is enough room on the internal buffer for at least
        one call to :meth:`write`
        """
        return self.samples_available <= self.total_samples - self.block_size

    def can_read(self) -> bool:
        """Whether there are enough samples in the internal buffer for at least
        one call to :meth:`read`
        """
        return self.samples_available >= self.gate_size

    def read(self) -> FloatArray:
        """Get the samples for one :term:`gating block`
        """
        return self._read()

    def _read(self) -> FloatArray:
        sl = self.gate_slice
        r: FloatArray = sl.slice(self.gate_view, axis=1)
        sl.increment(self.gate_view, axis=1)
        self.samples_available -= self.pad_size
        return r

    def clear(self) -> None:
        """Clear all samples and reset internal tracking variables
        """
        self._clear()

    def _clear(self) -> None:
        self.sample_array[...] = 0
        self.samples_available = 0
        self.write_slice.index = 0
        self.gate_slice.index = 0


class ThreadSafeSampler(Sampler):
    """A :class:`Sampler` subclass for use with threaded reads and writes
    """
    def __init__(self, block_size: int, num_channels: int, sample_rate: int = 48000) -> None:
        super().__init__(block_size, num_channels, sample_rate)
        self._lock = threading.RLock()

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

    def _write(self, samples: FloatArray) -> None:
        with self:
            super()._write(samples)

    def _read(self) -> FloatArray:
        with self:
            return super()._read()

    def _clear(self) -> None:
        with self:
            super()._clear()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()
