from __future__ import annotations

from typing import Generic, overload
import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from abc import ABC, abstractmethod

import numpy as np


from .types import *
from .typeutils import ensure_nd_array, ensure_meter_array, build_true_peak_array
from .filters import TruePeakFilter

__all__ = ('BlockProcessor', 'TruePeakProcessor')

SILENCE_DB: Floating = np.float64(-200.)
EPSILON: Floating = np.float64(1e-20)



class RunningSum:
    """Helper class to calculate the running sum of a series of values
    """
    __slots__ = ('_value', '_count', '_mean')
    def __init__(self) -> None:
        self._value: Floating = np.float64(0)
        self._count: int = 0
        self._mean: Floating|None = None

    @property
    def value(self) -> Floating:
        """The current running sum
        """
        return self._value
    @value.setter
    def value(self, value: Floating) -> None:
        if value == self._value:
            return
        self._value = value
        self._mean = None

    @property
    def count(self) -> int:
        """The number of values in the running sum
        """
        return self._count
    @count.setter
    def count(self, count: int) -> None:
        if count == self._count:
            return
        self._count = count
        self._mean = None

    @property
    def mean(self) -> Floating:
        """The mean of the running sum
        """
        m = self._mean
        if m is None:
            m = self._mean = self._value / self._count
        return m

    def add(self, value: Floating) -> None:
        """Add a data point to the running sum
        """
        self.value += value
        self.count += 1

    def clear(self) -> None:
        """Reset :attr:`value` and :attr:`count` to zero
        """
        self.value = np.float64(0)
        self.count = 0

    def __iadd__(self, value: Floating) -> Self:
        self.add(value)
        return self

    def __eq__(self, other: Floating) -> bool|np.bool:
        return self.value == other

    def __gt__(self, other: Floating) -> bool|np.bool:
        return self.value > other

    def __ge__(self, other: Floating) -> bool|np.bool:
        return self.value >= other

    def __lt__(self, other: Floating) -> bool|np.bool:
        return self.value < other

    def __le__(self, other: Floating) -> bool|np.bool:
        return self.value <= other

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self}>'

    def __str__(self) -> str:
        return f'value={float(self.value)}, count={self.count}'




@overload
def lk_log10(x: FloatArray, offset: float = -0.691, base: int = 10) -> FloatArray: ...
@overload
def lk_log10(x: np.floating, offset: float = -0.691, base: int = 10) -> np.floating: ...
def lk_log10(
    x: FloatArray|np.floating,
    offset: float = -0.691,
    base: int = 10
) -> FloatArray|np.floating:
    if isinstance(x, np.ndarray):
        x[np.less_equal(x, 0)] = EPSILON
    elif x <= 0:
        x = EPSILON
    r = offset + base * np.log10(x)
    return r

@overload
def from_lk_log10(x: FloatArray, offset: float = 0.691) -> FloatArray: ...
@overload
def from_lk_log10(x: np.floating, offset: float = 0.691) -> np.floating: ...
def from_lk_log10(
    x: FloatArray|np.floating,
    offset: float = 0.691
) -> FloatArray|np.floating:
    return 10 ** ((x + offset) / 10)

class BaseProcessor(ABC, Generic[NumChannelsT]):
    """
    """
    num_channels: NumChannelsT
    """Number of audio channels"""

    sample_rate: int
    """The sample rate of the audio data"""

    def __init__(self, num_channels: NumChannelsT, sample_rate: int = 48000) -> None:
        self.num_channels = num_channels
        self.sample_rate = sample_rate

    @abstractmethod
    def __call__(self, samples: Float2dArray) -> None:
        """Process one :term:`gating block`

        Input data must be of shape ``(num_channels, gate_size)``
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset all measurement data
        """
        raise NotImplementedError


class BlockProcessor(BaseProcessor[NumChannelsT]):
    """Process audio samples and store the resulting loudness data
    """

    gate_size: int
    """The length of one :term:`gating block` in samples"""

    integrated_lkfs: Floating
    """The current :term:`Integrated Loudness`"""

    lra: float
    """The current :term:`Loudness Range`"""

    MAX_BLOCKS = 144000 # <- 14400 seconds (4 hours) / .1 (100 milliseconds)
    _channel_weights = np.array([1, 1, 1, 1.41, 1.41])
    _block_data: MeterArray
    _Zij: Float2dArray
    _block_weighted_sums: Float1dArray
    _quarter_block_weighted_sums: Float1dArray
    _block_loudness: Float1dArray
    _blocks_above_abs_thresh: Any1dArray[np.dtype[np.bool_]]
    _blocks_above_rel_thresh: Any1dArray[np.dtype[np.bool_]]
    def __init__(
        self,
        num_channels: NumChannelsT,
        gate_size: int,
        sample_rate: int = 48000
    ) -> None:
        super().__init__(num_channels=num_channels, sample_rate=sample_rate)
        self.gate_size = gate_size
        self.pad_size = gate_size // 4
        self.weights = self._channel_weights[:self.num_channels]
        block_data = np.zeros(self.MAX_BLOCKS, dtype=MeterDtype)
        self._block_data = ensure_meter_array(block_data)

        self._Zij = np.zeros(
            (self.num_channels, self.MAX_BLOCKS),
            dtype=np.float64,
        )
        self._block_weighted_sums = np.zeros(self.MAX_BLOCKS, dtype=np.float64)
        self._quarter_block_weighted_sums = np.zeros(self.MAX_BLOCKS, dtype=np.float64)

        self._block_loudness = np.zeros(self.MAX_BLOCKS, dtype=np.float64)
        self._t = self._block_data['t']
        self._t[:] = np.arange(self.MAX_BLOCKS) / self.sample_rate * (self.gate_size / 4)


        self._blocks_above_abs_thresh = np.zeros(
            self.MAX_BLOCKS, dtype=bool
        )
        self._blocks_above_rel_thresh = np.zeros(
            self.MAX_BLOCKS, dtype=bool
        )
        self._above_abs_running_sum = RunningSum()
        self._above_rel_running_sum = RunningSum()
        self._rel_threshold: Floating = np.float64(SILENCE_DB)
        self._momentary_lkfs: Float1dArray = self._block_data['m']
        self._short_term_lkfs: Float1dArray = self._block_data['s']
        self.integrated_lkfs = SILENCE_DB
        self.lra = 0
        self.num_blocks = 0
        self.block_index = 0

    @property
    def block_data(self) -> MeterArray:
        """A structured array of measurement values with
        dtype :obj:`~.types.MeterDtype`
        """
        return self._block_data[:self.block_index]

    @property
    def momentary_lkfs(self) -> Float1dArray:
        """:term:`Momentary Loudness` for each 100ms block, averaged over 400ms
        (not gated)
        """
        return self.block_data['m']

    @property
    def short_term_lkfs(self) -> Float1dArray:
        """:term:`Short-Term Loudness` for each 100ms block, averaged over 3 seconds
        (not gated)
        """
        return self.block_data['s']

    @property
    def t(self) -> Float1dArray:
        """The measurement time for each element in :attr:`short_term_lkfs`
        and :attr:`momentary_lkfs`
        """
        return self.block_data['t']

    @property
    def Zij(self) -> Float2dArray:
        """Mean-squared values per channel in each 400ms block
        (not weighted)
        """
        zij = self._Zij[:,:self.block_index]
        return ensure_nd_array(zij, 2)

    def reset(self) -> None:
        """Reset all measurement data
        """
        self.block_data['m'][:] = 0
        self.block_data['s'][:] = 0
        self._Zij[...] = 0
        self._block_weighted_sums[:] = 0
        self._quarter_block_weighted_sums[:] = 0
        self._block_loudness[:] = 0
        self._rel_threshold = SILENCE_DB
        self._above_rel_running_sum.clear()
        self._above_abs_running_sum.clear()
        self.integrated_lkfs = SILENCE_DB
        self.lra = 0
        self.block_index = 0
        self.num_blocks = 0

    def __call__(self, samples: Float2dArray) -> None:
        self.process_block(samples)

    def __len__(self) -> int:
        return self.num_blocks

    def _calc_gating(self) -> None:
        block_lk = self._block_loudness[:self.block_index+1]
        block_wsums = self._block_weighted_sums[:self.block_index+1]
        cur_block_lk = block_lk[-1]
        cur_block_wsum = block_wsums[-1]
        above_abs = cur_block_lk >= -70

        if above_abs:
            self._above_abs_running_sum += cur_block_wsum

        if self._above_abs_running_sum == 0:
            rel_threshold = SILENCE_DB
        else:
            rel_threshold = lk_log10(self._above_abs_running_sum.mean) - 10
        self._rel_threshold = rel_threshold

        rs = self._above_rel_running_sum
        x = block_wsums[np.logical_and(
            np.greater_equal(block_lk, rel_threshold),
            np.greater_equal(block_lk, -70)
        )]
        rs.value = x.sum()
        rs.count = x.size
        if not rs.count:
            self.integrated_lkfs = SILENCE_DB
        else:
            self.integrated_lkfs = lk_log10(rs.mean)

    def _calc_momentary(self):
        block_index = self.block_index
        blocks = self._quarter_block_weighted_sums
        if block_index < 2:
            sl = slice(None, block_index+1)
            count = block_index + 1
        else:
            end_ix = self.block_index+1
            start_ix = end_ix - 3
            sl = slice(start_ix, end_ix)
            count = 3
        assert blocks.ndim == 1
        blocks = blocks[sl]
        assert blocks.size == count
        r = lk_log10(np.mean(blocks))
        if r < SILENCE_DB:
            r = SILENCE_DB
        self._momentary_lkfs[block_index] = r

    def __calc_short_term(self, block_index: int) -> Floating:
        num_blocks = 30    # 3 second window (100ms per block_index)
        blocks = self._quarter_block_weighted_sums[:block_index+1]

        if block_index < num_blocks:
            sl = slice(None, block_index+1)
            count = block_index + 1
        else:
            end_ix = block_index+1
            start_ix = end_ix - num_blocks
            sl = slice(start_ix, end_ix)
            count = num_blocks
        assert blocks.ndim == 1
        blocks = blocks[sl]
        assert blocks.size == count, f'{count=}, {blocks.size=}, {block_index=}, {sl=}'
        return lk_log10(np.mean(blocks))

    def _calc_short_term(self):
        block_index = self.block_index
        st = self.__calc_short_term(block_index)
        self._short_term_lkfs[block_index] = st

    def _calc_lra(self):
        block_index = self.block_index
        if block_index < 4:
            return
        st_loudness = self._short_term_lkfs[:block_index+1]
        abs_ix = np.greater_equal(st_loudness, -70)
        st_abs_gated = st_loudness[abs_ix]
        if not st_abs_gated.size:
            return
        st_abs_power = np.mean(from_lk_log10(st_abs_gated))

        st_integrated = lk_log10(st_abs_power)
        rel_threshold = st_integrated - 20
        rel_ix = np.greater_equal(st_abs_gated, rel_threshold)

        st_rel_gated = st_abs_gated[rel_ix]

        if not st_rel_gated.size:
            return
        lo_hi = np.quantile(st_rel_gated, [0.1, 0.95])

        self.lra = lo_hi[1] - lo_hi[0]

    def process_block(self, samples: Float2dArray):
        """Process one :term:`gating block`

        Input data must be of shape ``(num_channels, gate_size)``
        """
        assert samples.shape == (self.num_channels, self.gate_size)

        tg = 1 / self.gate_size
        sq_sum: Float1dArray = np.sum(np.square(samples), axis=1)

        _Zij = tg * sq_sum

        assert _Zij.shape == (self.num_channels,)
        weighted_sum = np.sum(_Zij * self.weights)
        self._block_weighted_sums[self.block_index] = weighted_sum

        block_loudness = lk_log10(weighted_sum)
        self._block_loudness[self.block_index] = block_loudness

        self._Zij[:,self.block_index] = _Zij

        self._calc_gating()

        self._process_quarter_block(samples)
        self._calc_momentary()
        self._calc_short_term()
        self._calc_lra()
        self.block_index += 1
        self.num_blocks += 1

    def _process_quarter_block(self, samples: Float2dArray):
        # Calculated the weighted squared-sums of the last 100ms segment within
        # this 400ms block. (for use in momentary calculation)
        # With an overlap of 75% in a 400ms block, the only
        # "new" samples will be the last 100ms

        quarter_blk_samples = samples[:,-self.pad_size:]
        assert quarter_blk_samples.shape[-1] == self.pad_size
        sq_sum: Float1dArray = np.sum(
            np.square(quarter_blk_samples), axis=1
        )
        tp = 1 / (self.pad_size)
        weighted_sum = np.sum((tp * sq_sum) * self.weights)
        self._quarter_block_weighted_sums[self.block_index] = weighted_sum


class TruePeakProcessor(BaseProcessor[NumChannelsT]):
    """Process audio samples to extract their :term:`True Peak` values
    """
    max_peak: Floating
    """Maximum :term:`True Peak` value detected"""

    gate_size: int
    """The length in samples to process per call"""

    MAX_TIME_SECONDS = 14400.0  # 4 hours

    def __init__(self, num_channels: NumChannelsT, gate_size: int, sample_rate: int = 48000) -> None:
        super().__init__(num_channels=num_channels, sample_rate=sample_rate)
        up_sample = 4 if sample_rate < 88100 else 2
        self.resample_filt = TruePeakFilter(
            num_channels=num_channels, upsample_factor=up_sample,
        )
        self.max_peak = SILENCE_DB
        self.gate_size = gate_size
        gate_t = gate_size / sample_rate
        max_blocks = int(self.MAX_TIME_SECONDS / gate_t)
        self._tp_array: TruePeakArray[NumChannelsT] = build_true_peak_array(
            num_channels=self.num_channels,
            size=max_blocks,
        )
        self._t = self._tp_array['t']
        self._t[:] = np.arange(max_blocks) * gate_t
        self._all_tp_values = self._tp_array['tp']
        self._all_tp_values[...] = SILENCE_DB
        self._block_index = 0

    @property
    def tp_array(self) -> TruePeakArray[NumChannelsT]:
        """A structured array of measurement values with
        dtype :obj:`~.types.TruePeakDtype`
        """
        return self._tp_array[:self._block_index]

    @property
    def t(self) -> Float1dArray:
        """The measurement times for all processed blocks in :attr:`tp_array`
        """
        return self.tp_array['t']

    @property
    def all_tp_values(self) -> np.ndarray[tuple[int, NumChannelsT], np.dtype[np.float64]]:
        """All :term:`True Peak` values per channel for each processed block
        """
        return self.tp_array['tp']

    @property
    def current_peaks(self) -> np.ndarray[tuple[NumChannelsT], np.dtype[np.float64]]:
        """:term:`True Peak` values per channel from the last processing period"""
        return self.all_tp_values[-1]

    def __call__(self, samples: Float2dArray) -> None:
        self.process(samples)

    def reset(self) -> None:
        self.max_peak = SILENCE_DB
        self._all_tp_values[0, :] = SILENCE_DB
        self._block_index = 0

    def process(self, samples: Float2dArray):
        tp_vals = self.resample_filt(samples)
        tp_amp_max = np.abs(tp_vals).max(axis=1)
        cur_peaks = lk_log10(tp_amp_max, offset=0, base=20)
        max_peak = cur_peaks.max()
        if max_peak > self.max_peak:
            self.max_peak = max_peak
        self._all_tp_values[self._block_index, :] = cur_peaks
        self._block_index += 1
