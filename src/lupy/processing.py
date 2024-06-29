from __future__ import annotations

from typing import overload, cast
from abc import ABC, abstractmethod

import numpy as np
# import numpy.ma as ma


from .types import *
from .filters import Filter, TruePeakFilter, MOMENTARY_HP_COEFF

__all__ = ('BlockProcessor', 'TruePeakProcessor')

SILENCE_DB: Floating = np.float64(-200.)
EPSILON: Floating = np.float64(1e-20)





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

class BaseProcessor(ABC):
    """
    """
    num_channels: int
    """Number of audio channels"""

    sample_rate = 48000

    def __init__(self, num_channels: int) -> None:
        self.num_channels = num_channels

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


class BlockProcessor(BaseProcessor):
    """Process audio samples and store the resulting loudness data
    """

    gate_size: int
    """The length of one :term:`gating block` in samples"""

    integrated_lkfs: Floating
    """The current :term:`Integrated Loudness`"""

    lra: float
    """The current :term:`Loudness Range`"""

    MAX_BLOCKS = 36000 # <- 14400 seconds (4 hours) / .4 (400 milliseconds)
    _channel_weights = np.array([1, 1, 1, 1.41, 1.41])
    def __init__(self, num_channels: int, gate_size: int):
        super().__init__(num_channels=num_channels)
        self.gate_size = gate_size
        self.pad_size = gate_size // 4
        self.weights = self._channel_weights[:self.num_channels]
        # self._block_data: MeterArray = np.zeros(self.MAX_BLOCKS, dtype=MeterDtype)
        block_data = np.zeros(
            self.MAX_BLOCKS,
            dtype=MeterDtype,
            # dtype=[
            #     ('t', 'f8'),
            #     ('m', 'f8'),
            #     ('s', 'f8')
            # ]
        )
        # self._block_data = cast(MeterArray, block_data)
        self._block_data: MeterArray = block_data.view(np.recarray)

        self._Zij: Float2dArray = np.zeros(
            (self.num_channels, self.MAX_BLOCKS),
            # (self.MAX_BLOCKS, self.num_channels),
            dtype=np.float64,
        )
        self._block_weighted_sums: FloatArray = np.zeros(self.MAX_BLOCKS, dtype=np.float64)
        self._quarter_block_weighted_sums = np.zeros(self.MAX_BLOCKS, dtype=np.float64)

        self._block_loudness: FloatArray = np.zeros(self.MAX_BLOCKS, dtype=np.float64)
        # self._t: FloatArray = np.arange(self.MAX_BLOCKS) / self.sample_rate * (self.gate_size / 4)
        self._t: Float1dArray = self._block_data['t']
        self._t[:] = np.arange(self.MAX_BLOCKS) / self.sample_rate * (self.gate_size / 4)


        self._blocks_above_abs_thresh: BoolArray = np.zeros(
            self.MAX_BLOCKS, dtype=bool
        )
        self._blocks_above_rel_thesh: BoolArray = np.zeros(
            self.MAX_BLOCKS, dtype=bool
        )
        self._rel_threshold: Floating = np.float64(SILENCE_DB)
        # self._momentary_lkfs: Float1dArray = np.zeros(self.MAX_BLOCKS, dtype=np.float64)
        self._momentary_lkfs: Float1dArray = self._block_data['m']
        # self._short_term_lkfs: Float1dArray = np.zeros(self.MAX_BLOCKS, dtype=np.float64)
        self._short_term_lkfs: Float1dArray = self._block_data['s']
        self.integrated_lkfs = SILENCE_DB
        # self.lra = LRA(SILENCE_DB, SILENCE_DB)
        self.lra = 0
        self.num_blocks = 0
        self.block_index = 0
        self._short_term_filter = Filter(MOMENTARY_HP_COEFF, num_channels=1)

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
        # return self._momentary_lkfs[:self.block_index]
        return self.block_data['m']

    @property
    def short_term_lkfs(self) -> Float1dArray:
        """:term:`Short-Term Loudness` for each 100ms block, averaged over 3 seconds
        (not gated)
        """
        return self.block_data['s']
        # return self._short_term_lkfs[:self.block_index]

    # @property
    # def lkfs(self) -> Floating:
    #     """Integrated loudness
    #     """
    #     if not len(self):
    #         return SILENCE_DB
    #     return self._lkfs[-1]

    @property
    def t(self) -> Float1dArray:
        """The measurement time for each element in :attr:`short_term_lkfs`
        and :attr:`momentary_lkfs`
        """
        return self.block_data['t']
        # return self._t[:self.block_index]

    @property
    def Zij(self) -> Float1dArray:
        """Mean-squared values per channel in each 400ms block
        (not weighted)
        """
        return self._Zij[:,:self.block_index]

    def reset(self) -> None:
        """Reset all measurement data
        """
        self.block_data['m'][:] = 0
        self.block_data['s'][:] = 0
        self._Zij[:] = 0
        self._block_weighted_sums[:] = 0
        self._quarter_block_weighted_sums[:] = 0
        self._block_loudness[:] = 0
        self._rel_threshold = SILENCE_DB
        self.block_index = 0
        self.num_blocks = 0

    def __call__(self, samples: Float2dArray) -> None:
        self.process_block(samples)

    def __len__(self) -> int:
        return self.num_blocks

    # @property
    # def block_loudness(self):
    #     return self._block_loudness[:self.block_index]

    def _calc_relative_threshold(self) -> bool:
        # prev_threshold = self._rel_threshold
        ix = self._blocks_above_abs_thresh[:self.block_index+1]
        block_lk = self._block_loudness[:self.block_index+1]
        block_wsums = self._block_weighted_sums[:self.block_index+1]
        # ix = np.greater_equal(block_lk, -70)
        # blocks: Float1dArray = self._block_loudness[:self.block_index+1]
        # blocks: Float1dArray = self._block_weighted_sums[:self.block_index+1]

        # J_g = blocks[ix]
        J_g: Float1dArray = block_wsums[ix]
        if not J_g.size:
            # print('J_g empty')
            rel_threshold = SILENCE_DB
        else:
            # print(f'{np.mean(J_g)=}')
            # m = np.mean(J_g)
            rel_threshold = lk_log10(np.mean(J_g)) - 10
        changed = rel_threshold == self._rel_threshold
        self._rel_threshold = rel_threshold
        if False:#not changed:
            above_thresh = block_lk[self.block_index] >= self._rel_threshold
            self._blocks_above_rel_thesh[self.block_index] = above_thresh
        else:
            # try:
            self._blocks_above_rel_thesh[:self.block_index+1] = np.greater_equal(
                block_lk, self._rel_threshold
            )
            # except ValueError:
            #     print(f'{ix.shape=}, {J_g.shape=}, {np.greater_equal(J_g, self._rel_threshold).shape=}, {self._blocks_above_rel_thesh[:self.block_index+1].shape=}')
            #     raise
        return changed

    def _calc_integrated(self, rel_changed: bool):
        if False:#not rel_changed:
            if (not self._blocks_above_abs_thresh[self.block_index] and
                not self._blocks_above_rel_thesh[self.block_index]):
                return
        abs_ix = self._blocks_above_abs_thresh[:self.block_index+1]
        rel_ix = self._blocks_above_rel_thesh[:self.block_index+1]
        # block_lk = self._block_loudness[:self.block_index+1]
        block_wsums = self._block_weighted_sums[:self.block_index+1]

        ix = np.logical_and(abs_ix, rel_ix)

        J_g = block_wsums[ix]
        if not J_g.size:
            self.integrated_lkfs = SILENCE_DB
        else:
            self.integrated_lkfs = lk_log10(np.mean(J_g))
            # self.integrated_lkfs = np.mean(J_g)

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
        st_abs_power = np.mean(from_lk_log10(st_abs_gated))#, offset=0))

        st_integrated = lk_log10(st_abs_power)#, offset=0)
        rel_threshold = st_integrated - 20
        rel_ix = np.greater_equal(st_abs_gated, rel_threshold)

        st_rel_gated = st_abs_gated[rel_ix]

        if not st_rel_gated.size:
            return
        lo_hi = np.percentile(st_rel_gated, [10, 95])

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
        Zij = self._Zij[:,:self.block_index+1]
        assert Zij.shape == (self.num_channels, self.block_index+1)

        above_abs = block_loudness >= -70
        self._blocks_above_abs_thresh[self.block_index] = above_abs
        rel_changed = self._calc_relative_threshold()

        self._process_quarter_block(samples)
        # self._process_quarter_block_filtered(samples)

        self._calc_integrated(rel_changed)
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
        # assert quarter_blk_samples.shape[1] == self.pad_size, f'{self.pad_size=} {quarter_blk_samples.shape}='
        sq_sum: Float1dArray = np.sum(
            np.square(quarter_blk_samples), axis=1
        )
        tp = 1 / (self.pad_size)
        weighted_sum = np.sum((tp * sq_sum) * self.weights)
        self._quarter_block_weighted_sums[self.block_index] = weighted_sum

    def _process_quarter_block_filtered(self, samples: Float2dArray):
        # Calculated the weighted squared-sums of the last 100ms segment within
        # this 400ms block. (for use in momentary calculation)
        # With an overlap of 75% in a 400ms block, the only
        # "new" samples will be the last 100ms
        quarter_blk_samples = samples[:,:-self.pad_size].copy()
        filtered = self._decimate_hp(quarter_blk_samples)
        tp = 1 / (self.pad_size)
        weighted_sum = np.sum(tp * filtered)
        self._quarter_block_weighted_sums[self.block_index] = weighted_sum

    def _decimate_hp(self, samples: Float2dArray) -> Float1dArray:
        # tg = 1 / samples.shape[0]
        sq_channel_sum = np.sum(np.square(samples) * self.weights[:,np.newaxis], axis=0)
        dec_fact = self.sample_rate // 320
        dec_shape = (dec_fact, samples.shape[1] // dec_fact)

        decimated = np.reshape(sq_channel_sum, dec_shape).mean(axis=0)
        assert decimated.size == dec_shape[-1]
        r = self._short_term_filter(decimated)
        assert r.size == decimated.size
        return r



        # decimated = np.reshape(samples, (self.num_channels, 150, 128)).mean(axis=1)
        # samples = self._short_term_filter(decimated)
        # sq_sum: FloatArray = np.sum(np.square(samples), axis=1)
        # return np.sum(sq_sum * self.weights)

    # def process_momentary(self, samples: Float2dArray) -> np.floating:
    #     assert samples.shape == (self.num_channels, self.gate_size)
    #     decimated = np.reshape(samples, (self.num_channels, 150, 128)).mean(axis=1)
    #     samples = self._momentary_filter(decimated)
    #     tg = 1 / samples.shape[-1]

    #     # Squared sum of the weighted channel samples
    #     sq_sum = np.sum((samples * samples) * self.weights[:,np.newaxis], axis=0)
    #     return lk_log10(np.mean(tg * sq_sum))

    #     # # Decimate from 48000 to 320 using arithmetic mean
    #     # # (400ms == 19200 samples at 48000, 128 samples at 320)
    #     # decimated = np.reshape(sq_sum, (150, 128)).mean(axis=0)

    #     # filtered = self._momentary_filter(decimated)

    #     # Not sure if this is right or not
    #     # return lk_log10(filtered.sum())


class TruePeakProcessor(BaseProcessor):
    """Process audio samples to extract their :term:`True Peak` values
    """
    max_peak: Floating
    """Maximum :term:`True Peak` value detected"""

    current_peaks: Float1dArray
    """:term:`True Peak` values per channel from the last processing period"""

    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels=num_channels)
        self.resample_filt = TruePeakFilter(num_channels=num_channels)
        self.max_peak = SILENCE_DB
        self.current_peaks: Float1dArray = np.zeros(self.num_channels, dtype=np.float64)
        self.current_peaks[:] = SILENCE_DB

    def __call__(self, samples: Float2dArray) -> None:
        self.process(samples)

    def reset(self) -> None:
        self.max_peak = SILENCE_DB
        self.current_peaks[:] = SILENCE_DB

    def process(self, samples: Float2dArray):
        tp_vals = self.resample_filt(samples)
        tp_vals = lk_log10(np.abs(tp_vals), offset=0, base=20)
        cur_peaks = tp_vals.max(axis=1)
        max_peak = cur_peaks.max()
        if max_peak > self.max_peak:
            self.max_peak = max_peak
        self.current_peaks[:] = cur_peaks
