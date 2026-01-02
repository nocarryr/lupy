from __future__ import annotations
from typing import TypeVar, cast

import numpy as np

from .sampling import Sampler, TruePeakSampler
from .processing import BlockProcessor, TruePeakProcessor
from .types import *
from .typeutils import is_2d_array, ensure_2d_array

__all__ = ('Meter',)


FloatDtypeT = TypeVar('FloatDtypeT', np.dtype[np.float32], np.dtype[np.float64], covariant=True)

class Meter:
    """

    Arguments:
        block_size: Number of input samples per call to :meth:`write`
        num_channels: Number of audio channels
        sampler_class: The class to use for the :attr:`sampler`
        sample_rate: The sample rate of the audio data
    """

    block_size: int
    """The number of input samples per call to :meth:`write`"""

    num_channels: int
    """Number of audio channels"""

    sampler: Sampler
    """The :class:`~.sampling.Sampler` instance to buffer input data"""

    true_peak_sampler: TruePeakSampler
    """Sample buffer to hold un-filtered samples for :attr:`true_peak_processor`"""

    processor: BlockProcessor
    """The :class:`~.processing.BlockProcessor` to perform the calulations"""

    true_peak_processor: TruePeakProcessor
    """The :class:`~.processing.TruePeakProcessor`"""

    sample_rate: int
    """The sample rate of the audio data"""
    def __init__(
        self,
        block_size: int,
        num_channels: int,
        sampler_class: type[Sampler] = Sampler,
        tp_sampler_class: type[TruePeakSampler] = TruePeakSampler,
        sample_rate: int = 48000
    ) -> None:
        self.block_size = block_size
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.sampler = sampler_class(
            block_size=block_size,
            num_channels=num_channels,
            sample_rate=sample_rate,
        )
        self.true_peak_sampler = tp_sampler_class(
            block_size=block_size,
            num_channels=num_channels,
            sample_rate=sample_rate,
        )
        self.processor = BlockProcessor(
            num_channels=num_channels,
            gate_size=self.sampler.gate_size,
            sample_rate=sample_rate,
        )
        self.true_peak_processor = TruePeakProcessor(
            num_channels=num_channels,
            sample_rate=sample_rate,
        )
        self._paused = False

    @property
    def paused(self) -> bool:
        """``True`` if processing is currently paused
        """
        return self._paused

    def can_write(self) -> bool:
        """Whether there is enough room on the internal buffer for at least
        one call to :meth:`write`
        """
        return self.sampler.can_write() and self.true_peak_sampler.can_write()

    def can_process(self) -> bool:
        """Whether there are enough samples in the internal buffer for at least
        one call to :meth:`process`
        """
        if self.paused:
            return False
        return self.sampler.can_read() or self.true_peak_sampler.can_read()

    def write(
        self,
        samples: Float2dArray|Float2dArray32,
        process: bool = True,
        process_all: bool = True
    ) -> None:
        """Store input data into the internal buffer

        The input data must be of shape ``(num_channels, block_size)``
        """
        if self.paused:
            return
        self.sampler.write(samples)
        self.true_peak_sampler.write(samples, apply_filter=False)
        if process and self.can_process():
            self.process(process_all=process_all)

    def write_all(self, samples: Any2dArray[FloatDtypeT]) -> None:
        """Write an arbitrary number of samples and process them

        If the number of samples is not a multiple of :attr:`block_size`, the
        samples will be truncated to the nearest multiple.
        """
        num_samples = samples.shape[1]
        assert samples.shape[0] == self.num_channels
        num_blocks = num_samples // self.block_size
        if num_samples % self.block_size != 0:
            num_samples = num_blocks * self.block_size
            samples = ensure_2d_array(samples[:,:num_samples])
        block_samples = np.reshape(samples, (self.num_channels, num_blocks, self.block_size))

        write_index = 0
        while write_index < num_blocks:
            while self.can_write() and write_index < num_blocks:
                _block_samples = ensure_2d_array(block_samples[:,write_index,:])
                _block_samples = cast(Float2dArray|Float2dArray32, _block_samples)
                self.write(_block_samples)
                write_index += 1

    def process(self, process_all: bool = True) -> None:
        """Process the samples for at least one :term:`gating block`

        Arguments:
            process_all: If ``True`` (the default), the :meth:`~.sampling.Sampler.read`
                method of the :attr:`sampler` will be called and the data passed to the
                :meth:`~.processing.BlockProcessor.process_block` method on the
                :attr:`processor` repeatedly until there are no
                :term:`gating block` samples available.
                Otherwise, only one call to each will be performed.

        """
        if process_all:
            while self.can_process():
                self._process()
        else:
            self._process()

    def _process(self) -> None:
        if self.sampler.can_read():
            samples = self.sampler.read()
            self.processor(samples)
        if self.true_peak_sampler.can_read():
            tp_samples = self.true_peak_sampler.read()
            assert is_2d_array(tp_samples)
            self.true_peak_processor(tp_samples)

    def reset(self) -> None:
        """Reset all values for :attr:`processor` and clear any buffered input
        samples
        """
        self.sampler.clear()
        self.true_peak_sampler.clear()
        self.processor.reset()
        self.true_peak_processor.reset()

    def set_paused(self, paused: bool) -> None:
        """Pause or unpause processing

        When paused, the current state of the :attr:`processor` is preserved
        and any input provided to the :meth:`write` method will be discarded.
        """
        if paused is self.paused:
            return
        self._paused = paused
        if paused:
            self.sampler.clear()
            self.true_peak_sampler.clear()

    @property
    def integrated_lkfs(self) -> Floating:
        """The current :term:`Integrated Loudness`"""
        return self.processor.integrated_lkfs

    @property
    def lra(self) -> float:
        """The current :term:`Loudness Range`"""
        return self.processor.lra

    @property
    def block_data(self) -> MeterArray:
        """A structured array of measurement values with
        dtype :obj:`~.types.MeterDtype`
        """
        return self.processor.block_data

    @property
    def momentary_lkfs(self) -> Float1dArray:
        """:term:`Momentary Loudness` for each 100ms block, averaged over 400ms
        (not gated)
        """
        return self.processor.momentary_lkfs

    @property
    def short_term_lkfs(self) -> Float1dArray:
        """:term:`Short-Term Loudness` for each 100ms block, averaged over 3 seconds
        (not gated)
        """
        return self.processor.short_term_lkfs

    @property
    def t(self) -> Float1dArray:
        """The measurement time for each element in :attr:`short_term_lkfs`
        and :attr:`momentary_lkfs`
        """
        return self.processor.t

    @property
    def true_peak_max(self) -> Floating:
        """Maximum :term:`True Peak` value detected"""
        return self.true_peak_processor.max_peak

    @property
    def true_peak_current(self) -> Float1dArray:
        """:term:`True Peak` values per channel from the last processing period"""
        return self.true_peak_processor.current_peaks
