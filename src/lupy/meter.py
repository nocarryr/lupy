from __future__ import annotations
from typing import Generic, Union, cast
from fractions import Fraction

import numpy as np

from .sampling import Sampler, TruePeakSampler
from .processing import BlockProcessor, TruePeakProcessor
from .arraytypes import MeterArray, TruePeakArray
from .types import *
from .typeutils import is_2d_array, ensure_2d_array

__all__ = ('Meter',)


FloatDtypeT = Union[np.dtype[np.float32], np.dtype[np.float64]]


class Meter(Generic[NumChannelsT]):
    """

    Arguments:
        block_size: Number of input samples per call to :meth:`write`
        num_channels: Number of audio channels
        sampler_class: The class to use for the :attr:`sampler`
        tp_sampler_class: The class to use for the :attr:`true_peak_sampler`
        sample_rate: The sample rate of the audio data
        true_peak_gate_duration: The processing duration for the
            :attr:`true_peak_processor` in seconds.
            See :attr:`TruePeakSampler.gate_duration <.sampling.TruePeakSampler.gate_duration>`
            for details.
        true_peak_enabled: Whether to enable :term:`True Peak` processing (default: ``True``)
        momentary_enabled: Whether to enable :term:`Momentary Loudness` processing (default: ``True``)
        short_term_enabled: Whether to enable :term:`Short-Term Loudness` processing (default: ``True``)
        lra_enabled: Whether to enable :term:`Loudness Range` processing (default: ``True``)

    .. important::

        If *short_term_enabled* is ``False``, *lra_enabled* must also be ``False``
        or a :class:`ValueError` will be raised.

        This is because :term:`Loudness Range` calculation depends on
        :term:`Short-Term Loudness` values.

    """

    block_size: int
    """The number of input samples per call to :meth:`write`"""

    num_channels: NumChannelsT
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
        num_channels: NumChannelsT,
        sampler_class: type[Sampler] = Sampler,
        tp_sampler_class: type[TruePeakSampler] = TruePeakSampler,
        sample_rate: int = 48000,
        true_peak_gate_duration: Fraction = Fraction(4, 10),
        true_peak_enabled: bool = True,
        momentary_enabled: bool = True,
        short_term_enabled: bool = True,
        lra_enabled: bool = True,
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
            gate_duration=true_peak_gate_duration,
        )
        self.processor = BlockProcessor(
            num_channels=num_channels,
            gate_size=self.sampler.gate_size,
            sample_rate=sample_rate,
            momentary_enabled=momentary_enabled,
            short_term_enabled=short_term_enabled,
            lra_enabled=lra_enabled,
        )
        self.true_peak_processor = TruePeakProcessor(
            num_channels=num_channels,
            gate_size=self.true_peak_sampler.gate_size,
            sample_rate=sample_rate,
        )
        self._paused = False
        self._true_peak_enabled = true_peak_enabled

    @property
    def true_peak_enabled(self) -> bool:
        """Whether :term:`True Peak` processing is enabled (read-only)
        """
        return self._true_peak_enabled

    @property
    def momentary_enabled(self) -> bool:
        """Whether :term:`Momentary Loudness` processing is enabled (read-only)
        """
        return self.processor.momentary_enabled

    @property
    def short_term_enabled(self) -> bool:
        """Whether :term:`Short-Term Loudness` processing is enabled (read-only)
        """
        return self.processor.short_term_enabled

    @property
    def lra_enabled(self) -> bool:
        """Whether :term:`Loudness Range` processing is enabled (read-only)
        """
        return self.processor.lra_enabled

    @property
    def paused(self) -> bool:
        """``True`` if processing is currently paused
        """
        return self._paused

    def can_write(self) -> bool:
        """Whether there is enough room on the internal buffer for at least
        one call to :meth:`write`
        """
        if self.paused:
            return False
        return self.sampler.can_write() and (
            not self.true_peak_enabled or self.true_peak_sampler.can_write()
        )

    def can_process(self) -> bool:
        """Whether there are enough samples in the internal buffer for at least
        one call to :meth:`process`
        """
        if self.paused:
            return False
        return (
            self.sampler.can_read() or
            (self.true_peak_enabled and self.true_peak_sampler.can_read())
        )

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
        if self.true_peak_enabled:
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
                _block_samples = cast('Float2dArray|Float2dArray32', _block_samples)
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
        if self.true_peak_enabled and self.true_peak_sampler.can_read():
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
        if self.true_peak_enabled:
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
            if self.true_peak_enabled:
                self.true_peak_sampler.clear()

    @property
    def integrated_lkfs(self) -> Floating:
        """The current :term:`Integrated Loudness`"""
        return self.processor.integrated_lkfs

    @property
    def lra(self) -> float:
        """The current :term:`Loudness Range`

        If :attr:`lra_enabled` is ``False``, this will always return ``0.0``.
        """
        return self.processor.lra

    @property
    def block_data(self) -> MeterArray:
        """A structured array of measurement values with
        dtype :obj:`~.arraytypes.MeterDtype`
        """
        return self.processor.block_data

    @property
    def momentary_lkfs(self) -> Float1dArray:
        """:term:`Momentary Loudness` for each 100ms block, averaged over 400ms
        (not gated)

        If :attr:`momentary_enabled` is ``False``, this will return an array of
        zeroes.
        """
        return self.processor.momentary_lkfs

    @property
    def short_term_lkfs(self) -> Float1dArray:
        """:term:`Short-Term Loudness` for each 100ms block, averaged over 3 seconds
        (not gated)

        If :attr:`short_term_enabled` is ``False``, this will return an array of
        zeroes.
        """
        return self.processor.short_term_lkfs

    @property
    def t(self) -> Float1dArray:
        """The measurement time for each element in :attr:`short_term_lkfs`
        and :attr:`momentary_lkfs`
        """
        return self.processor.t

    @property
    def true_peak_array(self) -> TruePeakArray[NumChannelsT]:
        """A structured array of :term:`True Peak` measurement values with
        dtype :obj:`~.arraytypes.TruePeakDtype`
        """
        return self.true_peak_processor.tp_array

    @property
    def true_peak_max(self) -> Floating:
        """Maximum :term:`True Peak` value detected

        If :attr:`true_peak_enabled` is ``False``, this will always return ``-inf``.
        """
        return self.true_peak_processor.max_peak

    @property
    def true_peak_current(self) -> np.ndarray[tuple[NumChannelsT], np.dtype[np.float64]]:
        """:term:`True Peak` values per channel from the last processing period

        If :attr:`true_peak_enabled` is ``False``, this will always return
        an array of ``-inf`` values.
        """
        return self.true_peak_processor.current_peaks
