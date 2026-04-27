from __future__ import annotations
from typing import Iterator
import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from os import PathLike
from pathlib import Path
import time
import argparse
from typing import Generic, Union, cast
from fractions import Fraction

import numpy as np
from scipy.io import wavfile

from .sampling import Sampler, TruePeakSampler
from .processing import BlockProcessor, TruePeakProcessor
from .io import MeterData, MeterMeta
from .arraytypes import MeterArray, TruePeakArray
from .types import *
from .types import NumChannelsOptions
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

        If *short_term_enabled* is ``False``, *lra_enabled* must also be ``False``.
        This is because :term:`Loudness Range` calculation depends on
        :term:`Short-Term Loudness` values.

    Raises:
        ValueError: If *short_term_enabled* is ``False`` and *lra_enabled* is ``True``

    """

    block_size: int
    """The number of input samples per call to :meth:`write`"""

    num_channels: NumChannelsT
    """Number of audio channels"""

    sampler: Sampler[NumChannelsT]
    """The :class:`~.sampling.Sampler` instance to buffer input data"""

    true_peak_sampler: TruePeakSampler[NumChannelsT]
    """Sample buffer to hold un-filtered samples for :attr:`true_peak_processor`"""

    processor: BlockProcessor[NumChannelsT]
    """The :class:`~.processing.BlockProcessor` to perform the calulations"""

    true_peak_processor: TruePeakProcessor[NumChannelsT]
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
        for num_written, num_remaining in self._write_all(samples):
            print(f'Written {num_written} samples, {num_remaining} remaining')

    def write_all_iter(self, samples: Any2dArray[FloatDtypeT]) -> Iterator[tuple[int, int]]:
        yield from self._write_all(samples)

    def _write_all(self, samples: Any2dArray[FloatDtypeT]) -> Iterator[tuple[int, int]]:
        num_samples = samples.shape[1]
        assert samples.shape[0] == self.num_channels, f'Expected {self.num_channels} channels, got {samples.shape[0]}'
        num_blocks = num_samples // self.block_size
        if num_samples % self.block_size != 0:
            num_samples = num_blocks * self.block_size
            samples = ensure_2d_array(samples[:,:num_samples])
        block_samples = np.reshape(samples, (self.num_channels, num_blocks, self.block_size))
        num_written = 0
        num_remaining = num_samples

        write_index = 0
        while write_index < num_blocks:
            while self.can_write() and write_index < num_blocks:
                _block_samples = ensure_2d_array(block_samples[:,write_index,:])
                _block_samples = cast('Float2dArray|Float2dArray32', _block_samples)
                self.write(_block_samples)
                write_index += 1
                num_written += self.block_size
                num_remaining -= self.block_size
                yield num_written, num_remaining


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

    def _get_object_data(self) -> MeterData:
        meta = MeterMeta(
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            block_size=self.block_size,
            block_processor=self.processor.get_object_data(),
            true_peak_processor=self.true_peak_processor.get_object_data(),
        )
        return MeterData(
            metadata=meta,
            arrays={},
        )

    @classmethod
    def _from_object_data(cls, obj_data: MeterData) -> Self:
        """Create a new :class:`Meter` from an :class:`~.io.ObjectData` instance"""
        meta = obj_data.metadata
        meter = cls(
            block_size=meta['block_size'],
            num_channels=meta['num_channels'],
            sample_rate=meta['sample_rate'],
        )
        meter.processor = BlockProcessor.from_object_data(meta['block_processor'])
        meter.true_peak_processor = TruePeakProcessor.from_object_data(meta['true_peak_processor'])
        return meter

    def save(self, filename: Path|str) -> None:
        """Save the current state of the meter to a file

        Arguments:
            filename: The path to the file to save the meter state to
        """
        obj_data = self._get_object_data()
        obj_data.save(filename)

    @classmethod
    def load(cls, filename: Path|str) -> Self:
        """Load a :class:`Meter` from a file

        Arguments:
            filename: The path to the file to load the meter state from
        """
        obj_data = MeterData.load(filename)
        return cls._from_object_data(obj_data)

    @staticmethod
    def from_wavfile(filename: PathLike, block_size: int = 1024, mmap: bool = False) -> Meter:
        """Create a new :class:`Meter` from a WAV file

        Arguments:
            filename: The path to the WAV file to read
        """
        def to_float32(data: np.ndarray) -> np.ndarray:
            if data.dtype == np.float32:
                return data
            if data.dtype == np.int16:
                data = data.astype(np.float64) / 32768
            elif data.dtype == np.int32:
                data = data.astype(np.float64) / 2147483648
            else:
                raise ValueError(f'Unsupported data type: {data.dtype}')
            return np.asarray(data, dtype=np.float32)

        start_ts = time.monotonic()
        # mmap = True
        sample_rate, data = wavfile.read(filename, mmap=mmap)

        num_samples = data.shape[0]
        num_channels = data.shape[1] if len(data.shape) > 1 else 1
        # assert isinstance(num_channels, int)
        # assert num_channels in NumChannels, f'Unsupported number of channels: {num_channels}'
        assert num_channels in NumChannelsOptions, f'Unsupported number of channels: {num_channels}'

        if sample_rate == 44100:
            block_size = 441

        data = data.T
        # block_size = 1024
        meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)
        data = to_float32(data)
        print(f'{data.flags=}')
        # meter.write_all(data)
        progress_ticks = {i * 10 for i in range(11)}

        for num_written, num_remaining in meter.write_all_iter(data):
            # print(f'{num_written=}, {num_remaining=}')
            elapsed = time.monotonic() - start_ts
            time_processed = num_written / sample_rate
            progress = int(round(num_written / num_samples * 100))
            if len(progress_ticks) and progress >= min(progress_ticks):
                print(f'{progress:3d}%\t(elapsed: {elapsed:6.1f}s)\t(time processed: {time_processed:6.1f}s)')
                progress_ticks.remove(progress)
        return meter

        # if not mmap:
        #     data = to_float32(data)
        # start_ix = 0
        # end_ix = start_ix + block_size
        # progress = 0
        # progress_ticks = {i * 10 for i in range(11)}
        # start_ts = time.monotonic()
        # while end_ix < num_samples:
        #     _data = data[:,start_ix:end_ix]
        #     if mmap:
        #         _data = to_float32(_data)
        #     meter.write(_data)
        #     elapsed = time.monotonic() - start_ts
        #     start_ix = end_ix
        #     end_ix += block_size
        #     progress = int(round(start_ix / num_samples * 100))
        #     time_processed = start_ix / sample_rate
        #     if len(progress_ticks) and progress >= min(progress_ticks):
        #         print(f'{progress:3d}%\t(elapsed: {elapsed:6.1f}s)\t(time processed: {time_processed:6.1f}s)')
        #         progress_ticks.remove(progress)
        # return meter

    # def __repr__(self) -> str:
    #     return f'<{type(self).__name__} {self.block_size=}, {self.num_channels=}>'

    def __str__(self) -> str:
        def rounded(x: Floating|float) -> str:
            return f'{x: 5.1f}'

        return f'Integrated:        {rounded(self.integrated_lkfs)} LUFS\n' + \
               f'dBTP:              {rounded(self.true_peak_max)} dBTP\n' + \
               f'Peak-to-Loudness:  {rounded(self.true_peak_max - self.integrated_lkfs)} dB\n' + \
               f'LRA:               {rounded(self.lra)} LU\n' + \
               f'Short-Term Max:    {rounded(self.short_term_lkfs.max())} LUFS\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('filename', type=Path)
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--mmap', action='store_true')
    args = p.parse_args()
    meter = Meter.from_wavfile(args.filename, block_size=args.block_size, mmap=args.mmap)
    print(str(meter))
