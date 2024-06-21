from __future__ import annotations

from .sampling import Sampler
from .processing import BlockProcessor
from .types import *

__all__ = ('Meter',)

class Meter:
    """
    """

    block_size: int
    """The number of input samples per call to :meth:`write`"""

    num_channels: int
    """Number of audio channels"""

    sampler: Sampler
    """The :class:`Sampler` instance to buffer input data"""

    processor: BlockProcessor
    """The :class:`BlockProcessor` to perform the calulations"""

    sample_rate: int = 48000
    def __init__(self, block_size: int, num_channels: int) -> None:
        self.block_size = block_size
        self.num_channels = num_channels
        self.sampler = Sampler(
            block_size=block_size,
            num_channels=num_channels,
        )
        self.processor = BlockProcessor(
            num_channels=num_channels,
            gate_size=self.sampler.gate_size,
        )

    def can_write(self) -> bool:
        """Whether there is enough room on the internal buffer for at least
        one call to :meth:`write`
        """
        return self.sampler.can_write()

    def can_process(self) -> bool:
        """Whether there are enough samples in the internal buffer for at least
        one call to :meth:`process`
        """
        return self.sampler.can_read()

    def write(
        self,
        samples: FloatArray,
        process: bool = True,
        process_all: bool = True
    ) -> None:
        """Store input data into the internal buffer

        The input data must be of shape ``(num_channels, block_size)``
        """
        self.sampler.write(samples)
        if process and self.can_process():
            self.process(process_all=process_all)

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
        def _do_process():
            samples = self.sampler.read()
            self.processor.process_block(samples)
        if process_all:
            while self.can_process():
                _do_process()
        else:
            _do_process()

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
