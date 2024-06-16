from __future__ import annotations

from .sampling import Sampler
from .processing import BlockProcessor
from .types import *

__all__ = ('Meter',)

class Meter:
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
        return self.sampler.can_write()

    def can_process(self) -> bool:
        return self.sampler.can_read()

    def write(
        self,
        samples: FloatArray,
        process: bool = True,
        process_all: bool = True
    ) -> None:
        self.sampler.write(samples)
        if process and self.can_process():
            self.process(process_all=process_all)

    def process(self, process_all: bool = True) -> None:
        def _do_process():
            samples = self.sampler.read()
            self.processor.process_block(samples)
        if process_all:
            while self.can_process():
                _do_process()
        else:
            _do_process()


    @property
    def block_data(self):
        return self.processor.block_data

    @property
    def momentary_lkfs(self) -> Float1dArray:
        """Short-term loudness for each 100ms block, averaged over 400ms
        (not gated)
        """
        return self.processor.momentary_lkfs

    @property
    def short_term_lkfs(self) -> Float1dArray:
        """Short-term loudness for each 100ms block, averaged over 3 seconds
        (not gated)
        """
        return self.processor.short_term_lkfs

    @property
    def t(self) -> Float1dArray:
        return self.processor.t
