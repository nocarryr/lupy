from __future__ import annotations

from typing import Callable, Iterable
import time
import pytest
import numpy as np

from lupy import Meter
from lupy.types import FloatArray

from conftest import gen_1k_sine, IS_CI


class DurationContext:
    """Helper to check processing time
    """
    source_duration: float
    """Length of the input data in seconds"""

    min_source_duration = 15
    """Minimum source length (s) to consider when checking durations"""

    max_duration_ratio = 35
    """Highest ratio (percent) of :attr:`duration` to :attr:`source_duration`"""
    def __init__(self, source_duration: float, max_duration_ratio: int = 35) -> None:
        self.source_duration = source_duration
        self.max_duration_ratio = max_duration_ratio
        self.start: float|None = None
        self.end: float|None = None

    @property
    def duration(self) -> float:
        """Time elapsed between context entry and exit
        """
        if self.start is None or self.end is None:
            raise ValueError('No start or end timestamp')
        return self.end - self.start

    def check_duration(self) -> None:
        """Check whether :attr:`duration` exceeds the :attr:`max_duration_ratio`
        compared to :attr:`source_duration`
        """
        if self.source_duration <= self.min_source_duration:
            return
        ratio = (self.duration / self.source_duration) * 100
        print(f'{self.source_duration=}, {self.duration=}, {ratio=}')
        if ratio > self.max_duration_ratio:
            raise Exception(f'Duration exceeded threshold: ' +
                            f'{self.source_duration=}, {self.duration=}, {ratio=}')

    def __enter__(self):
        assert self.start is None
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.end = time.monotonic()


@pytest.fixture(params=[True, False])
def is_silent(request) -> bool:
    return request.param


def build_samples(
    num_samples: int,
    num_channels: int,
    sample_rate: int,
    sine_channels: Iterable[int]|None,
    sine_amp: float = 1,
) -> FloatArray:
    samples = np.zeros((num_channels, num_samples), dtype=np.float64)
    if sine_channels is not None:
        sig = gen_1k_sine(num_samples, sample_rate, sine_amp)
        samples[np.array(sine_channels),...] = sig
    return samples


def test_integrated_lkfs(sample_rate, block_size, all_channels, is_silent):
    num_channels, sine_channel = all_channels
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    N, Fs = meter.sampler.total_samples, int(meter.sample_rate)
    num_blocks, gate_size = meter.sampler.num_blocks, meter.sampler.gate_size

    src_data = build_samples(N, num_channels, Fs, [sine_channel], 1)
    assert src_data.shape == (num_channels, N)

    if is_silent:
        meter.write_all(src_data)
        assert meter.integrated_lkfs > -120
        meter.reset()
        src_data[...] = 0

    meter.write_all(src_data)

    print(f'{N=}, {N / gate_size}, {len(meter.processor)=}')

    if is_silent:
        assert meter.integrated_lkfs <= -120
    elif sine_channel < 3:
        assert round(meter.integrated_lkfs, 2) == -3.01
        # assert -3.02 <= lkfs <= -3.00
    else:
        assert -1.53 <= round(meter.integrated_lkfs, 2) <= -1.51
        # assert -1.53 <= lkfs <= -1.51


# https://tech.ebu.ch/docs/tech/tech3341.pdf Section 2.9
# Stereo 1k (997 Hz) sine at -18 dBFS should read -18 LUFS
def test_integrated_lkfs_neg18(sample_rate, block_size):
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    N, Fs = meter.sampler.total_samples, int(meter.sample_rate)
    num_blocks, gate_size = meter.sampler.num_blocks, meter.sampler.gate_size

    sine_channels = [0, 1]
    amp = 10 ** (-18/20)
    src_data = build_samples(N, num_channels, Fs, sine_channels, amp)
    meter.write_all(src_data)

    print(f'{N=}, {N / gate_size}, {len(meter.processor)=}')

    assert round(meter.integrated_lkfs, 2) == -18


def test_compliance_cases(sample_rate, compliance_case):
    block_size = 128
    num_channels = 5
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    print('generating samples...')
    src_data = compliance_case.generate_samples(int(meter.sample_rate), block_size=block_size)
    N = src_data.shape[1]
    assert N % block_size == 0
    src_duration = N / meter.sample_rate

    print(f'processing {N} samples...')

    # for block_index in iter_process(sampler, processor, src_data):
    #     # print(f'{block_index=}, {processor.integrated_lkfs=}, {sampler.samples_available=}')
    #     pass

    duration_ratio = DurationContext.max_duration_ratio
    if IS_CI and sample_rate > 48000:
        duration_ratio *= 2
    with DurationContext(src_duration, max_duration_ratio=duration_ratio) as dur_ctx:
        meter.write_all(src_data)

    print(f'{meter.t[-1]=}')
    integrated = meter.integrated_lkfs
    momentary, short_term = meter.momentary_lkfs[-1], meter.short_term_lkfs[-1]
    print(f'{integrated=}, {momentary=}, {short_term=}')
    print(f'{meter.processor._rel_threshold=}')

    integrated_target = compliance_case.result.integrated
    momentary_target = compliance_case.result.momentary
    short_term_target = compliance_case.result.short_term
    lra_target = compliance_case.result.lra
    tp_target = compliance_case.result.true_peak

    # Allow for extra tolerance at 44.1k since these cases were likely designed
    # for 48k (although it's not explicitly stated)
    tol_padding_44100 = 0.1

    if momentary_target is not None:
        lufs, lu, tol = momentary_target
        if sample_rate == 44100:
            tol += tol_padding_44100
        assert lufs - tol <= momentary <= lufs + tol
    if short_term_target is not None:
        lufs, lu, tol = short_term_target
        if sample_rate == 44100:
            tol += tol_padding_44100
        assert lufs - tol <= short_term <= lufs + tol
    if integrated_target is not None:
        lufs, lu, tol = integrated_target
        if sample_rate == 44100:
            tol += tol_padding_44100
        assert lufs - tol <= integrated <= lufs + tol
    if lra_target is not None:
        lra_lu, tol = lra_target
        assert lra_lu - tol <= meter.lra <= lra_lu + tol
    if tp_target is not None:
        tp, neg_tol, pos_tol = tp_target
        tp_min, tp_max = tp - neg_tol, tp + pos_tol
        assert tp_min <= meter.true_peak_max <= tp_max

    dur_ctx.check_duration()


@pytest.mark.benchmark(group='meter')
def test_meter_benchmark(sample_rate, random_samples, benchmark):
    block_size = 1024
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    N = sample_rate * 1
    if N % block_size != 0:
        N += block_size - (N % block_size)
    assert N % block_size == 0
    src_data = random_samples(num_channels, N)

    def bench():
        meter.write_all(src_data)
        meter.reset()
    benchmark(bench)
