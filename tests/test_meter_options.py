from __future__ import annotations
from typing import Iterable

import pytest
import numpy as np

from lupy import Meter
from lupy.types import NumChannelsT
from lupy.typeutils import is_array_of_shape

from conftest import gen_1k_sine



def build_samples(
    num_samples: int,
    sample_rate: int,
    num_channels: NumChannelsT = 2,
    sine_channels: Iterable[int]|None = (0, 1),
    sine_amp: float = 10 ** (-18/20),
) -> np.ndarray[tuple[NumChannelsT, int], np.dtype[np.float64]]:
    samples = np.zeros((num_channels, num_samples), dtype=np.float64)
    assert is_array_of_shape(samples, (num_channels, num_samples))
    if sine_channels is not None:
        sig = gen_1k_sine(num_samples, sample_rate, sine_amp)
        samples[np.array(sine_channels),...] = sig
    return samples



def test_momentary_disabled():
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(
        block_size=block_size,
        num_channels=num_channels,
        sample_rate=sample_rate,
        momentary_enabled=False,
    )
    assert not meter.momentary_enabled
    assert meter.short_term_enabled
    assert meter.lra_enabled

    N = meter.sampler.total_samples
    src_data = build_samples(N, sample_rate, num_channels)
    meter.write_all(src_data)

    assert round(meter.integrated_lkfs, 2) == -18
    assert np.array_equal(meter.momentary_lkfs, np.zeros_like(meter.momentary_lkfs))
    assert not np.array_equal(meter.short_term_lkfs, np.zeros_like(meter.short_term_lkfs))
    assert meter.lra != 0.0
    assert meter.true_peak_max != -np.inf
    tp_array = meter.true_peak_array['tp']
    assert not np.array_equal(tp_array, np.full_like(tp_array, -np.inf))
    current_measurement = meter.current_measurement
    assert current_measurement.momentary == 0.0


def test_short_term_disabled_and_lra_enabled_raises():
    sample_rate = 48000
    block_size = 128
    num_channels = 2

    # LRA requires short-term to be enabled which raises an error.
    with pytest.raises(ValueError) as excinfo:
        meter = Meter(
            block_size=block_size,
            num_channels=num_channels,
            sample_rate=sample_rate,
            short_term_enabled=False,
            lra_enabled=True,
        )
    assert 'lra' in str(excinfo.value).lower() and 'short-term' in str(excinfo.value).lower()


def test_short_term_disabled():
    sample_rate = 48000
    block_size = 128
    num_channels = 2

    meter = Meter(
        block_size=block_size,
        num_channels=num_channels,
        sample_rate=sample_rate,
        short_term_enabled=False,
        lra_enabled=False,
    )
    assert not meter.short_term_enabled
    assert not meter.lra_enabled
    assert meter.momentary_enabled

    N = meter.sampler.total_samples
    src_data = build_samples(N, sample_rate, num_channels)
    meter.write_all(src_data)

    assert round(meter.integrated_lkfs, 2) == -18
    assert not np.array_equal(meter.momentary_lkfs, np.zeros_like(meter.momentary_lkfs))
    assert np.array_equal(meter.short_term_lkfs, np.zeros_like(meter.short_term_lkfs))
    assert meter.lra == 0.0
    assert meter.true_peak_max != -np.inf
    tp_array = meter.true_peak_array['tp']
    assert not np.array_equal(tp_array, np.full_like(tp_array, -np.inf))
    current_measurement = meter.current_measurement
    assert current_measurement.short_term == 0.0


def test_lra_disabled():
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(
        block_size=block_size,
        num_channels=num_channels,
        sample_rate=sample_rate,
        lra_enabled=False,
    )
    assert not meter.lra_enabled
    assert meter.momentary_enabled
    assert meter.short_term_enabled

    N = meter.sampler.total_samples
    src_data = build_samples(N, sample_rate, num_channels)
    meter.write_all(src_data)

    assert round(meter.integrated_lkfs, 2) == -18
    assert not np.array_equal(meter.momentary_lkfs, np.zeros_like(meter.momentary_lkfs))
    assert not np.array_equal(meter.short_term_lkfs, np.zeros_like(meter.short_term_lkfs))
    assert meter.lra == 0.0
    assert meter.true_peak_max != -np.inf
    tp_array = meter.true_peak_array['tp']
    assert not np.array_equal(tp_array, np.full_like(tp_array, -np.inf))
    current_measurement = meter.current_measurement
    assert current_measurement.lra == 0.0


def test_true_peak_disabled():
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(
        block_size=block_size,
        num_channels=num_channels,
        sample_rate=sample_rate,
        true_peak_enabled=False,
    )
    assert not meter.true_peak_enabled

    N = meter.sampler.total_samples
    src_data = build_samples(N, sample_rate, num_channels)
    meter.write_all(src_data)

    assert round(meter.integrated_lkfs, 2) == -18
    assert not np.array_equal(meter.momentary_lkfs, np.zeros_like(meter.momentary_lkfs))
    assert not np.array_equal(meter.short_term_lkfs, np.zeros_like(meter.short_term_lkfs))
    assert meter.lra != 0.0
    assert meter.true_peak_max == -np.inf
    tp_array = meter.true_peak_array['tp']
    assert np.array_equal(tp_array, np.full_like(tp_array, -np.inf))
    current_measurement = meter.current_measurement
    assert current_measurement.true_peak_max == -np.inf
    assert np.all(current_measurement.true_peak_current == -np.inf)


def test_set_paused_blocks_write_and_process():
    """When paused, can_write/can_process return False and writes are discarded."""
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    assert not meter.paused
    assert meter.can_write()

    meter.set_paused(True)
    assert meter.paused
    assert not meter.can_write()
    assert not meter.can_process()

    # Writing while paused is a no-op: samples_available stays at zero
    samples = build_samples(block_size, sample_rate, num_channels)
    meter.write(samples)
    assert meter.sampler.samples_available == 0


def test_set_paused_clears_samplers():
    """Pausing a meter that has buffered data clears the internal samplers."""
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    samples = build_samples(block_size, sample_rate, num_channels)
    meter.write(samples, process=False)
    assert meter.sampler.samples_available > 0

    meter.set_paused(True)
    assert meter.sampler.samples_available == 0


def test_set_paused_noop_when_same_state():
    """Calling set_paused with the current state is a no-op."""
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    samples = build_samples(block_size, sample_rate, num_channels)
    meter.write(samples, process=False)
    before = meter.sampler.samples_available

    # Already unpaused: set_paused(False) should not touch buffers
    meter.set_paused(False)
    assert meter.sampler.samples_available == before

    # Pausing then calling set_paused(True) again is a no-op
    meter.set_paused(True)
    meter.set_paused(True)
    assert meter.paused


def test_resume_after_pause():
    """After unpausing, writes and processing resume normally."""
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    meter.set_paused(True)
    assert not meter.can_write()

    meter.set_paused(False)
    assert not meter.paused
    assert meter.can_write()

    samples = build_samples(block_size, sample_rate, num_channels)
    meter.write(samples, process=False)
    assert meter.sampler.samples_available > 0


def test_reset_resets_true_peak_processor():
    """reset() resets the true_peak_processor when true_peak_enabled is True."""
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(
        block_size=block_size, num_channels=num_channels,
        sample_rate=sample_rate, true_peak_enabled=True,
    )
    N = meter.sampler.total_samples
    src_data = build_samples(N, sample_rate, num_channels)
    meter.write_all(src_data)
    assert meter.true_peak_max != -np.inf

    meter.reset()
    assert meter.true_peak_max == -np.inf


def test_process_single_block():
    """process(process_all=False) processes exactly one gating block."""
    sample_rate = 48000
    block_size = 128
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    N = meter.sampler.total_samples
    src_data = build_samples(N, sample_rate, num_channels)
    num_blocks = N // block_size
    for i in range(num_blocks):
        meter.write(src_data[:, i * block_size:(i + 1) * block_size], process=False)

    assert meter.sampler.can_read()
    meter.process(process_all=False)
    assert len(meter.block_data) == 1


def test_write_all_truncates_non_multiple():
    """write_all silently truncates input not divisible by block_size."""
    sample_rate = 48000
    block_size = 512
    num_channels = 2
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    extra = 100
    N = meter.sampler.total_samples + extra
    assert N % block_size != 0

    src_data = build_samples(N, sample_rate, num_channels)
    # Should not raise; extra samples are silently dropped
    meter.write_all(src_data)
