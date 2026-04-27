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


def make_meter(
    block_size: int = 128,
    num_channels: NumChannelsT = 2,
    sample_rate: int = 48000,
    true_peak_enabled: bool = True,
) -> Meter[NumChannelsT]:
    return Meter(
        block_size=block_size,
        num_channels=num_channels,
        sample_rate=sample_rate,
        true_peak_enabled=true_peak_enabled,
    )


def test_set_paused_blocks_writes():
    """Writes are discarded while paused; can_write and can_process return False."""
    meter = make_meter()
    meter.set_paused(True)
    assert meter.paused

    assert not meter.can_write()
    assert not meter.can_process()

    block = build_samples(meter.block_size, 48000)
    meter.write(block)  # must not raise; data is silently discarded
    assert meter.sampler.samples_available == 0


def test_set_paused_clears_buffer():
    """Buffered samples are cleared when the meter is paused."""
    meter = make_meter()
    # Write enough blocks to have some buffered data
    block = build_samples(meter.block_size, meter.sample_rate)
    meter.write(block, process=False)
    assert meter.sampler.samples_available > 0

    meter.set_paused(True)
    assert meter.sampler.samples_available == 0


def test_set_paused_noop_if_same_state():
    """set_paused is idempotent: calling with the current state does nothing."""
    meter = make_meter()
    block = build_samples(meter.block_size, meter.sample_rate)
    meter.write(block, process=False)
    samples_before = meter.sampler.samples_available

    meter.set_paused(False)  # already not paused — must not clear the buffer
    assert meter.sampler.samples_available == samples_before


def test_set_paused_resume():
    """Unpausing allows writes to proceed again."""
    meter = make_meter()
    meter.set_paused(True)
    meter.set_paused(False)
    assert not meter.paused
    assert meter.can_write()


def test_reset_with_true_peak_enabled():
    """reset() reinitialises the true-peak processor when true_peak_enabled."""
    meter = make_meter(true_peak_enabled=True)
    N = meter.sampler.total_samples
    src_data = build_samples(N, meter.sample_rate, meter.num_channels)
    meter.write_all(src_data)

    assert meter.true_peak_max != -np.inf

    meter.reset()
    assert meter.true_peak_max == -np.inf
    assert meter.true_peak_processor.tp_array.size == 0


def test_process_single_block():
    """process(process_all=False) processes exactly one gating block."""
    meter = make_meter()

    gate_size = meter.sampler.gate_size
    block = build_samples(meter.block_size, meter.sample_rate, meter.num_channels)
    blocks_needed = gate_size // meter.block_size + 1
    for _ in range(blocks_needed):
        meter.write(block, process=False)

    assert meter.can_process()
    before = len(meter.block_data)
    meter.process(process_all=False)
    assert len(meter.block_data) == before + 1




def test_current_measurement_empty():
    """current_measurement returns silence defaults before any block is processed."""
    from lupy.processing import SILENCE_DB
    meter = make_meter()

    m = meter.current_measurement
    assert m.momentary == SILENCE_DB
    assert m.short_term == SILENCE_DB
    assert m.time == 0


def test_write_all_truncates_non_multiple():
    """write_all discards trailing samples that don't fill a complete block.

    With an input length of ``3 * block_size + extra`` (well below gate_size),
    the extra samples must be discarded and no gating blocks should be processed.
    """
    meter = make_meter()

    extra = 100  # not a multiple of block_size
    num_full_blocks = 3
    num_samples = num_full_blocks * meter.block_size + extra
    src_data = build_samples(num_samples, meter.sample_rate, meter.num_channels)

    meter.write_all(src_data)

    # The extra samples must have been discarded; only complete blocks are stored
    assert meter.sampler.samples_available == num_full_blocks * meter.block_size
    # Not enough data for even one gating block, so no blocks were processed
    assert len(meter.block_data) == 0
