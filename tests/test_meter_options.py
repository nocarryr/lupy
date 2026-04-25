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
