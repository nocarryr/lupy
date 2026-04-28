from __future__ import annotations

import numpy as np
from scipy import signal
import pytest

from lupy.filters import HS_COEFF, HP_COEFF, Coeff, Filter, FilterGroup, TruePeakFilter



@pytest.fixture(params=[48000])
def bench_sample_rate(request) -> int:
    return request.param


@pytest.mark.parametrize('coeff', [HS_COEFF, HP_COEFF])
def test_filter_requantize(coeff, sample_rate):
    orig_sample_rate = coeff.sample_rate
    quantized = coeff.as_sample_rate(sample_rate)

    re_quantized = quantized.as_sample_rate(orig_sample_rate)
    assert np.allclose(re_quantized.b, coeff.b)
    assert np.allclose(re_quantized.a, coeff.a)

    w0, h0 = signal.freqz(coeff.b, coeff.a)
    w1, h1 = signal.freqz(re_quantized.b, re_quantized.a)
    assert isinstance(w0, np.ndarray)
    assert isinstance(w1, np.ndarray)
    assert isinstance(h0, np.ndarray)
    assert isinstance(h1, np.ndarray)
    assert np.allclose(w0, w1)
    assert np.allclose(h0, h1)


def test_coeff_combine_different_sample_rates() -> None:
    """Coeff.combine raises ValueError when sample rates differ."""
    coeff_44k = HS_COEFF.as_sample_rate(44100)
    with pytest.raises(ValueError, match="different sample rates"):
        HS_COEFF.combine(coeff_44k)


def test_filter_single_channel_1d_input() -> None:
    """Filter accepts a 1-D array for single-channel input, reshaping to (1, N)."""
    filt = Filter(coeff=HS_COEFF, num_channels=1)
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(480).astype(np.float64)
    result = filt(samples)
    assert result.ndim == 2
    assert result.shape[0] == 1
    assert result.shape[1] == 480


def test_filter_group_single_channel_1d_input() -> None:
    """FilterGroup accepts a 1-D array for single-channel input."""
    fg = FilterGroup(HS_COEFF, HP_COEFF, num_channels=1)
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(480).astype(np.float64)
    result = fg(samples)
    assert result.ndim == 2
    assert result.shape[0] == 1
    assert result.shape[1] == 480


@pytest.mark.benchmark(group='filter')
def test_filter_benchmark(benchmark, random_samples, num_channels, bench_sample_rate):
    sample_rate = bench_sample_rate
    block_size = sample_rate // 100
    assert sample_rate % block_size == 0

    samples = random_samples(num_channels, block_size)
    assert samples.shape == (num_channels, block_size)
    coeff = [HS_COEFF, HP_COEFF]
    if sample_rate != 48000:
        coeff = [c.as_sample_rate(sample_rate) for c in coeff]
    fg = FilterGroup(*coeff, num_channels=num_channels)

    filtered = np.zeros(samples.shape, dtype=np.float64)

    def bench():
        filtered[...] = fg(samples)

    benchmark(bench)

@pytest.mark.benchmark(group='truepeak_filter')
def test_truepeak_filter_benchmark(benchmark, random_samples, num_channels, bench_sample_rate):
    sample_rate = bench_sample_rate
    up_sample = 4 if sample_rate < 88200 else 2
    block_size = sample_rate // 100
    assert sample_rate % block_size == 0

    samples = random_samples(num_channels, block_size)
    assert samples.shape == (num_channels, block_size)
    tp_filter = TruePeakFilter(num_channels=num_channels, upsample_factor=up_sample)
    num_output_samples = samples.shape[1] * tp_filter.upsample_factor

    filtered = np.zeros((samples.shape[0], num_output_samples), dtype=np.float64)

    def bench():
        filtered[...] = tp_filter(samples)

    benchmark(bench)
