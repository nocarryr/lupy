from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from lupy.filters import HP_COEFF, HS_COEFF, Filter, FilterGroup, TruePeakFilter
from lupy.signalutils.resample import (
    DEFAULT_INPUT_SAMPLES,
    ResamplePoly,
    calc_tp_fir_win,
)


@pytest.fixture(params=[48000])
def bench_sample_rate(request) -> int:
    return request.param


@pytest.mark.parametrize('coeff', [HS_COEFF, HP_COEFF])
def test_filter_requantize(coeff, sample_rate) -> None:
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
    """FilterGroup accepts a 1-D array for single-channel input and produces the same values as 2-D."""
    fg = FilterGroup(HS_COEFF, HP_COEFF, num_channels=1)
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(480).astype(np.float64)

    result = fg(samples)
    assert result.ndim == 2
    assert result.shape[0] == 1
    assert result.shape[1] == 480

    fg2 = FilterGroup(HS_COEFF, HP_COEFF, num_channels=1)
    samples_2d = samples[np.newaxis, :]
    result_2d = fg2(samples_2d)
    assert result == pytest.approx(result_2d)


@pytest.mark.benchmark(group='filter')
def test_filter_benchmark(benchmark, random_samples, num_channels, bench_sample_rate) -> None:
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
def test_truepeak_filter_benchmark(benchmark, random_samples, num_channels, bench_sample_rate) -> None:
    sample_rate = bench_sample_rate
    up_sample = 4 if sample_rate < 88200 else 2
    block_size = sample_rate // 100
    assert sample_rate % block_size == 0

    samples = random_samples(num_channels, block_size)
    assert samples.shape == (num_channels, block_size)
    tp_filter = TruePeakFilter(
        num_channels=num_channels,
        num_input_samples=block_size,
        upsample_factor=up_sample,
    )
    num_output_samples = samples.shape[1] * tp_filter.upsample_factor

    filtered = np.zeros((samples.shape[0], num_output_samples), dtype=np.float64)

    # The first streamed call returns fewer samples due to the filter delay
    tp_filter(samples)

    def bench():
        filtered[...] = tp_filter(samples)

    benchmark(bench)


def _reference_upsample(x: np.ndarray, up: int) -> np.ndarray:
    ref = ResamplePoly(
        up=up,
        down=1,
        window=calc_tp_fir_win(up),
        num_channels=x.shape[0],
        num_input_samples=x.shape[1],
    )
    return ref.apply(x).copy()


def _stream_upsample(
    x: np.ndarray, up: int, block_size: int, flush: bool = True
) -> np.ndarray:
    num_channels, n = x.shape
    tp_filter = TruePeakFilter(
        num_channels=num_channels, num_input_samples=block_size, upsample_factor=up,
    )
    outputs = []
    for i in range(0, n, block_size):
        block = x[:, i:i + block_size]
        if block.shape[1] < block_size:
            padded = np.zeros((num_channels, block_size), dtype=np.float64)
            padded[:, :block.shape[1]] = block
            block = padded
        outputs.append(tp_filter(block).copy())
    if flush:
        outputs.append(tp_filter.flush().copy())
    return np.concatenate(outputs, axis=1)[:, :n * up]


@pytest.mark.parametrize('up', [2, 4])
@pytest.mark.parametrize('block_size', [1, 5, 128, 1000])
@pytest.mark.parametrize('signal_type', ['sine', 'random'])
@pytest.mark.parametrize('length', [4000, 4013])
def test_truepeak_stream_matches_continuous(
    up: int, block_size: int, signal_type: str, length: int
) -> None:
    rng = np.random.default_rng(1)
    if signal_type == 'sine':
        t = np.arange(length) / 48000
        x = np.stack([np.sin(2 * np.pi * 997 * t), np.cos(2 * np.pi * 1234 * t)])
    else:
        x = rng.uniform(-1, 1, (2, length))
    x = np.ascontiguousarray(x, dtype=np.float64)
    ref = _reference_upsample(x, up)
    out = _stream_upsample(x, up, block_size)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize('up', [2, 4])
@pytest.mark.parametrize('peak_offset', [-1, 0, 1])
def test_truepeak_stream_peak_at_block_boundary(up: int, peak_offset: int) -> None:
    block_size = 128
    x = np.zeros((1, block_size * 4), dtype=np.float64)
    x[0, block_size * 2 + peak_offset] = 1.0
    x[0, block_size * 2 + peak_offset + 1] = 1.0
    ref = _reference_upsample(x, up)
    out = _stream_upsample(x, up, block_size)
    np.testing.assert_allclose(out, ref, atol=1e-12)
    assert np.abs(out).max() == pytest.approx(np.abs(ref).max())


@pytest.mark.parametrize('up', [2, 4])
def test_truepeak_filter_first_call_and_steady_output_size(up: int) -> None:
    block_size = 64
    tp_filter = TruePeakFilter(
        num_channels=1, num_input_samples=block_size, upsample_factor=up,
    )
    x = np.ones((1, block_size), dtype=np.float64)
    first = tp_filter(x)
    second = tp_filter(x)
    assert second.shape == (1, block_size * up)
    assert first.shape[1] < second.shape[1]
    assert first.shape[1] + tp_filter.flush().shape[1] == block_size * up


@pytest.mark.parametrize('up', [2, 4])
def test_truepeak_filter_reset(up: int) -> None:
    rng = np.random.default_rng(2)
    block_size = 100
    blocks = [rng.uniform(-1, 1, (2, block_size)) for _ in range(3)]
    f_new = TruePeakFilter(num_channels=2, num_input_samples=block_size, upsample_factor=up)
    expected = [f_new(b).copy() for b in blocks]

    f = TruePeakFilter(num_channels=2, num_input_samples=block_size, upsample_factor=up)
    for b in blocks:
        f(b)
    f.reset()
    for b, e in zip(blocks, expected):
        np.testing.assert_array_equal(f(b), e)


@pytest.mark.parametrize('block_size', [128, 4800, 9600, 19200])
def test_truepeak_issue_235_block_size_independent(block_size: int) -> None:
    sample_rate = 48000
    n = sample_rate * 20
    t = np.arange(n) / sample_rate
    amp = 10 ** (-23 / 20)
    x = (amp * np.sin(2 * np.pi * 997 * t)).reshape(1, n)
    x = np.ascontiguousarray(x, dtype=np.float64)
    ref_peak = np.abs(_reference_upsample(x, 4)).max()
    out = _stream_upsample(x, 4, block_size, flush=False)
    peak_db = 20 * np.log10(np.abs(out).max())
    assert np.abs(out).max() == pytest.approx(ref_peak, abs=1e-9)
    assert peak_db == pytest.approx(-23, abs=0.05)


@pytest.mark.parametrize('up', [2, 4])
def test_truepeak_filter_flush_after_reset(up: int) -> None:
    block_size = 64
    tp_filter = TruePeakFilter(
        num_channels=1, num_input_samples=block_size, upsample_factor=up,
    )
    tp_filter(np.ones((1, block_size), dtype=np.float64))
    tp_filter.reset()
    assert not np.any(tp_filter.flush())


def test_resample_poly_batch_apply_resizes_input() -> None:
    up = 4
    x = np.random.default_rng(3).uniform(-1, 1, (1, 300))
    resampler = ResamplePoly(
        up=up, down=1, window=calc_tp_fir_win(up), num_channels=1,
    )
    y = resampler.apply(x)
    assert resampler.num_input_samples == 300
    assert y.shape == (1, 300 * up)


def test_resample_poly_stream_input_validation() -> None:
    up = 4
    resampler = ResamplePoly(
        up=up,
        down=1,
        window=calc_tp_fir_win(up),
        num_channels=1,
        num_input_samples=32,
    )
    with pytest.raises(ValueError, match='dtype'):
        resampler.apply_stream(np.zeros((1, 32), dtype=np.float32))
    with pytest.raises(ValueError, match='shape'):
        resampler.apply_stream(np.zeros((1, 31), dtype=np.float64))


def test_resample_poly_stream_requires_no_downsampling() -> None:
    resampler = ResamplePoly(
        up=3,
        down=2,
        window=calc_tp_fir_win(3),
        num_channels=1,
        num_input_samples=32,
    )
    resampler.reset()
    with pytest.raises(ValueError, match='down == 1'):
        resampler.apply_stream(np.zeros((1, 32), dtype=np.float64))
    with pytest.raises(ValueError, match='down == 1'):
        resampler.flush()


def _make_stream_filter(up: int = 4, block_size: int = 64) -> TruePeakFilter:
    return TruePeakFilter(num_channels=2, num_input_samples=block_size, upsample_factor=up)


@pytest.mark.parametrize('up', [2, 4])
def test_resampler_flush_after_stream(up: int) -> None:
    rng = np.random.default_rng(5)
    block_size = 64
    x = rng.uniform(-1, 1, (2, block_size * 3))
    ref = _reference_upsample(x, up)

    tp_filter = _make_stream_filter(up, block_size)
    outputs = [
        tp_filter(x[:, i:i + block_size]).copy()
        for i in range(0, x.shape[1], block_size)
    ]
    tail = tp_filter.flush()
    assert tail.shape[1] > 0
    np.testing.assert_allclose(
        np.concatenate([*outputs, tail], axis=1), ref, atol=1e-12,
    )


def test_resampler_flush_is_idempotent() -> None:
    tp_filter = _make_stream_filter()
    tp_filter(np.ones((2, 64), dtype=np.float64))
    assert tp_filter.flush().shape[1] > 0
    for _ in range(2):
        tail = tp_filter.flush()
        assert tail.shape == (2, 0)


def test_resampler_stream_after_flush_raises() -> None:
    tp_filter = _make_stream_filter()
    tp_filter.flush()
    with pytest.raises(RuntimeError, match='reset'):
        tp_filter(np.ones((2, 64), dtype=np.float64))


@pytest.mark.parametrize('up', [2, 4])
def test_resampler_reset_after_flush(up: int) -> None:
    rng = np.random.default_rng(6)
    block_size = 64
    blocks = [rng.uniform(-1, 1, (2, block_size)) for _ in range(3)]

    def run(f: TruePeakFilter) -> np.ndarray:
        out = [f(b).copy() for b in blocks]
        out.append(f.flush().copy())
        return np.concatenate(out, axis=1)

    expected = run(_make_stream_filter(up, block_size))
    tp_filter = _make_stream_filter(up, block_size)
    first = run(tp_filter)
    tp_filter.reset()
    np.testing.assert_array_equal(run(tp_filter), expected)
    np.testing.assert_array_equal(first, expected)


def test_resampler_batch_apply_independent_of_stream_lifecycle() -> None:
    up = 4
    x = np.random.default_rng(7).uniform(-1, 1, (1, 128))
    resampler = ResamplePoly(
        up=up,
        down=1,
        window=calc_tp_fir_win(up),
        num_channels=1,
        num_input_samples=128,
    )
    expected = resampler.apply(x).copy()
    resampler.apply_stream(x)
    resampler.flush()
    np.testing.assert_array_equal(resampler.apply(x), expected)


@pytest.mark.parametrize('up', [2, 4])
def test_truepeak_filter_default_block_size(up: int) -> None:
    tp_filter = TruePeakFilter(num_channels=1, upsample_factor=up)
    x = np.zeros((1, DEFAULT_INPUT_SAMPLES), dtype=np.float64)
    tp_filter(x)
    assert tp_filter(x).shape == (1, DEFAULT_INPUT_SAMPLES * up)


def test_truepeak_filter_positional_args_and_explicit_block_size() -> None:
    tp_filter = TruePeakFilter(2, 2)
    assert tp_filter.upsample_factor == 2

    tp_filter = TruePeakFilter(2, 4, 256)
    tp_filter(np.zeros((2, 256), dtype=np.float64))
    with pytest.raises(ValueError, match='shape'):
        tp_filter(np.zeros((2, 128), dtype=np.float64))
