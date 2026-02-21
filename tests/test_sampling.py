from __future__ import annotations

import pytest
import numpy as np

from lupy.sampling import Sampler, TruePeakSampler, calc_buffer_length, Slice



def test_slice_non_overlap():
    src_array_shape = (2, 48000)
    chunk_size = 1000
    overlap = 0
    num_chunks = src_array_shape[1] // chunk_size
    assert src_array_shape[1] % chunk_size == 0

    src_array = np.zeros(src_array_shape, dtype=int)
    src_array[0, :] = np.arange(src_array_shape[1])
    src_array[1, :] = np.arange(src_array_shape[1]) * 10
    expected_array = np.reshape(src_array, (2, num_chunks, chunk_size))

    slc = Slice(
        step=chunk_size,
        max_index=num_chunks-1,
        overlap=overlap,
    )

    assert slc.calc_shape(src_array, axis=1) == (2, chunk_size)

    for i in range(num_chunks):
        assert slc.index == i
        assert not slc.is_wrapped(src_array, axis=1)
        chunk = slc.slice(src_array, axis=1)
        assert chunk.shape == (2, chunk_size)
        assert np.array_equal(chunk, expected_array[:, i, :])
        slc.increment(src_array, axis=1)

    assert slc.index == 0
    chunk = slc.slice(src_array, axis=1)
    assert chunk.shape == (2, chunk_size)
    assert np.array_equal(chunk, expected_array[:, 0, :])


def test_slice_with_overlap():
    N = 48000
    src_array_shape = (2, N)
    chunk_size = 1000
    overlap_pct = 0.75
    step = 1 - overlap_pct
    step_count = int(step * chunk_size)

    def get_chunk_indices(j: int) -> tuple[int, int]:
        start_ix = int(chunk_size * (j * step))
        end_ix = int(chunk_size * (j * step + 1))
        return start_ix, end_ix

    # j \in 0, 1, \ldots, \frac{N - chunk\_size}{chunk\_size \cdot step}
    num_chunks = int(np.floor((N - chunk_size) / (chunk_size * step))) + 1

    src_array = np.zeros(src_array_shape, dtype=int)
    src_array[0, :] = np.arange(src_array_shape[1])
    src_array[1, :] = np.arange(src_array_shape[1]) * 10

    expected_array = np.zeros((2, num_chunks, chunk_size), dtype=int)
    for i in range(num_chunks):
        start_ix, end_ix = get_chunk_indices(i)
        values = np.arange(start_ix, end_ix)

        expected_array[0, i, :] = values
        expected_array[1, i, :] = values * 10

    slc = Slice(
        step=chunk_size,
        max_index=0,
        overlap=step_count,
    )

    assert slc.calc_shape(src_array, axis=1) == (src_array_shape[0], chunk_size)

    for i in range(num_chunks):
        start_ix, end_ix = get_chunk_indices(i)
        assert slc.index == i
        if slc.end_index > src_array.shape[1]:
            assert slc.is_wrapped(src_array, axis=1)
        else:
            assert not slc.is_wrapped(src_array, axis=1)
            assert slc.start_index == start_ix
            assert slc.end_index == end_ix
        chunk = slc.slice(src_array, axis=1)
        assert chunk.shape == (src_array_shape[0], chunk_size)
        assert np.array_equal(chunk, expected_array[:, i, :])
        slc.increment(src_array, axis=1)

    assert end_ix == src_array.shape[1]
    i = num_chunks
    real_i = num_chunks
    wrap_completed = False
    while i < num_chunks * 2:
        start_ix, end_ix = get_chunk_indices(i)
        is_wrapped = start_ix < src_array.shape[1] and end_ix > src_array.shape[1]
        assert slc.is_wrapped(src_array, axis=1) == is_wrapped
        if is_wrapped:
            if wrap_completed:
                break
            assert slc.index == i
            chunk_start = src_array[:, start_ix:]
            chunk_end = src_array[:, :end_ix % src_array.shape[1]]
            chunk_expected = np.concatenate((chunk_start, chunk_end), axis=1)
        else:
            if not wrap_completed:
                real_i = 0
                wrap_completed = True
            start_ix, end_ix = get_chunk_indices(real_i)
            assert slc.index == real_i
            chunk_expected = expected_array[:, real_i, :]

        assert slc.start_index == start_ix
        assert slc.end_index == end_ix
        chunk = slc.slice(src_array, axis=1)
        assert chunk.shape == (src_array_shape[0], chunk_size)
        assert np.array_equal(chunk, chunk_expected)

        slc.increment(src_array, axis=1)
        i += 1
        real_i += 1

    assert wrap_completed



def test_buffer_length(sample_rate, block_size):
    bfr_shape = calc_buffer_length(sample_rate, block_size)
    print(f'{block_size=} {bfr_shape=}')
    assert bfr_shape.total_samples % bfr_shape.block_size == 0
    assert bfr_shape.total_samples % bfr_shape.pad_size == 0
    T_g = .4
    overlap = .75
    step = 1 - overlap
    N = bfr_shape.total_samples
    T = N / sample_rate
    n_blocks = int(np.round(((T - T_g) / (T_g * step)))+1)
    assert bfr_shape.num_gate_blocks == n_blocks
    assert bfr_shape.gate_size / bfr_shape.pad_size == 4


def test_true_peak_gate_size(true_peak_gate_duration, sample_rate, block_size):
    gate_size = int(true_peak_gate_duration * sample_rate)
    # Ensure the gate duration in samples is an exact integer (no rounding error)
    assert gate_size == true_peak_gate_duration * sample_rate
    assert gate_size > 0
    sampler = TruePeakSampler(
        block_size=block_size,
        num_channels=1,
        sample_rate=sample_rate,
        gate_duration=true_peak_gate_duration,
    )
    assert sampler.gate_size == gate_size
    assert sampler.num_gate_blocks >= 1


@pytest.fixture(params=[False, True])
def use_random(request) -> bool:
    return request.param


def test_write(sample_rate, block_size, num_channels, random_samples, inc_samples, use_random):
    sampler = Sampler(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)
    print(f'{sampler.bfr_shape=}')
    print(f'{sampler.gate_view.shape=}')

    assert sampler.sample_array.shape[0] == num_channels
    num_blocks = sampler.num_blocks
    total_samples = sampler.total_samples
    pad_size, gate_size = sampler.pad_size, sampler.gate_size

    # assert sampler.gate_view.shape == (sampler.num_gate_blocks, num_channels, gate_size)

    if use_random:
        src_data = random_samples(num_channels, num_blocks*2, block_size)
    else:
        src_data = inc_samples(
            count=total_samples*2,
            shape=(num_channels, num_blocks*2, block_size),
        )
        for i in range(num_channels):
            src_data[i,...] += i
        if num_channels > 1:
            assert np.array_equal(src_data[:,0,0], np.arange(num_channels))

    src_data_flat = np.reshape(src_data, (num_channels, total_samples*2))
    src_data_index = 0
    write_block_index = 0
    num_written = 0
    num_read = 0

    def write_samples():
        nonlocal write_block_index
        nonlocal num_written

        assert sampler.can_write()
        sampler.write(src_data[:,write_block_index,:], apply_filter=False)
        write_block_index += 1
        num_written += block_size
        if num_written >= gate_size:
            assert sampler.can_read()
        else:
            assert not sampler.can_read()

    def get_gate_block_bounds(j: int) -> tuple[int, int]:
        T_g = .4
        overlap = .75
        step = 1 - overlap
        l = int(T_g * (j * step    ) * sampler.sample_rate)
        u = int(T_g * (j * step + 1) * sampler.sample_rate)
        return l, u

    def read_block():
        nonlocal src_data_index
        nonlocal num_read
        nonlocal num_written

        assert sampler.can_read()
        read_data = sampler.read()
        num_read += pad_size
        num_written -= pad_size

        assert read_data.shape == (num_channels, gate_size)
        lb, ub = get_gate_block_bounds(src_data_index)
        _src_data = src_data_flat[:,lb:ub]
        src_data_index += 1
        # _src_data = src_data_flat[:,src_data_index:src_data_index+gate_size]
        # src_data_index += pad_size
        assert _src_data.shape == read_data.shape
        assert np.array_equal(read_data, _src_data)
        if num_read >= block_size:
            assert sampler.can_write()
        else:
            assert not sampler.can_write()

    print('writing')
    for i in range(num_blocks):
        write_samples()
    print(f'{num_written=}, {num_read=}')
    assert not sampler.can_write()

    print('reading')
    while sampler.can_read():
        read_block()
    print(f'{num_written=}, {num_read=}')
    assert not sampler.can_read()
    assert sampler.can_write()

    print('writing')
    while sampler.can_write():
        write_samples()
    print(f'{num_written=}, {num_read=}')
    assert sampler.can_read()

    print('reading')
    while sampler.can_read():
        read_block()
    print(f'{num_written=}, {num_read=}')
    assert not sampler.can_read()

    assert write_block_index < src_data.shape[1]

    # this should make the read buffer wrap around
    print('writing')
    while sampler.can_write() and write_block_index < src_data.shape[1]:
        write_samples()
    print(f'{num_written=}, {num_read=}')
    assert sampler.can_read()

    print('reading')
    while sampler.can_read():
        read_block()
    print(f'{num_written=}, {num_read=}')
    assert not sampler.can_read()
