import pytest
import numpy as np

from lupy.sampling import Sampler, Slice, calc_buffer_length

# def test_slice():



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


@pytest.fixture(params=[False, True])
def use_random(request) -> bool:
    return request.param


def test_write(block_size, num_channels, random_samples, inc_samples, use_random):
    sampler = Sampler(block_size=block_size, num_channels=num_channels)
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
