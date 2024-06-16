from __future__ import annotations

from typing import Callable, Iterable
import pytest
import numpy as np

from lupy import Sampler, BlockProcessor
from lupy.types import FloatArray

from conftest import gen_1k_sine


@pytest.fixture(params=[True, False])
def is_silent(request) -> bool:
    return request.param

@pytest.fixture(params=[False])
def reset_process(request) -> bool:
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

def iter_process(sampler: Sampler, processor: BlockProcessor, src_data: FloatArray):
    assert src_data.ndim == 3

    num_channels, num_blocks, block_size = src_data.shape
    assert num_channels == sampler.num_channels
    assert block_size == sampler.block_size

    write_index = 0

    while write_index < num_blocks:
        while sampler.can_write() and write_index < num_blocks:
            sampler.write(src_data[:,write_index,:])
            write_index += 1

        while sampler.can_read():
            blk_samples = sampler.read()
            assert blk_samples.shape == (num_channels, sampler.gate_size)
            block_index = processor.block_index
            processor.process_block(blk_samples)
            yield block_index

def process_all(sampler: Sampler, processor: BlockProcessor, src_data: FloatArray):
    for _ in iter_process(sampler, processor, src_data):
        pass


def test_integrated_lkfs(block_size, all_channels, is_silent, reset_process):
    num_channels, sine_channel = all_channels
    sampler = Sampler(block_size=block_size, num_channels=num_channels)

    N, Fs = sampler.total_samples, int(sampler.sample_rate)
    num_blocks, gate_size = sampler.num_blocks, sampler.gate_size

    processor = BlockProcessor(num_channels=num_channels, gate_size=gate_size)

    _sine_channels = None if is_silent else [sine_channel]
    src_data = build_samples(N, num_channels, Fs, _sine_channels, 1)
    src_data = src_data.reshape((num_channels, num_blocks, block_size))

    if reset_process:
        for block_index in iter_process(sampler, processor, src_data):
            if block_index >= num_blocks // 2:
                break
        processor.reset()

    for block_index in iter_process(sampler, processor, src_data):
        print(f'{block_index=}, {processor._rel_threshold=}, {processor.integrated_lkfs=}')


    print(f'{N=}, {N / gate_size}, {len(processor)=}')

    lkfs = round(processor.integrated_lkfs, 2)
    if is_silent:
        assert processor.integrated_lkfs <= 120
    elif sine_channel < 3:
        assert round(processor.integrated_lkfs, 2) == -3.01
        # assert -3.02 <= lkfs <= -3.00
    else:
        assert round(processor.integrated_lkfs, 2) == -1.52
        # assert -1.53 <= lkfs <= -1.51


# https://tech.ebu.ch/docs/tech/tech3341.pdf Section 2.9
# Stereo 1k (997 Hz) sine at -18 dBFS should read -18 LUFS
def test_integrated_lkfs_neg18(block_size):
    num_channels = 2
    sampler = Sampler(block_size=block_size, num_channels=num_channels)

    N, Fs = sampler.total_samples, int(sampler.sample_rate)
    num_blocks, gate_size = sampler.num_blocks, sampler.gate_size

    processor = BlockProcessor(num_channels=num_channels, gate_size=gate_size)

    sine_channels = [0, 1]
    amp = 10 ** (-18/20)
    src_data = build_samples(N, num_channels, Fs, sine_channels, amp)
    src_data = src_data.reshape((num_channels, num_blocks, block_size))

    for block_index in iter_process(sampler, processor, src_data):
        print(f'{block_index=}, {processor._rel_threshold=}, {processor.integrated_lkfs=}')

    print(f'{N=}, {N / gate_size}, {len(processor)=}')

    assert round(processor.integrated_lkfs, 2) == -18

def test_tech_3341_compliance(tech_3341_compliance_case):
    block_size = 128
    num_channels = 5
    sampler = Sampler(block_size=block_size, num_channels=num_channels)

    processor = BlockProcessor(
        num_channels=num_channels, gate_size=sampler.gate_size
    )
    print('generating samples...')
    src_data = tech_3341_compliance_case.generate_samples(int(sampler.sample_rate))
    N = src_data.shape[1]
    # assert N % block_size == 0
    # num_blocks = N // block_size
    remain = N % block_size
    if remain > 0:
        src_data = src_data[:,:N-remain]
    num_blocks = N // block_size

    src_data = np.reshape(src_data, (num_channels, num_blocks, block_size))

    print(f'processing {N} samples...')

    # for block_index in iter_process(sampler, processor, src_data):
    #     # print(f'{block_index=}, {processor.integrated_lkfs=}, {sampler.samples_available=}')
    #     pass
    process_all(sampler, processor, src_data)

    print(f'{processor.t[-1]=}')
    integrated = processor.integrated_lkfs
    momentary, short_term = processor.momentary_lkfs[-1], processor.short_term_lkfs[-1]
    print(f'{integrated=}, {momentary=}, {short_term=}')
    print(f'{processor._rel_threshold=}')

    integrated_target = tech_3341_compliance_case.result.integrated
    momentary_target = tech_3341_compliance_case.result.momentary
    short_term_target = tech_3341_compliance_case.result.short_term
    lra_target = tech_3341_compliance_case.result.lra

    if momentary_target is not None:
        lufs, lu, tol = momentary_target
        assert lufs - tol <= momentary <= lufs + tol
    if short_term_target is not None:
        lufs, lu, tol = short_term_target
        assert lufs - tol <= short_term <= lufs + tol
    if integrated_target is not None:
        lufs, lu, tol = integrated_target
        assert lufs - tol <= integrated <= lufs + tol
    if lra_target is not None:
        lra_lu, tol = lra_target
        assert lra_lu - tol <= processor.lra <= lra_lu + tol

def test_tech_3342_compliance(tech_3342_compliance_case):
    block_size = 128
    num_channels = 5
    sampler = Sampler(block_size=block_size, num_channels=num_channels)

    processor = BlockProcessor(
        num_channels=num_channels, gate_size=sampler.gate_size
    )
    print('generating samples...')
    src_data = tech_3342_compliance_case.generate_samples(int(sampler.sample_rate))
    N = src_data.shape[1]
    # assert N % block_size == 0
    # num_blocks = N // block_size
    remain = N % block_size
    if remain > 0:
        src_data = src_data[:,:N-remain]
    num_blocks = N // block_size

    src_data = np.reshape(src_data, (num_channels, num_blocks, block_size))

    print(f'processing {N} samples...')

    # for block_index in iter_process(sampler, processor, src_data):
    #     # print(f'{block_index=}, {processor.integrated_lkfs=}, {sampler.samples_available=}')
    #     pass
    process_all(sampler, processor, src_data)

    print(f'{processor.t[-1]=}')
    integrated = processor.integrated_lkfs
    momentary, short_term = processor.momentary_lkfs[-1], processor.short_term_lkfs[-1]
    print(f'{integrated=}, {momentary=}, {short_term=}')
    print(f'{processor._rel_threshold=}')

    integrated_target = tech_3342_compliance_case.result.integrated
    momentary_target = tech_3342_compliance_case.result.momentary
    short_term_target = tech_3342_compliance_case.result.short_term
    lra_target = tech_3342_compliance_case.result.lra

    if momentary_target is not None:
        lufs, lu, tol = momentary_target
        assert lufs - tol <= momentary <= lufs + tol
    if short_term_target is not None:
        lufs, lu, tol = short_term_target
        assert lufs - tol <= short_term <= lufs + tol
    if integrated_target is not None:
        lufs, lu, tol = integrated_target
        assert lufs - tol <= integrated <= lufs + tol
    if lra_target is not None:
        lra_lu, tol = lra_target
        assert lra_lu - tol <= processor.lra <= lra_lu + tol
