from __future__ import annotations
from pathlib import Path

import pytest
import numpy as np
from lupy import Meter
from lupy.typeutils import ensure_2d_array

from compliance_cases import cases_by_name, all_cases, ComplianceBase

_cases = [
    cases_by_name['3341']['case1'],
    cases_by_name['3341']['case3'],
    cases_by_name['3341']['case5'],
    cases_by_name['3341']['case9'],
    cases_by_name['3341']['case20'],
    cases_by_name['3342']['case1'],
]

@pytest.fixture(
    params=_cases,
    ids=[f'{c.name} {i}' for i, c in enumerate(_cases)]
)
def io_compliance_case(request) -> ComplianceBase:
    return request.param


def test_meter_serialization(tmpdir, sample_rate, io_compliance_case: ComplianceBase):
    block_size = 128
    num_channels = 5
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    print('generating samples...')
    src_data = io_compliance_case.generate_samples(int(meter.sample_rate), block_size=block_size, num_channels=num_channels)
    N = src_data.shape[1]
    assert N % block_size == 0

    print(f'processing {N} samples...')
    meter.write_all(src_data)

    del src_data  # free memory

    out_file = tmpdir / 'meter.npz'
    print(f'saving meter to {out_file}...')
    meter.save(out_file)

    print('loading meter from file...')
    loaded_meter = Meter.load(out_file)
    assert loaded_meter.sample_rate == meter.sample_rate
    assert loaded_meter.block_size == meter.block_size
    assert loaded_meter.num_channels == meter.num_channels

    assert loaded_meter.integrated_lkfs == meter.integrated_lkfs
    assert loaded_meter.lra == meter.lra
    assert np.array_equal(loaded_meter.momentary_lkfs, meter.momentary_lkfs)
    assert np.array_equal(loaded_meter.short_term_lkfs, meter.short_term_lkfs)
    assert np.array_equal(loaded_meter.true_peak_max, meter.true_peak_max)
    assert np.array_equal(loaded_meter.t, meter.t)
    assert np.array_equal(loaded_meter.block_data, meter.block_data)

    blk, loaded_blk = meter.processor, loaded_meter.processor
    tp, loaded_tp = meter.true_peak_processor, loaded_meter.true_peak_processor

    assert blk.sample_rate == loaded_blk.sample_rate
    assert blk.num_channels == loaded_blk.num_channels
    assert blk.gate_size == loaded_blk.gate_size
    assert blk.block_index == loaded_blk.block_index
    assert blk.num_blocks == loaded_blk.num_blocks
    assert blk.integrated_lkfs == loaded_blk.integrated_lkfs
    assert blk.lra == loaded_blk.lra
    assert blk._rel_threshold == pytest.approx(loaded_blk._rel_threshold)
    assert np.array_equal(blk._block_data, loaded_blk._block_data)
    assert np.array_equal(blk.Zij, loaded_blk.Zij)
    assert np.array_equal(blk._block_weighted_sums, loaded_blk._block_weighted_sums)
    assert np.array_equal(blk._quarter_block_weighted_sums, loaded_blk._quarter_block_weighted_sums)
    assert np.array_equal(blk._block_loudness, loaded_blk._block_loudness)
    # assert np.array_equal(blk._blocks_above_abs_thresh, loaded_blk._blocks_above_abs_thresh)
    # assert np.array_equal(blk._blocks_above_rel_thresh, loaded_blk._blocks_above_rel_thresh)

    assert tp.sample_rate == loaded_tp.sample_rate
    assert tp.num_channels == loaded_tp.num_channels
    assert tp.max_peak == loaded_tp.max_peak
    assert np.array_equal(tp.current_peaks, loaded_tp.current_peaks)


def test_meter_resume_processing(tmpdir, sample_rate, io_compliance_case: ComplianceBase):
    compliance_case = io_compliance_case

    # Use a block size of 100 ms to make sure we don't split on a block boundary
    block_size = sample_rate / 10
    assert block_size % 1 == 0
    block_size = int(block_size)
    num_channels = 5
    meter = Meter(block_size=block_size, num_channels=num_channels, sample_rate=sample_rate)

    print('generating samples...')
    src_data = compliance_case.generate_samples(int(meter.sample_rate), block_size=block_size, num_channels=num_channels)
    N = src_data.shape[1]
    assert N % block_size == 0

    n_first_half = N // 2
    if n_first_half % block_size != 0:
        n_first_half -= n_first_half % block_size

    first_half = ensure_2d_array(src_data[:, :n_first_half])
    second_half = ensure_2d_array(src_data[:, n_first_half:])
    del src_data

    print(f'processing first half of {N} samples...')
    meter.write_all(first_half)

    del first_half  # free memory

    out_file = tmpdir / 'meter.npz'
    print(f'saving meter to {out_file}...')
    meter.save(out_file)

    del meter

    print('loading meter from file...')
    meter = Meter.load(out_file)
    assert meter.sample_rate == sample_rate
    assert meter.block_size == block_size
    assert meter.num_channels == num_channels

    print(f'resuming processing with second half of {N} samples...')
    meter.write_all(second_half)

    del second_half  # free memory

    integrated = meter.integrated_lkfs
    momentary, short_term = meter.momentary_lkfs[-1], meter.short_term_lkfs[-1]

    integrated_target = compliance_case.result.integrated
    momentary_target = compliance_case.result.momentary
    short_term_target = compliance_case.result.short_term
    lra_target = compliance_case.result.lra
    tp_target = compliance_case.result.true_peak

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
        assert lra_lu - tol <= meter.lra <= lra_lu + tol
    if tp_target is not None:
        tp, neg_tol, pos_tol = tp_target
        tp_min, tp_max = tp - neg_tol, tp + pos_tol
        assert tp_min <= meter.true_peak_max <= tp_max
