"""Tests for signalutils error paths and edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from lupy.signalutils.sosfilt import validate_sos, sosfilt
from lupy.signalutils.resample import _UpFIRDn, ResamplePoly, calc_tp_fir_win
from lupy.typeutils import ensure_2d_array


def test_validate_sos_not_2d():
    """validate_sos raises ValueError for non-2D input."""
    with pytest.raises(ValueError, match='sos array must be 2D'):
        validate_sos(np.ones(6))  # type: ignore[arg-type]


def test_validate_sos_wrong_columns():
    """validate_sos raises ValueError when column count is not 6."""
    with pytest.raises(ValueError, match='sos array must be shape'):
        validate_sos(ensure_2d_array(np.ones((2, 5))))


def test_validate_sos_nonunit_denominator():
    """validate_sos raises ValueError when sos[:, 3] is not all ones."""
    sos = np.ones((2, 6))
    sos[0, 3] = 2.0
    with pytest.raises(ValueError, match=r'sos\[:, 3\] should be all ones'):
        validate_sos(ensure_2d_array(sos))


def test_validate_sos_returns_same_array():
    """validate_sos returns the array unchanged for valid input."""
    sos = np.ones((3, 6))
    result = validate_sos(ensure_2d_array(sos))
    assert result is sos


def test_sosfilt_zi_shape_mismatch():
    """sosfilt raises ValueError when zi has the wrong shape."""
    sos = np.ones((2, 6))
    x = np.zeros((1, 100))
    zi_wrong = np.zeros((2, 1, 3))  # correct would be (2, 1, 2)
    with pytest.raises(ValueError, match='Invalid zi shape'):
        sosfilt(sos, x, zi_wrong)  # type: ignore[arg-type]


def test_upfirdn_h_not_1d():
    """_UpFIRDn raises ValueError when h is not 1-D."""
    with pytest.raises(ValueError, match='h must be 1-D'):
        _UpFIRDn(h=np.ones((3, 4)), up=1, down=1, input_shape=(1, 16))  # type: ignore[arg-type]


def test_upfirdn_h_empty():
    """_UpFIRDn raises ValueError when h is an empty 1-D array."""
    with pytest.raises(ValueError, match='h must be 1-D'):
        _UpFIRDn(h=np.array([]), up=1, down=1, input_shape=(1, 16))


def test_upfirdn_up_zero():
    """_UpFIRDn raises ValueError when up == 0."""
    with pytest.raises(ValueError, match='Both up and down must be >= 1'):
        _UpFIRDn(h=np.ones(5), up=0, down=1, input_shape=(1, 16))


def test_upfirdn_down_zero():
    """_UpFIRDn raises ValueError when down == 0."""
    with pytest.raises(ValueError, match='Both up and down must be >= 1'):
        _UpFIRDn(h=np.ones(5), up=1, down=0, input_shape=(1, 16))


def test_upfirdn_apply_filter_dtype_mismatch():
    """_UpFIRDn.apply_filter raises ValueError for non-float64 input."""
    h = calc_tp_fir_win(4)
    filt = _UpFIRDn(h=h, up=4, down=1, input_shape=(1, 100))
    x_float32 = np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match='dtype'):
        filt.apply_filter(x_float32)  # type: ignore[arg-type]



@pytest.fixture
def resample_poly_128():
    h = calc_tp_fir_win(4)
    return ResamplePoly(up=4, down=1, num_channels=1, window=h, num_input_samples=128)


def test_resample_poly_num_input_samples_same_value_is_noop(resample_poly_128):
    """Setting num_input_samples to the current value skips recalculation."""
    resampler = resample_poly_128
    original_params = resampler.params
    original_upfirdn = resampler._upfirdn

    resampler.num_input_samples = 128  # same value — should not recalculate

    assert resampler.params is original_params
    assert resampler._upfirdn is original_upfirdn


def test_resample_poly_num_input_samples_change_recalculates(resample_poly_128):
    """Setting num_input_samples to a different value triggers recalculation."""
    resampler = resample_poly_128
    original_params = resampler.params

    resampler.num_input_samples = 256  # different value — must recalculate

    assert resampler.params is not original_params
    assert resampler.params.n_in == 256


def test_resample_poly_num_output_samples(resample_poly_128) -> None:
    """ResamplePoly.num_output_samples returns the expected output sample count.

    For the resample_poly_128 fixture (up=4, down=1, num_input_samples=128),
    expected n_out = 128 * 4 = 512.
    """
    resampler = resample_poly_128
    assert resampler.num_output_samples == 512


def test_resample_poly_output_shape(resample_poly_128) -> None:
    """ResamplePoly.output_shape returns (num_channels, num_output_samples).

    For the resample_poly_128 fixture (num_channels=1, n_out=512),
    expected shape is (1, 512).
    """
    resampler = resample_poly_128
    assert resampler.output_shape == (1, 512)
