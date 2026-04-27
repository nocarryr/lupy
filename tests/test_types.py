from __future__ import annotations

import numpy as np
import pytest

from lupy.typeutils import (
    build_true_peak_array, is_true_peak_array, build_true_peak_dtype,
    ensure_3d_array, ensure_nd_array, ensure_array_of_shape,
    is_array_of_dtype, is_float_array, is_float32_array, is_index_array,
    is_bool_array, is_complex_array,
    ensure_meter_array, ensure_true_peak_array, build_meter_array,
)


# ---------------------------------------------------------------------------
# ensure_*d_array helpers
# ---------------------------------------------------------------------------

def test_ensure_3d_array_happy():
    arr = np.zeros((2, 3, 4))
    assert ensure_3d_array(arr) is arr


def test_ensure_3d_array_wrong_ndim():
    arr = np.zeros((2, 3))
    with pytest.raises(AssertionError):
        ensure_3d_array(arr)


def test_ensure_nd_array_happy():
    arr = np.zeros((2, 3, 4, 5))
    assert ensure_nd_array(arr, 4) is arr


def test_ensure_nd_array_wrong_ndim():
    arr = np.zeros((2, 3))
    with pytest.raises(AssertionError):
        ensure_nd_array(arr, 3)


def test_ensure_array_of_shape_happy():
    arr = np.zeros((2, 3))
    assert ensure_array_of_shape(arr, (2, 3)) is arr


def test_ensure_array_of_shape_wrong_shape():
    arr = np.zeros((2, 3))
    with pytest.raises(AssertionError):
        ensure_array_of_shape(arr, (2, 4))


# ---------------------------------------------------------------------------
# is_* dtype helpers
# ---------------------------------------------------------------------------

def test_is_array_of_dtype():
    arr_f64 = np.zeros(4, dtype=np.float64)
    arr_i32 = np.zeros(4, dtype=np.int32)
    assert is_array_of_dtype(arr_f64, np.dtype(np.float64))
    assert not is_array_of_dtype(arr_i32, np.dtype(np.float64))


def test_is_float_array():
    assert is_float_array(np.zeros(4, dtype=np.float64))
    assert is_float_array(np.zeros(4, dtype=np.float32))
    assert not is_float_array(np.zeros(4, dtype=np.int32))


def test_is_float32_array():
    assert is_float32_array(np.zeros(4, dtype=np.float32))
    assert not is_float32_array(np.zeros(4, dtype=np.float64))


def test_is_index_array():
    assert is_index_array(np.zeros(4, dtype=np.intp))
    assert not is_index_array(np.zeros(4, dtype=np.int32))


def test_is_bool_array():
    assert is_bool_array(np.zeros(4, dtype=np.bool_))
    assert not is_bool_array(np.zeros(4, dtype=np.float64))


def test_is_complex_array():
    assert is_complex_array(np.zeros(4, dtype=np.complex128))
    assert not is_complex_array(np.zeros(4, dtype=np.float64))


# ---------------------------------------------------------------------------
# ensure_meter_array
# ---------------------------------------------------------------------------

def test_ensure_meter_array_happy():
    arr = build_meter_array(5)
    assert ensure_meter_array(arr) is arr


def test_ensure_meter_array_wrong_dtype():
    arr = np.zeros(5, dtype=np.float64)
    with pytest.raises(AssertionError):
        ensure_meter_array(arr)


# ---------------------------------------------------------------------------
# ensure_true_peak_array (happy path + channel mismatch + wrong dtype + wrong shape)
# ---------------------------------------------------------------------------

def test_ensure_true_peak_array_happy():
    num_channels = 2
    arr = build_true_peak_array(num_channels, 5)
    assert ensure_true_peak_array(arr, num_channels) is arr


def test_ensure_true_peak_array_channel_mismatch():
    arr = build_true_peak_array(2, 5)
    with pytest.raises(AssertionError):
        ensure_true_peak_array(arr, 3)


def test_ensure_true_peak_array_wrong_dtype():
    """A TruePeakArray built for 3 channels fails ensure_true_peak_array when num_channels=2.

    The shape (5,) is valid; only the structured dtype differs (3-channel vs 2-channel),
    so the test isolates the dtype check.
    """
    arr = build_true_peak_array(3, 5)
    with pytest.raises(AssertionError):
        ensure_true_peak_array(arr, 2)


def test_true_peak_array_matching():
    num_channels = 3
    size = 5
    arr1 = build_true_peak_array(num_channels, size)
    assert is_true_peak_array(arr1, num_channels)
    assert arr1.dtype == build_true_peak_dtype(num_channels)
    assert arr1.shape == (size,)

    assert is_true_peak_array(arr1.copy(), num_channels)
    assert is_true_peak_array(arr1.view(arr1.dtype), num_channels)
    assert is_true_peak_array(arr1[:-1], num_channels)

    arr2 = build_true_peak_array(2, size)
    assert is_true_peak_array(arr2, 2)
    assert not is_true_peak_array(arr2, num_channels)
    assert not is_true_peak_array(np.zeros((size,)), num_channels)
