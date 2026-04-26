from __future__ import annotations

import pytest
import numpy as np

from lupy.typeutils import (
    build_true_peak_array, is_true_peak_array, build_true_peak_dtype,
    ensure_3d_array, ensure_nd_array, ensure_array_of_shape,
    is_array_of_dtype, is_float_array, is_float32_array,
    is_index_array, is_bool_array, is_complex_array,
    ensure_meter_array, ensure_true_peak_array, build_meter_array,
)


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


def test_ensure_3d_array():
    arr3d = np.zeros((2, 3, 4))
    assert ensure_3d_array(arr3d) is arr3d
    with pytest.raises(AssertionError):
        ensure_3d_array(np.zeros((2, 3)))  # type: ignore[arg-type]


def test_ensure_nd_array():
    arr = np.zeros((2, 3))
    assert ensure_nd_array(arr, 2) is arr
    with pytest.raises(AssertionError):
        ensure_nd_array(arr, 3)


def test_ensure_array_of_shape():
    arr = np.zeros((3, 4))
    assert ensure_array_of_shape(arr, (3, 4)) is arr
    with pytest.raises(AssertionError):
        ensure_array_of_shape(arr, (4, 3))


def test_is_array_of_dtype():
    arr = np.zeros(5, dtype=np.float64)
    assert is_array_of_dtype(arr, np.dtype(np.float64))
    assert not is_array_of_dtype(arr, np.dtype(np.float32))


def test_is_float_array():
    assert is_float_array(np.zeros(3, dtype=np.float64))
    assert is_float_array(np.zeros(3, dtype=np.float32))
    assert not is_float_array(np.zeros(3, dtype=np.int32))


def test_is_float32_array():
    assert is_float32_array(np.zeros(3, dtype=np.float32))
    assert not is_float32_array(np.zeros(3, dtype=np.float64))


def test_is_index_array():
    assert is_index_array(np.zeros(3, dtype=np.intp))
    assert not is_index_array(np.zeros(3, dtype=np.int32))


def test_is_bool_array():
    assert is_bool_array(np.zeros(3, dtype=np.bool_))
    assert not is_bool_array(np.zeros(3, dtype=np.uint8))


def test_is_complex_array():
    assert is_complex_array(np.zeros(3, dtype=np.complex128))
    assert not is_complex_array(np.zeros(3, dtype=np.float64))


def test_ensure_meter_array():
    arr = build_meter_array(5)
    assert ensure_meter_array(arr) is arr
    with pytest.raises(AssertionError):
        ensure_meter_array(np.zeros(5, dtype=np.float64))


def test_ensure_true_peak_array():
    arr = build_true_peak_array(2, 5)
    assert ensure_true_peak_array(arr, 2) is arr
    with pytest.raises(AssertionError):
        ensure_true_peak_array(arr, 3)
