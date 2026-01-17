from __future__ import annotations

import numpy as np

from lupy.typeutils import build_true_peak_array, is_true_peak_array, build_true_peak_dtype


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
