from __future__ import annotations

import numpy as np
from scipy import signal
import pytest

from lupy.filters import HS_COEFF, HP_COEFF


@pytest.mark.parametrize('coeff', [HS_COEFF, HP_COEFF])
def test_filter_requantize(coeff, sample_rate):
    orig_sample_rate = coeff.sample_rate
    quantized = coeff.as_sample_rate(sample_rate)

    re_quantized = quantized.as_sample_rate(orig_sample_rate)
    assert np.allclose(re_quantized.b, coeff.b)
    assert np.allclose(re_quantized.a, coeff.a)

    w0, h0 = signal.freqz(coeff.b, coeff.a)
    w1, h1 = signal.freqz(re_quantized.b, re_quantized.a)
    assert np.allclose(w0, w1)
    assert np.allclose(h0, h1)
