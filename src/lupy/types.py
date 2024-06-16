from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt



__all__ = (
    'Floating', 'Complex', 'MeterDtype', 'MeterArray',
    'AnyArray', 'BoolArray', 'IndexArray', 'FloatArray', 'ComplexArray',
    'Float1dArray', 'Float2dArray', 'Float3dArray',
)


Floating = np.floating
Complex = np.complex128

MeterDtype = np.dtype([
    ('t', np.float64),
    ('m', np.float64),
    ('s', np.float64),
])
"""Structured data type for loudness results

.. attribute:: t
    :type: numpy.float64

    The time in seconds for each measurement

.. attribute:: m
    :type: numpy.float64

    The :term:`Momentary Loudness` at time :attr:`t`

.. attribute:: s
    :type: numpy.float64

    The :term:`Short-Term Loudness` at time :attr:`t`

"""



AnyArray = npt.NDArray[Any]
BoolArray = npt.NDArray[np.bool_]
IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[Floating]
ComplexArray = npt.NDArray[Complex]
MeterArray = npt.NDArray[np.void]   #: Array with dtype :obj:`MeterDtype`
Float1dArray = npt.NDArray[Floating]
Float2dArray = npt.NDArray[Floating]
Float3dArray = npt.NDArray[Floating]
