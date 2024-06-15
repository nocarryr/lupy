from typing import Any
import warnings

import numpy as np
import numpy.typing as npt

with warnings.catch_warnings(category=DeprecationWarning):
    warnings.simplefilter('ignore')
    from nptyping import NDArray, Structure


__all__ = (
    'Floating', 'Complex', 'MeterDtype', 'MeterStruct', 'MeterArray',
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

MeterStruct = Structure['t: Floating, m: Floating, s: Floating']


AnyArray = npt.NDArray[Any]
BoolArray = npt.NDArray[np.bool_]
IndexArray = npt.NDArray[np.intp]
FloatArray = npt.NDArray[Floating]
ComplexArray = npt.NDArray[Complex]
MeterArray = NDArray[Any, MeterStruct]
Float1dArray = npt.NDArray[Floating]#NDArray[Shape['*'], _Floating]
Float2dArray = npt.NDArray[Floating]#NDArray[Shape['*, *'], _Floating]
Float3dArray = npt.NDArray[Floating]#NDArray[Shape['*, *, *'], _Floating]
