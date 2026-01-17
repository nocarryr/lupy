from __future__ import annotations

from typing import TypeVar, Generic, Literal, Any
import sys
if sys.version_info < (3, 11):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

import numpy as np
import numpy.typing as npt



__all__ = (
    'Floating', 'Complex', 'MeterDtype', 'MeterArray', 'SosCoeff', 'SosZI',
    'AnyArray', 'BoolArray', 'IndexArray', 'FloatArray', 'ComplexArray',
    'Float1dArray', 'Float2dArray', 'Float3dArray', 'Float2dArray32', 'AnyFloatArray',
    'AnyNdArray', 'Any1dArray', 'Any2dArray', 'Any3dArray', 'ShapeT',
    'TruePeakDtype', 'TruePeakArray', 'NumChannelsT',
)

NumChannels = Literal[1, 2, 3, 5]
""""""
NumChannelsT = TypeVar('NumChannelsT', bound=NumChannels)
""""""

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

class TruePeakDtype(np.void, Generic[NumChannelsT]):
    """A :term:`structured data type` for true peak results

    .. attribute:: t
        :type: numpy.float64

        The time in seconds for each measurement

    .. attribute:: tp
        :type: tuple[numpy.float64, ...]

        The :term:`True Peak` values per channel at time :attr:`t` as a :term:`subarray`
        of length :obj:`NumChannelsT`

    """


_AnyDtype: TypeAlias = np.dtype[Any]
DType_co = TypeVar("DType_co", bound=np.dtype[Any], covariant=True)
""""""

_1D: TypeAlias = tuple[int]
_2D: TypeAlias = tuple[int, int]
_3D: TypeAlias = tuple[int, int, int]

ShapeT = TypeVar('ShapeT', _1D, _2D, _3D)
ShapeT_co = TypeVar('ShapeT_co', bound=tuple[int,...], covariant=True)

_1DArray = np.ndarray[_1D, DType_co]
_2DArray = np.ndarray[_2D, DType_co]
_3DArray = np.ndarray[_3D, DType_co]


# type AnyNdArray[_St: (tuple[int,...]), _Dt: (_AnyDtype)] = np.ndarray[_St, _Dt]
AnyNdArray: TypeAlias = np.ndarray[ShapeT, DType_co]
"""A generic type for numpy ND arrays"""
# type Any1dArray[_Dt: (_AnyDtype)] = _1DArray[_Dt]
Any1dArray: TypeAlias = _1DArray[DType_co]
"""A generic type for numpy 1D arrays"""
# type Any2dArray[_Dt: (_AnyDtype)] = _2DArray[_Dt]
Any2dArray: TypeAlias = _2DArray[DType_co]
"""A generic type for numpy 2D arrays"""
# type Any3dArray[_Dt: (_AnyDtype)] = _3DArray[_Dt]
Any3dArray: TypeAlias = _3DArray[DType_co]
"""A generic type for numpy 3D arrays"""

AnyFloatArray = AnyNdArray[ShapeT, np.dtype[Floating]]

AnyArray = npt.NDArray[Any]
BoolArray = AnyNdArray[ShapeT, np.dtype[np.bool_]]
IndexArray = AnyNdArray[ShapeT, np.dtype[np.intp]]
FloatArray = AnyNdArray[ShapeT, np.dtype[np.float64]]
ComplexArray = AnyNdArray[ShapeT, np.dtype[Complex]]
Float1dArray = Any1dArray[np.dtype[np.float64]]
"""1D array of :class:`~numpy.float64`"""
Float2dArray = Any2dArray[np.dtype[np.float64]]
"""2D array of :class:`~numpy.float64`"""
Float3dArray = Any3dArray[np.dtype[np.float64]]
"""3D array of :class:`~numpy.float64`"""

Float2dArray32 = Any2dArray[np.dtype[np.float32]]
"""2D array of :class:`~numpy.float32`"""

SosCoeff = np.ndarray[tuple[int, Literal[6]], np.dtype[np.float64]]
"""Array representing second-order sections filter coefficients"""

SosZI = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.float64]]
"""Array representing initial conditions for second-order sections filtering"""


class MeterArray(npt.NDArray[np.void]):
    """Array with dtype :obj:`MeterDtype`
    """
    pass


class TruePeakArray(npt.NDArray[np.void], Generic[NumChannelsT]):
    """Array with dtype :obj:`TruePeakDtype`
    """
    pass
