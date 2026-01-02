from __future__ import annotations

from typing import TypeVar, TypeAlias, Literal, Any, Self, overload


import numpy as np
from numpy import dtype

import numpy.typing as npt

__all__ = (
    'Floating', 'Complex', 'MeterDtype', 'MeterArray', 'SosCoeff', 'SosZI',
    'AnyArray', 'BoolArray', 'IndexArray', 'FloatArray', 'ComplexArray',
    'Float1dArray', 'Float2dArray', 'Float3dArray', 'Float2dArray32', 'AnyFloatArray',
    'AnyNdArray', 'Any1dArray', 'Any2dArray', 'Any3dArray', 'ShapeT',
)


_AnyDtype: TypeAlias = dtype[Any]
DType_co = TypeVar("DType_co", bound=dtype[Any], covariant=True)


_1D: TypeAlias = tuple[int]
_2D: TypeAlias = tuple[int, int]
_3D: TypeAlias = tuple[int, int, int]

ShapeT = TypeVar('ShapeT', _1D, _2D, _3D)
ShapeT_co = TypeVar('ShapeT_co', bound=tuple[int,...], covariant=True)

_1DArray = np.ndarray[_1D, DType_co]
_2DArray = np.ndarray[_2D, DType_co]
_3DArray = np.ndarray[_3D, DType_co]

MeterDtype = np.dtype([
    ('t', np.float64),
    ('m', np.float64),
    ('s', np.float64),
])

Floating: TypeAlias = np.floating
Complex = np.complex128

AnyArray = npt.NDArray[Any]

type AnyNdArray[_St: (tuple[int,...]), _Dt: (_AnyDtype)] = np.ndarray[_St, _Dt]
type Any1dArray[_Dt: (_AnyDtype)] = _1DArray[_Dt]
type Any2dArray[_Dt: (_AnyDtype)] = _2DArray[_Dt]
type Any3dArray[_Dt: (_AnyDtype)] = _3DArray[_Dt]

BoolArray = AnyNdArray[ShapeT, np.dtype[np.bool_]]
IndexArray = AnyNdArray[ShapeT, np.dtype[np.intp]]
FloatArray = AnyNdArray[ShapeT, np.dtype[np.float64]]
ComplexArray = AnyNdArray[ShapeT, np.dtype[Complex]]

AnyFloatArray = AnyNdArray[ShapeT, np.dtype[Floating]]

Float1dArray = Any1dArray[np.dtype[np.float64]]
Float2dArray = Any2dArray[np.dtype[np.float64]]
Float3dArray = Any3dArray[np.dtype[np.float64]]

Float2dArray32 = Any2dArray[np.dtype[np.float32]]

SosCoeff = np.ndarray[tuple[int, Literal[6]], np.dtype[np.float64]]
SosZI = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.float64]]

_MeterArrayFields = Literal['t', 'm', 's']


class MeterArray(npt.NDArray[np.void]):
    @overload
    def __getitem__(self, key: int|slice[Any, Any, Any]) -> Self: ...
    @overload
    def __getitem__(self, key: _MeterArrayFields) -> Float1dArray: ...
    def __getitem__(self, key: int|slice[Any, Any, Any]|_MeterArrayFields) -> Float1dArray|Self: ...

    def view(self, dtype: np.dtype|type[npt.NDArray[Any]]) -> Self: ...
