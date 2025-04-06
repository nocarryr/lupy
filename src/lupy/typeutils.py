"""
Utility functions for type checking array dimensions and types.

These functions are intended to be used as type guards for static type checking
with runtime assertions.
"""
from __future__ import annotations

from typing import Literal, Any, overload
import sys
if sys.version_info < (3, 11):
    from typing_extensions import TypeIs
else:
    from typing import TypeIs

import numpy as np

from .types import (
    AnyArray, AnyNdArray, Any1dArray, Any2dArray, Any3dArray, AnyFloatArray,
    IndexArray, BoolArray, ComplexArray, MeterDtype, MeterArray,
    ShapeT, DType_co,
)


def is_1d_array(arr: AnyNdArray[Any, DType_co]) -> TypeIs[Any1dArray[DType_co]]:
    """Check if the given array is a 1-dimensional array
    """
    return is_nd_array(arr, 1)

def is_2d_array(arr: AnyNdArray[Any, DType_co]) -> TypeIs[Any2dArray[DType_co]]:
    """Check if the given array is a 2-dimensional array
    """
    return is_nd_array(arr, 2)

def is_3d_array(arr: AnyNdArray[Any, DType_co]) -> TypeIs[Any3dArray[DType_co]]:
    """Check if the given array is a 3-dimensional array
    """
    return is_nd_array(arr, 3)


@overload
def is_nd_array(arr: AnyArray, ndim: Literal[1]) -> TypeIs[Any1dArray]: ...
@overload
def is_nd_array(arr: AnyArray, ndim: Literal[2]) -> TypeIs[Any2dArray]: ...
@overload
def is_nd_array(arr: AnyArray, ndim: Literal[3]) -> TypeIs[Any3dArray]: ...
@overload
def is_nd_array(arr: AnyArray, ndim: int) -> TypeIs[AnyNdArray[Any, DType_co]]: ...
def is_nd_array(arr: AnyArray, ndim: int) -> TypeIs[AnyNdArray[Any, DType_co]]:
    """Check if the given array shape matches the specified number of dimensions
    """
    return arr.ndim == ndim


def ensure_1d_array(arr: AnyNdArray[Any, DType_co]) -> Any1dArray[DType_co]:
    """Ensure the given array is 1-dimensional and return it
    """
    assert is_1d_array(arr)
    return arr

def ensure_2d_array(arr: AnyNdArray[Any, DType_co]) -> Any2dArray[DType_co]:
    """Ensure the given array is 2-dimensional and return it
    """
    assert is_2d_array(arr)
    return arr

def ensure_3d_array(arr: AnyNdArray[Any, DType_co]) -> Any3dArray[DType_co]:
    """Ensure the given array is 3-dimensional and return it
    """
    assert is_3d_array(arr)
    return arr


@overload
def ensure_nd_array(arr: AnyNdArray[Any, DType_co], ndim: Literal[1]) -> Any1dArray[DType_co]: ...
@overload
def ensure_nd_array(arr: AnyNdArray[Any, DType_co], ndim: Literal[2]) -> Any2dArray[DType_co]: ...
@overload
def ensure_nd_array(arr: AnyNdArray[Any, DType_co], ndim: Literal[3]) -> Any3dArray[DType_co]: ...
@overload
def ensure_nd_array(arr: AnyNdArray[Any, DType_co], ndim: int) -> AnyNdArray[Any, DType_co]: ...
def ensure_nd_array(arr: AnyNdArray[Any, DType_co], ndim: int) -> AnyNdArray[Any, DType_co]:
    """Ensure the given array has the specified number of dimensions and return it
    """
    assert arr.ndim == ndim
    return arr



def is_float_array(arr: AnyNdArray[ShapeT, Any]) -> TypeIs[AnyFloatArray[ShapeT]]:
    """Check if the given array's dtype is a floating-point type
    """
    return np.issubdtype(arr.dtype, np.floating)

def is_index_array(arr: AnyNdArray[ShapeT, Any]) -> TypeIs[IndexArray[ShapeT]]:
    """Check if the given array's dtype is an integer type suitable for indexing
    """
    return arr.dtype == np.intp

def is_bool_array(arr: AnyNdArray[ShapeT, Any]) -> TypeIs[BoolArray[ShapeT]]:
    """Check if the given array's dtype is boolean
    """
    return arr.dtype == np.bool_

def is_complex_array(arr: AnyNdArray[ShapeT, Any]) -> TypeIs[ComplexArray[ShapeT]]:
    """Check if the given array's dtype is a complex floating-point type
    """
    return np.issubdtype(arr.dtype, np.complexfloating)

def is_meter_array(arr: AnyArray) -> TypeIs[MeterArray]:
    """Check if the given array is a :class:`~.types.MeterArray`
    """
    return isinstance(arr, np.ndarray) and arr.dtype == MeterDtype

def ensure_meter_array(arr: AnyArray) -> MeterArray:
    """Ensure the given array is a :class:`~.types.MeterArray` and return it
    """
    assert is_meter_array(arr)
    return arr
