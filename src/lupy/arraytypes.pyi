# mypy: disable-error-code="override"
from __future__ import annotations

from typing import Generic, Literal, Any, Self, overload

import numpy as np
import numpy.typing as npt

from lupy.types import Float1dArray, Float2dArray, NumChannelsT


__all__ = (
    'MeterDtype', 'MeterArray', 'TruePeakDtype', 'TruePeakArray',
)

MeterDtype = np.dtype([
    ('t', np.float64),
    ('m', np.float64),
    ('s', np.float64),
])

class TruePeakDtype(np.void, Generic[NumChannelsT]): ... # type: ignore[misc]


_MeterArrayFields = Literal['t', 'm', 's']


class MeterArray(npt.NDArray[np.void]):
    @overload
    def __getitem__(self, key: int|slice[Any, Any, Any]) -> Self: ...
    @overload
    def __getitem__(self, key: _MeterArrayFields) -> Float1dArray: ...

    def view(self, dtype: np.dtype|type[npt.NDArray[Any]]) -> Self: ...


_TruePeakArrayFields = Literal['t', 'tp']


class TruePeakArray(npt.NDArray[np.void], Generic[NumChannelsT]):
    @overload
    def __getitem__(self, key: int|slice[Any, Any, Any]) -> Self: ...
    @overload
    def __getitem__(self, key: Literal['t']) -> Float1dArray: ...
    @overload
    def __getitem__(self, key: Literal['tp']) -> np.ndarray[tuple[int, NumChannelsT], np.dtype[np.float64]]: ...

    def view(self, dtype: np.dtype|type[npt.NDArray[Any]]) -> Self: ...
