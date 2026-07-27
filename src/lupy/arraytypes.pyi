# mypy: disable-error-code="override"

from typing import Any, Generic, Literal, Self, TypeAlias, overload

import numpy as np
import numpy.typing as npt

from lupy.types import Float1dArray, NumChannelsT

__all__ = (
    'MeterArray',
    'MeterDtype',
    'TruePeakArray',
    'TruePeakDtype',
)

MeterDtype = np.dtype([
    ('t', np.float64),
    ('m', np.float64),
    ('s', np.float64),
])

class TruePeakDtype(np.void, Generic[NumChannelsT]): ... # type: ignore[misc]


_MeterArrayFields: TypeAlias = Literal['t', 'm', 's']


class MeterArray(np.ndarray[tuple[int], np.dtype[np.void]]):
    @overload
    def __getitem__(self, key: int|slice[Any, Any, Any]) -> Self: ...
    @overload
    def __getitem__(self, key: _MeterArrayFields) -> Float1dArray: ...

    def view(self, dtype: np.dtype|type[npt.NDArray[Any]]) -> Self: ...


class TruePeakArray(np.ndarray[tuple[int], np.dtype[np.void]], Generic[NumChannelsT]):
    @overload
    def __getitem__(self, key: int|slice[Any, Any, Any]) -> Self: ...
    @overload
    def __getitem__(self, key: Literal['t']) -> Float1dArray: ...
    @overload
    def __getitem__(self, key: Literal['tp']) -> np.ndarray[tuple[int, NumChannelsT], np.dtype[np.float64]]: ...

    def view(self, dtype: np.dtype|type[npt.NDArray[Any]]) -> Self: ...
