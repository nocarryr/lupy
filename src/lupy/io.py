from __future__ import annotations
from typing import (
    TypeVar, NewType, Any, Mapping, TypedDict, NamedTuple, Generic, Iterator, ClassVar,
    get_type_hints, is_typeddict, Protocol
)
import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .types import Float1dArray, Float2dArray, Any1dArray, AnyNdArray, NumChannels, NumChannelsT
from .typeutils import is_1d_array, ensure_meter_array, MeterArray, MeterDtype

ObjectType = NewType('ObjectType', str)

_DType_co = TypeVar("_DType_co", bound=np.dtype[Any], covariant=True)
MType = TypeVar('MType', bound=Mapping[str, Any])
AType = TypeVar('AType', bound=Mapping[str, npt.NDArray[Any]])

_T_co = TypeVar("_T_co", covariant=True)
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, default=Any, covariant=True)
# _DType = TypeVar("_DType", bound=np.dtype[Any])
# _DType_co = TypeVar("_DType_co", bound=np.dtype[Any], covariant=True)



# ArrayMap = Mapping[str, npt.NDArray[_ScalarT_co]]
# MetaDataMap = Mapping[str, _T_co]

type DType_co = np.dtype[np.generic]
# type ArrayT_co[_T_co: np.generic] = npt.NDArray[_T_co]
# type ArrayT_co[_T_co: (np.dtype[Any])] = np.ndarray[tuple[int, ...], _T_co]
type ArrayT_co[T: (DType_co)] = np.ndarray[tuple[int, ...], T]
type ArrayMap[V: (ArrayT_co)] = Mapping[str, V]

type MetaDataMap[K: str, V] = Mapping[K, V]

# # class ArrayMapProtocol[T: (ArrayT_co)](Protocol):
# #     def __getitem__(self, key: str) -> T: ...
# #     def __setitem__(self, key: str, value: T) -> None: ...
# #     def __contains__(self, key: str) -> bool: ...
# #     def keys(self) -> Iterator[str]: ...
# #     def items(self) -> Iterator[tuple[str, T]]: ...
# #     def values(self) -> Iterator[T]: ...


# # a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
# # block_data = np.zeros(10, dtype=MeterDtype)
# # _block_data = ensure_meter_array(block_data)
# # assert is_1d_array(a)
# # foo: ArrayMap[Float1dArray] = {
# #     'a': a,
# #     'm': _block_data,
# # }

@dataclass
class ObjectData[#(Generic[MType, AType], ABC):

    # OT: ObjectType,
    # MType: ObjectMeta,
    # AType: ArrayMeta
    # MType: MetaDataMap,
    # AType: ArrayMap,
    # _Dt_co: (np.dtype[Any]),
    MType: Mapping[str, Any],
    # AType: Mapping[str, np.ndarray[tuple[int, ...], DType_co]],
    # AType: Mapping[str, np.ndarray]
    # AType: ArrayMapProtocol[ArrayT_co[DType_co]],
    AType: Mapping[str, object],
    # AType: Mapping[str, npt.NDArray[Any]],
    # AType: ArrayMap[np.ndarray[
    #     tuple[int, ...], DType_co
    # ]],

    # AK: str, AV: ArrayT_co,
    # MK: str, MV
](ABC):
    # object_type: OT
    metadata: MType
    arrays: AType
    # child_objects: ClassVar
    # parent_object_data: ObjectData|None = None

    # def __post_init__(self):
    #     for child in self.get_child_object_data(recursive=False):
    #         child.parent_object_data = self

    @property
    def prefix(self) -> str:
        p = self.get_type_name()
        # if self.parent_object_data is not None:
        #     p = f'{self.parent_object_data.prefix}.{p}'
        return p

    @classmethod
    def prefix_key(cls, subkey: str) -> str:
        return f'{cls.get_type_name()}.{subkey}'

    @classmethod
    def get_metadata_key(cls) -> str:
        return cls.prefix_key('_metadata_')

    @classmethod
    def unprefix_dict(cls, d: Mapping[str, npt.NDArray]) -> dict[str, npt.NDArray]:
        result = {}
        prefix = f'{cls.get_type_name()}.'
        prefix_len = len(prefix)
        for k, v in d.items():
            if not k.startswith(prefix):
                continue
            result[k[prefix_len:]] = v
        return result

    @classmethod
    def kwargs_from_array_dict(cls, d: Mapping[str, npt.NDArray]):
        # my_metadata_arr = d.pop(cls.get_metadata_key())
        my_arrays = cls.unprefix_dict(d)
        my_metadata = my_arrays.pop('_metadata_')
        meta_types = cls.get_concrete_meta()
        metadata = {k: my_metadata[k][0] for k, v in meta_types}
        for key, meta_type in cls.get_child_objects():
            child = meta_type.from_arrays(d)
            metadata[key] = child
        return {
            'metadata': metadata,
            'arrays': my_arrays,
        }

    @classmethod
    def from_arrays(cls, d: Mapping[str, npt.NDArray]) -> Self:
        kw = cls.kwargs_from_array_dict(d)
        return cls(**kw)

    def get_child_object_data(self, recursive: bool = False) -> Iterator[ObjectData]:
        for v in self.metadata.values():
            if isinstance(v, ObjectData):
                yield v
                if recursive:
                    yield from v.get_child_object_data(recursive=True)

    @classmethod
    @abstractmethod
    def _get_meta_type(cls) -> type[MType]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_type_name(cls) -> ObjectType:
        raise NotImplementedError

    @classmethod
    def get_concrete_meta(cls):
        for k, v in get_type_hints(cls._get_meta_type()).items():
            if isinstance(v, type) and issubclass(v, ObjectData):
                continue
            yield k, v

    @classmethod
    def get_child_objects(cls):
        for k, v in get_type_hints(cls._get_meta_type()).items():
            if isinstance(v, type) and issubclass(v, ObjectData):
                yield k, v

    @classmethod
    def get_dtype_for_meta(cls) -> np.dtype:
        """Get the dtype for the metadata of this object data"""
        meta_type = cls._get_meta_type()
        assert is_typeddict(meta_type)
        return np.dtype([(k, v) for k, v in cls.get_concrete_meta()])

    def meta_to_array(self) -> np.ndarray[tuple[int], np.dtype[Any]]:
        dtype = self.get_dtype_for_meta()
        keys = [k for k, v in self.get_concrete_meta()]
        arr = np.zeros(1, dtype=dtype)
        for k in keys:
            arr[k][0] = self.metadata[k]
        return arr

    def get_arrays(self, recursive: bool = True) -> dict[str, npt.NDArray]:
        arrays: dict[str, npt.NDArray] = {}
        arrays[self.get_metadata_key()] = self.meta_to_array()
        for k, v in self.arrays.items():
            assert isinstance(v, np.ndarray)
            arrays[self.prefix_key(k)] = v
        if recursive:
            for child in self.get_child_object_data(recursive=True):
                child_arrays = child.get_arrays(recursive=True)
                assert set(child_arrays.keys()) & set(arrays.keys()) == set()
                arrays.update(child_arrays)
        return arrays

    def save(self, filename: Path|str):
        """Save this object data to a file"""
        arrays = self.get_arrays(recursive=True)
        np.savez_compressed(filename, allow_pickle=True, **arrays)

    @classmethod
    def load(cls, filename: Path|str) -> Self:
        """Load an object data from a file"""
        # arrays = np.load(filename, allow_pickle=False)
        # return cls.from_arrays(arrays)
        with np.load(str(filename), allow_pickle=False) as data:
            arrays = {k:v for k, v in data.items()}
        return cls.from_arrays(arrays)



class ObjectMeta(TypedDict):
    # type: str
    pass




class BlockMeta(TypedDict, Generic[NumChannelsT]):
    sample_rate: int
    num_channels: NumChannelsT
    gate_size: int
    integrated_lkfs: float
    lra: float
    num_blocks: int
    block_index: int


class BlockArrays(TypedDict):
    block_data: MeterArray
    Zij: Float2dArray
    block_weighted_sums: Float1dArray
    quarter_block_weighted_sums: Float1dArray
    block_loudness: Float1dArray


# BlockObjectType = ObjectType('BlockProcessor')
# BlockData = ObjectData[BlockMeta, BlockArrays]
@dataclass
class BlockData(ObjectData[BlockMeta, BlockArrays]):

    @classmethod
    def _get_meta_type(cls) -> type[BlockMeta]:
        return BlockMeta

    @classmethod
    def get_type_name(cls) -> ObjectType:
        return ObjectType('BlockProcessor')

    @classmethod
    def get_concrete_meta(cls):
        for k, v in super().get_concrete_meta():
            if k == 'num_channels':
                yield k, int
            else:
                yield k, v


class TPMeta(ObjectMeta, Generic[NumChannelsT]):
    sample_rate: int
    num_channels: NumChannelsT
    max_peak: float
    gate_size: int

class TPArrays(TypedDict):
    current_peaks: Float1dArray

# TPObjectType = ObjectType('TruePeakProcessor')
# TPData = ObjectData[TPMeta, TPArrays]
@dataclass
class TPData(ObjectData[TPMeta, TPArrays]):

    @classmethod
    def _get_meta_type(cls) -> type[TPMeta]:
        return TPMeta

    @classmethod
    def get_type_name(cls) -> ObjectType:
        return ObjectType('TruePeakProcessor')

    @classmethod
    def get_concrete_meta(cls):
        for k, v in super().get_concrete_meta():
            if k == 'num_channels':
                yield k, int
            else:
                yield k, v



class MeterMeta(ObjectMeta, Generic[NumChannelsT]):
    sample_rate: int
    num_channels: NumChannelsT
    block_size: int
    block_processor: BlockData
    true_peak_processor: TPData

class MeterArrays(TypedDict):
    pass

# MeterData = ObjectData[MeterMeta, MeterArrays]
@dataclass
class MeterData(ObjectData[MeterMeta, MeterArrays]):
    # block_processor: BlockData
    # true_peak_processor: TPData

    @classmethod
    def _get_meta_type(cls) -> type[MeterMeta]:
        return MeterMeta

    @classmethod
    def get_type_name(cls) -> ObjectType:
        return ObjectType('MeterProcessor')

    @classmethod
    def get_concrete_meta(cls):
        for k, v in super().get_concrete_meta():
            if k == 'num_channels':
                yield k, int
            else:
                yield k, v

# class MeterArrays(TypedDict):


# def get_dtype_for_object_data[Mt: MetaDataMap, At: ArrayMap](obj_data: ObjectData[Mt, At], t: type[ObjectData[Mt, At]]):
#     meta = obj_data['metadata']
#     type_map = get_type_hints(t)
