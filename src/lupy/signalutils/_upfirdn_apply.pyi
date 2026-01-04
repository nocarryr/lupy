from __future__ import annotations
from typing import Literal

import numpy as np
import numpy.typing as npt

type DTYPE_t = np.floating|np.complexfloating
type IntLike = int|np.integer

ModeName = Literal[
    'constant',
    'symmetric',
    'constant_edge',
    'smooth',
    'periodic',
    'reflect',
    'antisymmetric',
    'antireflect',
    'line',
]


def _output_len(
    len_h: IntLike,
    in_len: IntLike,
    up: IntLike,
    down: IntLike
) -> IntLike:
    ...


def mode_enum(mode: ModeName) -> int:
    ...


def _pad_test(data: npt.NDArray[DTYPE_t], npre: IntLike = ..., npost: IntLike = ..., mode: ModeName = ...) -> npt.NDArray[DTYPE_t]:
    ...


def _apply(
    data: npt.NDArray[DTYPE_t],
    h_trans_flip: npt.NDArray[DTYPE_t],
    out: npt.NDArray[DTYPE_t],
    up: IntLike,
    down: IntLike,
    axis: IntLike,
    mode: int,
    cval: IntLike|DTYPE_t|float
) -> None:
    ...
