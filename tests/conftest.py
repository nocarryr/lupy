from __future__ import annotations

import os
from fractions import Fraction
from typing import Protocol, overload

import numpy as np
import pytest
from compliance_cases import (
    ComplianceBase,
    all_cases,
    bs_2217_compliance_cases,
    cases_by_name,
    true_peak_cases,
)

from lupy.types import ChannelIndex, FloatArray, NumChannels, ShapeT
from lupy.typeutils import ensure_array_of_shape

IS_CI = 'CI' in os.environ

nan = np.nan
rng = np.random.default_rng(seed=42)



class RandomSamplesCallable(Protocol):
    def __call__(self, shape: ShapeT) -> FloatArray[ShapeT]: ...


class IncSamplesCallable(Protocol):
    def __call__(self, count: int, shape: ShapeT) -> FloatArray[ShapeT]: ...


@pytest.fixture
def random_samples() -> RandomSamplesCallable:
    def gen(shape: ShapeT) -> FloatArray[ShapeT]:
        n = 1
        for s in shape:
            n *= s
        r = rng.random(n).reshape(shape)
        return ensure_array_of_shape(r, shape)
    return gen

@pytest.fixture
def inc_samples() -> IncSamplesCallable:
    def gen(count: int, shape: ShapeT) -> FloatArray[ShapeT]:
        a = np.arange(count, dtype=np.float64)
        a = np.resize(a, shape)
        return ensure_array_of_shape(a, shape)
    return gen


@overload
def gen_1k_sine(count: int, sample_rate: int, amp: float) -> FloatArray[tuple[int]]: ...
@overload
def gen_1k_sine(count: int, sample_rate: int, amp: FloatArray[ShapeT]) -> FloatArray[ShapeT]: ...
def gen_1k_sine(
    count: int,
    sample_rate: int,
    amp: float|FloatArray[ShapeT] = 1
) -> FloatArray[ShapeT] | FloatArray[tuple[int]]:
    fc = 997
    t = np.arange(count) / sample_rate
    return amp * np.sin(2 * np.pi * fc * t)

@pytest.fixture(params=[512, 1024, 128, 256])
def block_size(request: pytest.FixtureRequest) -> int:
    param: int = request.param
    return param

@pytest.fixture(params=[1, 2, 3, 5])
def num_channels(request: pytest.FixtureRequest) -> NumChannels:
    param: NumChannels = request.param
    return param

@pytest.fixture(params=[
    Fraction(1, 10),
    Fraction(2, 10),
    Fraction(4, 10),
    Fraction(8, 10),
])
def true_peak_gate_duration(request: pytest.FixtureRequest) -> Fraction:
    param: Fraction = request.param
    return param

# @pytest.fixture(params=[0, 1, 2])
# def front_channel(request, num_channels) -> int:
#     ch: int = request.param
#     if ch > num_channels-1:
#         ch = num_channels - 1
#     return ch

@pytest.fixture(params=[
    (1, 0),
    (2, 0), (2, 1),
    (3, 0), (3, 1), (3, 2),
    (5, 0), (5, 1), (5, 2),
])
def front_channels(request: pytest.FixtureRequest) -> tuple[NumChannels, ChannelIndex]:
    param: tuple[NumChannels, ChannelIndex] = request.param
    return param

@pytest.fixture(params=[
    (1, 0),
    (2, 0), (2, 1),
    (3, 0), (3, 1), (3, 2),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
])
def all_channels(request: pytest.FixtureRequest) -> tuple[NumChannels, ChannelIndex]:
    param: tuple[NumChannels, ChannelIndex] = request.param
    return param

@pytest.fixture(
    params=cases_by_name['3341'].values(),
    ids=list(cases_by_name['3341'].keys())
)
def tech_3341_compliance_case(request: pytest.FixtureRequest) -> ComplianceBase:
    param: ComplianceBase = request.param
    return param

@pytest.fixture(
    params=cases_by_name['3342'].values(),
    ids=list(cases_by_name['3342'].keys())
)
def tech_3342_compliance_case(request: pytest.FixtureRequest) -> ComplianceBase:
    param: ComplianceBase = request.param
    return param


@pytest.fixture(
    params=bs_2217_compliance_cases.values(),
    ids=list(bs_2217_compliance_cases.keys())
)
def bs_2217_compliance_case(request: pytest.FixtureRequest) -> ComplianceBase:
    param: ComplianceBase = request.param
    return param


@pytest.fixture(params=all_cases.values(), ids=list(all_cases.keys()))
def compliance_case(request: pytest.FixtureRequest) -> ComplianceBase:
    param: ComplianceBase = request.param
    return param


@pytest.fixture(params=true_peak_cases.values(), ids=list(true_peak_cases.keys()))
def true_peak_compliance_case(request: pytest.FixtureRequest) -> ComplianceBase:
    param: ComplianceBase = request.param
    return param


@pytest.fixture(params=[48000, 44100, 88200, 96000])
def sample_rate(request: pytest.FixtureRequest) -> int:
    param: int = request.param
    return param
