from __future__ import annotations

from typing import Callable
import numpy as np
import pytest

from lupy.types import FloatArray
from compliance_cases import ComplianceBase, cases_by_name, all_cases


nan = np.nan
rng = np.random.default_rng()


@pytest.fixture
def random_samples():
    def gen(*shape: int):
        n = 1
        for s in shape:
            n *= s
        return rng.random(n).reshape(shape)
    return gen

@pytest.fixture
def inc_samples():
    def gen(count: int, shape: tuple[int,...]|None = None):
        a = np.arange(count)
        if shape is not None:
            a = np.resize(a, shape)
        return a
    return gen

@pytest.fixture
def lkfs_1k_sine() -> Callable[[int, int, float], FloatArray]:
    # def gen(count: int, sample_rate: int, amp: float = 1):
    #     fc = 997
    #     t = np.arange(count) / sample_rate
    #     return amp * np.sin(2 * np.pi * fc * t)
    return gen_1k_sine

def gen_1k_sine(count: int, sample_rate: int, amp: float = 1):
    fc = 997
    t = np.arange(count) / sample_rate
    return amp * np.sin(2 * np.pi * fc * t)

@pytest.fixture(params=[512, 1024, 128, 256])
def block_size(request) -> int:
    return request.param

@pytest.fixture(params=[1, 2, 3, 5])
def num_channels(request) -> int:
    return request.param

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
def front_channels(request) -> tuple[int, int]:
    return request.param

@pytest.fixture(params=[
    (1, 0),
    (2, 0), (2, 1),
    (3, 0), (3, 1), (3, 2),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
])
def all_channels(request) -> tuple[int, int]:
    return request.param

@pytest.fixture(
    params=cases_by_name['3341'].values(),
    ids=list(cases_by_name['3341'].keys())
)
def tech_3341_compliance_case(request) -> ComplianceBase:
    return request.param

@pytest.fixture(
    params=cases_by_name['3342'].values(),
    ids=list(cases_by_name['3342'].keys())
)
def tech_3342_compliance_case(request) -> ComplianceBase:
    return request.param

@pytest.fixture(params=all_cases.values(), ids=list(all_cases.keys()))
def compliance_case(request) -> ComplianceBase:
    return request.param


@pytest.fixture(params=[48000])
def sample_rate(request) -> int:
    return request.param
