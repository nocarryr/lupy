from __future__ import annotations

from typing import Callable, NamedTuple, ClassVar, Literal
import numpy as np
from scipy import signal
import pytest

from lupy.types import FloatArray


nan = np.nan
rng = np.random.default_rng()

class ComplianceInput(NamedTuple):
    dBFS: tuple[float, float, float, float, float]
    duration: float

    def generate(self, sample_rate: int) -> FloatArray:
        N = int(round(sample_rate * self.duration))
        samples = np.zeros((5, N), dtype=np.float64)
        sig = gen_1k_sine(N, sample_rate, 1)
        for ch, sig_dB in enumerate(self.dBFS):
            if np.isnan(sig_dB):
                continue
            amp = 10 ** (sig_dB / 20)
            _sig = sig * amp
            samples[ch,...] = _sig
        return samples


class ComplianceResult(NamedTuple):
    momentary: tuple[float, float, float]|None  # (LUFS, LU, Tolerance)
    short_term: tuple[float, float, float]|None # (LUFS, LU, Tolerance)
    integrated: tuple[float, float, float]|None # (LUFS, LU, Tolerance)
    lra: tuple[float, float]|None               # (LRA, Tolerance)

class ComplianceBase(NamedTuple):
    input: list[ComplianceInput]
    result: ComplianceResult
    # target_lu: ClassVar = -23
    name: str

    def generate_samples(self, sample_rate: int) -> FloatArray:
        inputs = [inp.generate(sample_rate) for inp in self.input]
        N = sum([inp.shape[1] for inp in inputs])
        result = np.zeros((5, N), dtype=np.float64)
        ix = 0
        for inp in inputs:
            assert np.all(np.equal(result[:,ix:ix+inp.shape[1]], 0))
            result[:,ix:ix+inp.shape[1]] = inp
            ix += inp.shape[1]
        return result


class Tech3341Compliance(ComplianceBase):
    pass

class Tech3342Compliance(ComplianceBase):
    pass


_tech_3341_compliance_cases: list[ComplianceBase] = [
    # Case 1
    Tech3341Compliance(
        name='case1',
        input=[
            ComplianceInput(dBFS=(-23, nan, -23, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=(-23, 0, .1),
            short_term=(-23, 0, .1),
            integrated=(-23, 0., .1),
            lra=None,
        ),
    ),
    # Case 2
    Tech3341Compliance(
        name='case2',
        input=[
            ComplianceInput(dBFS=(-33, nan, -33, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=(-33, 0, .1),
            short_term=(-33, 0, .1),
            integrated=(-33, 0., .1),
            lra=None,
        ),
    ),
    # Case 3
    Tech3341Compliance(
        name='case3',
        input=[
            ComplianceInput(dBFS=(-36, nan, -36, nan, nan), duration=10),
            ComplianceInput(dBFS=(-23, nan, -23, nan, nan), duration=60),
            ComplianceInput(dBFS=(-36, nan, -36, nan, nan), duration=10),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=(-23, 0, .1),
            lra=None,
        ),
    ),
    # Case 4
    Tech3341Compliance(
        name='case4',
        input=[
            ComplianceInput(dBFS=(-72, nan, -72, nan, nan), duration=10),
            ComplianceInput(dBFS=(-36, nan, -36, nan, nan), duration=10),
            ComplianceInput(dBFS=(-23, nan, -23, nan, nan), duration=60),
            ComplianceInput(dBFS=(-36, nan, -36, nan, nan), duration=10),
            ComplianceInput(dBFS=(-72, nan, -72, nan, nan), duration=10),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=(-23, 0, .1),
            lra=None,
        )
    ),
    # Case 5
    Tech3341Compliance(
        name='case5',
        input=[
            ComplianceInput(dBFS=(-26, nan, -26, nan, nan), duration=20),
            ComplianceInput(dBFS=(-20, nan, -20, nan, nan), duration=20.1),
            ComplianceInput(dBFS=(-26, nan, -26, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=(-23, 0, .1),
            lra=None,
        ),
    ),
    # Case 6
    Tech3341Compliance(
        name='case6',
        input=[
            ComplianceInput(dBFS=(-28, -24, -28, -30, -30), duration=20),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=(-23, 0, .1),
            lra=None,
        ),
    ),
    Tech3341Compliance(
        name='case9',
        input=[
            ComplianceInput(dBFS=(-20, nan, -20, nan, nan), duration=1.34),
            ComplianceInput(dBFS=(-30, nan, -30, nan, nan), duration=1.66),
        ] * 5,
        result=ComplianceResult(
            momentary=None,
            short_term=(-23, 0, .1),
            integrated=None,
            lra=None,
        ),
    ),
]

_tech_3342_compliance_cases = [
    Tech3342Compliance(
        name='case1',
        input=[
            ComplianceInput(dBFS=(-20, nan, -20, nan, nan), duration=20),
            ComplianceInput(dBFS=(-30, nan, -30, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=None, short_term=None, integrated=None,
            lra=(10, 1),
        ),
    ),
    Tech3342Compliance(
        name='case2',
        input=[
            ComplianceInput(dBFS=(-20, nan, -20, nan, nan), duration=20),
            ComplianceInput(dBFS=(-15, nan, -15, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=None, short_term=None, integrated=None,
            lra=(5, 1),
        ),
    ),
    Tech3342Compliance(
        name='case3',
        input=[
            ComplianceInput(dBFS=(-40, nan, -40, nan, nan), duration=20),
            ComplianceInput(dBFS=(-20, nan, -20, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=None, short_term=None, integrated=None,
            lra=(20, 1),
        ),
    ),
    Tech3342Compliance(
        name='case4',
        input=[
            ComplianceInput(dBFS=(-50, nan, -50, nan, nan), duration=20),
            ComplianceInput(dBFS=(-35, nan, -35, nan, nan), duration=20),
            ComplianceInput(dBFS=(-20, nan, -20, nan, nan), duration=20),
            ComplianceInput(dBFS=(-35, nan, -35, nan, nan), duration=20),
            ComplianceInput(dBFS=(-50, nan, -50, nan, nan), duration=20),
        ],
        result=ComplianceResult(
            momentary=None, short_term=None, integrated=None,
            lra=(15, 1),
        ),
    ),
]

cases_by_name: dict[Literal['3341', '3342'], dict[str, ComplianceBase]] = {
    '3341':{c.name: c for c in _tech_3341_compliance_cases},
    '3342':{c.name: c for c in _tech_3342_compliance_cases},
}


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

@pytest.fixture(params=cases_by_name['3341'].values())
def tech_3341_compliance_case(request) -> ComplianceBase:
    return request.param

@pytest.fixture(params=cases_by_name['3342'].values())
def tech_3342_compliance_case(request) -> ComplianceBase:
    return request.param

@pytest.fixture(params=_tech_3341_compliance_cases + _tech_3342_compliance_cases)
def compliance_case(request) -> ComplianceBase:
    return request.param


@pytest.fixture(params=[48000])
def sample_rate(request) -> int:
    return request.param
