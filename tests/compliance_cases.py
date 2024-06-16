from typing import NamedTuple, Literal
import itertools
import numpy as np

from lupy.types import FloatArray


nan = np.nan


def gen_1k_sine(count: int, sample_rate: int, amp: float = 1):
    fc = 997
    t = np.arange(count) / sample_rate
    return amp * np.sin(2 * np.pi * fc * t)



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

all_cases = {
    f'{v.__class__.__name__}.{k}':v for k,v in itertools.chain(*[
        d.items() for d in cases_by_name.values()
    ]
)}
