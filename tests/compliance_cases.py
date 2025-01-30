from __future__ import annotations

from typing import NamedTuple, Literal
from pathlib import Path
import itertools
import numpy as np
from scipy import signal
from scipy.io import wavfile

from lupy.types import FloatArray

HERE = Path(__file__).parent
DATA = HERE / 'data'
EBU_ROOT = DATA / 'ebu-loudness-test-setv05'

nan = np.nan


def gen_1k_sine(
    count: int,
    sample_rate: int,
    amp: float = 1,
    fc: float = 997,
    phase_deg: float = 0
):
    t = np.arange(count) / sample_rate
    ph = np.deg2rad(phase_deg)
    return amp * np.sin(2 * np.pi * fc * t + ph)



class ComplianceInput(NamedTuple):
    dBFS: tuple[float, float, float, float, float]
    duration: float
    fc: float = 997 / 48000
    phase: tuple[float, float, float, float, float]|None = None
    taper_dur: float|None = None

    def generate(self, sample_rate: int) -> FloatArray:
        N = int(round(sample_rate * self.duration))
        samples = np.zeros((5, N), dtype=np.float64)
        fc = self.fc * sample_rate
        sig = gen_1k_sine(N, sample_rate, 1, fc=fc)
        taper_len, taper_win = None, None
        if self.taper_dur is not None:
            taper_len = int(self.taper_dur * sample_rate)
            taper_win = signal.windows.hann(taper_len * 2).reshape((2, taper_len))

        for ch, sig_dB in enumerate(self.dBFS):
            if np.isnan(sig_dB):
                continue
            amp = 10 ** (sig_dB / 20)
            if self.phase is not None and not np.isnan(self.phase[ch]):
                _sig = gen_1k_sine(N, sample_rate, amp, fc, self.phase[ch])
            else:
                _sig = sig * amp
            if taper_win is not None:
                taper_len = taper_win.shape[1]
                _sig[:taper_len] *= taper_win[0]
                _sig[-taper_len:] *= taper_win[1]
            samples[ch,...] = _sig
        return samples

class ComplianceSource(NamedTuple):
    filename: Path
    bit_depth: Literal[16, 24, 32]
    is_float: bool = False

    def generate(self, sample_rate: int) -> FloatArray:
        if self.filename.suffix == '.npz':
            with np.load(self.filename) as data:
                samples = data['samples']
                _fs_arr = data['sample_rate']
                fs = _fs_arr[0]
        else:
            fs, samples = wavfile.read(self.filename)
        samples = np.asarray(samples, dtype=np.float64)
        if self.is_float:
            assert self.bit_depth == 32
            samples = samples
        elif self.bit_depth == 24:
            # 24-bit PCM is stored in the MSB of int32
            samples /= 1 << 31
        else:
            samples /= 1 << (self.bit_depth-1)

        if fs != sample_rate:
            samples = signal.resample_poly(samples, sample_rate, fs)

        # Temp array to match channels to their expected indices
        _samples = np.zeros((samples.shape[0], 5), dtype=np.float64)
        if samples.shape[1] == 2:
            _samples[:,0] = samples[:,0]
            _samples[:,2] = samples[:,1]
        else:
            _samples[:,0] = samples[:,0]
            _samples[:,1] = samples[:,2]
            _samples[:,2] = samples[:,1]
            if samples.shape[1] > 3:
                _samples[:,3:] = samples[:,3:]

        # Swap from ``(n_samp, n_chan)`` to ``(n_chan, n_samp)``
        return np.swapaxes(_samples, 0, 1)


class ComplianceResult(NamedTuple):
    momentary: tuple[float, float, float]|None          # (LUFS, LU, Tolerance)
    short_term: tuple[float, float, float]|None         # (LUFS, LU, Tolerance)
    integrated: tuple[float, float, float]|None         # (LUFS, LU, Tolerance)
    lra: tuple[float, float]|None                       # (LRA, Tolerance)
    true_peak: tuple[float, float, float]|None = None   # (TP, NegTol, PosTol)


class ComplianceBase(NamedTuple):
    input: list[ComplianceInput|ComplianceSource]
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
    Tech3341Compliance(
        name='case15',
        input=[
            ComplianceInput(
                dBFS=(20*np.log10(.5), nan, 20*np.log10(.5), nan, nan),
                duration=1,
                fc=0.25,
                taper_dur=.1,
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(-6, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case16',
        input=[
            ComplianceInput(
                dBFS=(20*np.log10(.5), nan, 20*np.log10(.5), nan, nan),
                duration=1,
                fc=0.25,
                taper_dur=.1,
                phase=(0, nan, 45, nan, nan),
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(-6, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case17',
        input=[
            ComplianceInput(
                dBFS=(20*np.log10(.5), nan, 20*np.log10(.5), nan, nan),
                duration=1,
                fc=0.25,
                taper_dur=.1,
                phase=(0, nan, 60, nan, nan),
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(-6, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case18',
        input=[
            ComplianceInput(
                dBFS=(20*np.log10(.5), nan, 20*np.log10(.5), nan, nan),
                duration=1,
                fc=0.25,
                taper_dur=.1,
                phase=(0, nan, 67.5, nan, nan),
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(-6, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case19',
        input=[
            ComplianceInput(
                dBFS=(20*np.log10(1.41), nan, 20*np.log10(1.41), nan, nan),
                duration=1,
                fc=0.25,
                taper_dur=.1,
                phase=(0, nan, 45, nan, nan),
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(3, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case20',
        input=[
            ComplianceSource(
                filename=EBU_ROOT / 'seq-3341-20-24bit.wav.npz',
                bit_depth=24,
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(0, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case21',
        input=[
            ComplianceSource(
                filename=EBU_ROOT / 'seq-3341-21-24bit.wav.npz',
                bit_depth=24,
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(0, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case22',
        input=[
            ComplianceSource(
                filename=EBU_ROOT / 'seq-3341-22-24bit.wav.npz',
                bit_depth=24,
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(0, .4, .2),
        ),
    ),
    Tech3341Compliance(
        name='case23',
        input=[
            ComplianceSource(
                filename=EBU_ROOT / 'seq-3341-23-24bit.wav.npz',
                bit_depth=24,
            ),
        ],
        result=ComplianceResult(
            momentary=None,
            short_term=None,
            integrated=None,
            lra=None,
            true_peak=(0, .4, .2),
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
