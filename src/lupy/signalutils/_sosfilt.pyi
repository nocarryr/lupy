from __future__ import annotations

import numpy as np

from ..types import SosZI, SosCoeff

def _sosfilt(sos: SosCoeff, x: np.ndarray[tuple[int, int], np.dtype[np.floating]], zi: SosZI) -> None:
    ...
