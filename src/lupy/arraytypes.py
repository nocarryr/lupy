from __future__ import annotations

from typing import Generic

import numpy as np
import numpy.typing as npt


from .types import NumChannelsT

__all__ = (
    'MeterDtype', 'MeterArray', 'TruePeakDtype', 'TruePeakArray',
)


MeterDtype = np.dtype([
    ('t', np.float64),
    ('m', np.float64),
    ('s', np.float64),
])
"""A :term:`structured data type` for loudness results

.. attribute:: t
    :type: numpy.float64

    The time in seconds for each measurement

.. attribute:: m
    :type: numpy.float64

    The :term:`Momentary Loudness` at time :attr:`t`

.. attribute:: s
    :type: numpy.float64

    The :term:`Short-Term Loudness` at time :attr:`t`

"""

class TruePeakDtype(np.void, Generic[NumChannelsT]):
    """A :term:`structured data type` for true peak results

    .. attribute:: t
        :type: numpy.float64

        The time in seconds for each measurement

    .. attribute:: tp
        :type: tuple[numpy.float64, ...]

        The :term:`True Peak` values per channel at time :attr:`t` as a :term:`subarray`
        of length :obj:`NumChannelsT`

    """


class MeterArray(npt.NDArray[np.void]):
    """Array with dtype :obj:`MeterDtype`
    """
    pass


class TruePeakArray(npt.NDArray[np.void], Generic[NumChannelsT]):
    """Array with dtype :obj:`TruePeakDtype`
    """
    pass
