from __future__ import annotations
from typing import cast

import numpy as np
from scipy.signal._sosfilt import _sosfilt

from ..types import (
    Float2dArray,
    SosZI,
    SosCoeff,
)
from ..typeutils import ensure_2d_array




# Adapted from:
# https://github.com/scipy/scipy/blob/b9ae1430d33a36c2e95b51932963278c255c100d/scipy/signal/_filter_design.py#L823-L837
#
def validate_sos(sos: Float2dArray) -> SosCoeff:
    """Helper to validate a SOS input"""
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    _, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return cast(SosCoeff, sos)




# Adapted from:
# https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_signaltools.py#L4601-L4715
#
def sosfilt(sos: SosCoeff, x: Float2dArray, zi: SosZI, axis: int = -1) -> tuple[Float2dArray, SosZI]:
    """
    Filter data along one dimension using cascaded second-order sections.

    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    .. note::

        This is a stripped down version of :func:`scipy.signal.sosfilt` to reduce
        overhead in repeated calls.
        It removes input validation and assumes 2D input of type
        :obj:`~numpy.float64`.

        It also requires the initial conditions *zi* to be provided and always
        returns the final conditions *zf*.

    Parameters
    ----------
    sos : ndarray
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : ndarray
        A 2-dimensional input array of dtype :obj:`~numpy.float64`.
    zi : ndarray
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) array of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray
        The final filter delay values.

    """
    n_sections = sos.shape[0]
    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)

    dtype = np.float64
    assert x.ndim == 2
    assert x.dtype == sos.dtype == dtype


    # make a copy so that we can operate in place
    zi = np.array(zi, dtype) # type: ignore[arg-type]
    if zi.shape != x_zi_shape:
        raise ValueError('Invalid zi shape. With axis=%r, an input with '
                         'shape %r, and an sos array with %d sections, zi '
                         'must have shape %r, got %r.' %
                         (axis, x.shape, n_sections, x_zi_shape, zi.shape))

    axis = axis % x.ndim  # make positive

    # move section axis to front, axis to last-but-one
    zi = np.moveaxis(zi, [0, axis + 1], [-2, -1]) # type: ignore[arg-type]
    x_shape, zi_shape = x.shape, zi.shape

    x = ensure_2d_array(np.array(x, dtype, order='C'))  # make a copy, can modify in place

    assert zi.flags.c_contiguous

    _sosfilt(sos, x, zi)
    x.shape = x_shape

    zi.shape = zi_shape

    # move section axis back to front, axis back to original position
    zi = np.moveaxis(zi, [-2, -1], [0, axis + 1]) # type: ignore[arg-type]

    return x, cast(SosZI, zi)
