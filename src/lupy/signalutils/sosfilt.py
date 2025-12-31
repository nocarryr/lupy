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



# # https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_signaltools.py#L96-L109
# def _reject_objects(arr, name):
#     """Warn if arr.dtype is object or longdouble.
#     """
#     dt = np.asarray(arr).dtype
#     if not (np.issubdtype(dt, np.integer)
#             or dt in [np.bool_, np.float16, np.float32, np.float64,
#                       np.complex64, np.complex128]
#     ):
#         msg = (
#             f"dtype={dt} is not supported by {name} and will raise an error in "
#             f"SciPy 1.17.0. Supported dtypes are: boolean, integer, `np.float16`,"
#             f"`np.float32`, `np.float64`, `np.complex64`, `np.complex128`."
#         )
#         warnings.warn(msg, category=DeprecationWarning, stacklevel=3)


def validate_sos(sos: Float2dArray) -> SosCoeff:
    """Helper to validate a SOS input"""
    # sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    _, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return cast(SosCoeff, sos)


# # https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_signaltools.py#L4594-L4598
# def _validate_x(x):
#     x = np.asarray(x)
#     if x.ndim == 0:
#         raise ValueError('x must be at least 1-D')
#     return x



# https://github.com/scipy/scipy/blob/e29dcb65a2040f04819b426a04b60d44a8f69c04/scipy/signal/_signaltools.py#L4601-L4715
def sosfilt(sos: SosCoeff, x: Float2dArray, axis: int, zi: SosZI) -> tuple[Float2dArray, SosZI]:
    """Filter data along one dimension using a digital filter defined by
    second-order sections.

    This is a stripped down version of :func:`scipy.signal.sosfilt` to reduce
    overhead in repeated calls.

    It removes input validation and assumes 2D input with filtering along the last axis.

    It also requires the initial conditions *zi* to be provided and returns
    the final conditions.
    """

    # We validate these below to np.float64
    # _reject_objects(sos, 'sosfilt')
    # _reject_objects(x, 'sosfilt')
    # # if zi is not None:
    # _reject_objects(zi, 'sosfilt')

    # x = _validate_x(x)
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
