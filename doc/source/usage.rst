Usage
=====

.. currentmodule:: lupy.meter


.. warning::

    None of the example code below has been tested yet.

    Use at your own risk!


.. todo::

    Test the example code below


Setup
-----

The main interface with this library is the :class:`Meter`
class.

>>> import threading
>>> import time
>>> import numpy as np
>>> from lupy import Meter

Typical audio interface libraries have a configurable :term:`block size`
which must be known before creating the Meter. For this example
we will be using `python-sounddevice`_.

>>> import sounddevice as sd

>>> block_size = 512
>>> num_channels = 1
>>> meter = Meter(block_size=block_size, num_channels=1)


Configuring Measurements
------------------------

By default all measurement types are enabled. You can selectively disable
measurements you don't need to reduce processing overhead:

All ``*_enabled`` parameters default to ``True``, so they are not required
when all measurements are desired:

>>> meter = Meter(
...     block_size=block_size,
...     num_channels=1,
...     momentary_enabled=True,
...     short_term_enabled=True,
...     lra_enabled=True,
...     true_peak_enabled=True,
... )

For example, if you only need :term:`Integrated Loudness` you can disable
the others:

>>> meter = Meter(
...     block_size=block_size,
...     num_channels=1,
...     momentary_enabled=False,
...     short_term_enabled=False,
...     lra_enabled=False,
...     true_peak_enabled=False,
... )

.. important::

    :term:`Loudness Range` (LRA) calculation depends on
    :term:`Short-Term Loudness`.  If *short_term_enabled* is ``False``,
    *lra_enabled* must also be ``False``.  Passing
    ``short_term_enabled=False, lra_enabled=True`` raises a
    :exc:`ValueError`.


Now we need to define a callback for the sample data. The samples in
the callback's ``indata`` are shaped as ``(num_samples, num_channels)``
so we need to swap the axes.

Then we make sure there is enough room in the meter's :attr:`sample buffer <Meter.sampler>`
using :meth:`Meter.can_write` and give it to the meter using its :meth:`Meter.write` method.

The ``process=False`` argument is set to avoid doing any more work within the
callback (since it's called from the audio thread). Instead we'll use a
:class:`threading.Event` to signal when it's been written.


Handling Input
--------------

>>> samples_ready = threading.Event()

>>> def audio_callback(indata, outdata, frames, time, status):
...     if status:
...         print(status)
...     indata = np.swapaxes(indata, 0, 1)
...     if not meter.can_write():
...         print('buffer full!')
...     meter.write(indata, process=False)
...     samples_ready.set()


Processing
----------

The processing can then be done from the main loop using the :meth:`Meter.can_process`
and :meth:`Meter.process` methods. We'll limit the total duration
to ten seconds for this example.

>>> max_duration = 10
>>> time_remaining = time.time() + max_duration
>>> with sd.InputStream(
...     device=None,
...     channels=meter.num_channels,
...     samplerate=meter.sample_rate,
...     callback=audio_callback,
... ):
...     while True:
...         samples_ready.wait()
...         samples_ready.clear()
...         if meter.can_process():
...             meter.process()
...         if time.time() >= time_remaining:
...             break


.. channel-layout:

Channel Layout
--------------

When measuring more than one channel, the expected layout should follow
the order below.  This is important because channels are weighted differently
in the calculations.


.. list-table:: Stereo
    :header-rows: 1
    :align: left
    :widths: auto

    * - Index
      - Name
    * - 0
      - Left
    * - 1
      - Right


.. list-table:: LCR
    :header-rows: 1
    :align: left
    :widths: auto

    * - Index
      - Name
    * - 0
      - Left
    * - 1
      - Center
    * - 2
      - Right


.. list-table:: Surround
    :header-rows: 1
    :align: left
    :widths: auto

    * - Index
      - Name
    * - 0
      - Left
    * - 1
      - Center
    * - 2
      - Right
    * - 3
      - Left Surround
    * - 4
      - Right Surround



Measurement Data
----------------


Scalar Values
^^^^^^^^^^^^^

These attributes contain a single value which is updated
each time the meter processes new samples.

* :attr:`Meter.integrated_lkfs` holds the latest integrated loudness value.
* :attr:`Meter.lra` holds the latest loudness range value.
* :attr:`Meter.true_peak_current` holds the latest true peak values for each channel.
* :attr:`Meter.true_peak_max` holds the maximum true peak value observed across all channels.

.. tip::

  For convenience, all relevant measurement values can be accessed together
  through :attr:`Meter.current_measurement`. This returns an object containing
  the items listed above as well as other useful measurements.

  See the :class:`~.CurrentMeasurement` class reference for the full list.


Array Values
^^^^^^^^^^^^

These attributes contain arrays of the measurement
values from each processing iteration (every 100 ms).

When accessed, the arrays will contain the most
recent data point in their last element. In other words,
when a new 100ms chunk of input has been processed, the
arrays will be one element larger (when accessed).


Structured Arrays
"""""""""""""""""

The historical :term:`Momentary Loudness` and :term:`Short-Term Loudness`
measurements can be accessed as a :term:`structured array` through :attr:`Meter.block_data`:

>>> ms_times = meter.block_data['t']           # A 1D array of times in seconds
>>> momentary_values = meter.block_data['m']   # A 1D array of momentary loudness values
>>> short_term_values = meter.block_data['s']  # A 1D array of short-term loudness values


The historical :term:`True Peak` measurements can be accessed
from :attr:`Meter.true_peak_array` which contains the peak values
for each channel along with their measurement times:

>>> tp_times = meter.true_peak_array['t']   # A 1D array of times in seconds
>>> tp_values = meter.true_peak_array['tp'] # A 2D array of shape (n, num_channels)


Direct Access
"""""""""""""

* :attr:`Meter.t` holds the times for momentary/short-term measurements.
* :attr:`Meter.momentary_lkfs` holds the historical momentary loudness values.
* :attr:`Meter.short_term_lkfs` holds the historical short-term loudness values.







.. _python-sounddevice: https://python-sounddevice.readthedocs.io/en/0.4.7/examples.html
