lupy
####

Python library for `Audio Loudness`_ measurements.
Yet another implementation of `ITU-R BS.1770`_, `ITU-R BS.1771`_,
`EBU Tech 3341`_ and `EBU Tech 3342`_.


Features
========

* Supports calculations for the following measurements

  * Integrated Loudness
  * Short-Term Loudness
  * Momentary Loudness
  * Loudness Range (LRA)
  * True Peak

* Handles various input lengths suitable for most audio engines



Installation
============

Pip
---

Pip install not yet available since there is another project named "lupy" on
the Python Package Index (PyPI).

Renaming and/or publishing to PyPI will hopefully come soon.


..

.. .. code-block:: bash

..    $ pip install lupy



Dependencies
============

* numpy
* scipy


Project Links
=============


Homepage
   https://github.com/nocarryr/lupy

Documentation
   https://lupy-nocarryr.readthedocs.io/



License
=======

This project is licensed under the MIT License - see the `LICENSE`_ file for details.


Portions of the code (src/lupy/signalutils) are derived from the Scipy library, which is
licensed under the BSD License.  See the `LICENSE-SCIPY`_ file for details.


.. _Audio Loudness: https://tech.ebu.ch/loudness
.. _ITU-R BS.1770: https://www.itu.int/rec/R-REC-BS.1770/en
.. _ITU-R BS.1771: https://www.itu.int/rec/R-REC-BS.1771/en
.. _EBU Tech 3341: https://tech.ebu.ch/publications/tech3341
.. _EBU Tech 3342: https://tech.ebu.ch/publications/tech3342
.. _LICENSE: LICENSE
.. _LICENSE-SCIPY: LICENSE-SCIPY
