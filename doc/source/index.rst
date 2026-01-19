.. lupy documentation master file, created by
   sphinx-quickstart on Sun Jun 16 15:50:01 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================
Welcome to lupy's documentation!
================================

A Python library for :term:`Loudness` measurements of audio signals.


Purpose
=======

This library provides a way to measure audio loudness and true peak measurements
validated against the official EBU compliance test materials.

Its design focuses more on real-time processing, using a block-based approach
suitable for audio engines, rather than offline processing of entire audio files.


Features
========

* Historical tracking of all measurements over time
* Multi-channel audio support
* Compliance-based validation of measurements


Measurement Types
-----------------

* :term:`Integrated Loudness`
* :term:`Short-Term Loudness`
* :term:`Momentary Loudness`
* :term:`Loudness Range` (LRA)
* :term:`True Peak`



Compliance
----------

This library is validated against comprehensive compliance test cases based on
the official EBU and ITU-R documents:

`EBU Tech 3341`_
  Integrated, Momentary, Short-Term and True Peak measurements

`EBU Tech 3342`_
  Loudness Range (LRA) measurements

`ITU-R BS.2217`_
  Integrated Loudness measurements for various channel configurations




Installation
============

Pip
---

Pip install not yet available since there is another project named "lupy" on
the Python Package Index (PyPI).

Renaming and/or publishing to PyPI will hopefully come soon.


.. .. code-block:: bash

..    $ pip install lupy


Dependencies
============

.. project-dependencies::


Project Links
=============


Homepage
   :project-url:`Homepage`

Documentation
   :project-url:`Documentation`


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   reference/index.rst
   glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _EBU Tech 3341: https://tech.ebu.ch/publications/tech3341
.. _EBU Tech 3342: https://tech.ebu.ch/publications/tech3342
.. _ITU-R BS.2217: https://www.itu.int/pub/R-REP-BS.2217
