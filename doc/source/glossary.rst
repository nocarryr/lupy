Glossary
========

.. glossary::

    gating block
        A sliding window of 400 milliseconds with an overlap of 75%
        used to take loudness measurements.

    Integrated Loudness
        Loudness measurement of an entire "program" generated using

        - A 2-stage "K-weighting" pre-filter
        - Mean-square summation using per-channel weighting
        - Gating of 400-millisecond blocks with 75% overlap, using an absolute and relative threshold

        This algorithm is described in `ITU-R BS.1770`_.

    Momentary Loudness
        Loudness measurement generated using the algorithm described
        for :term:`Integrated Loudness`, but with a one-sample integration period.

        This algorithm is described in `ITU-R BS.1771`_.

    Short-Term Loudness
        Loudness measurement generated using the algorithm described
        for :term:`Integrated Loudness`, but without gating and with a measurement
        window of 3 seconds.

        This algorithm is described in `ITU-R BS.1771`_

    Loudness Range
        The variation of loudness throughout an entire "program" using
        a sliding 3-second analysis window and a variation of the gating
        algorithm for :term:`Integrated Loudness`.  The result is then
        computed as the difference between the upper 95th and lower 10th
        percentiles.

        This algorithm is described in `EBU Tech 3342`_

    BS 1770
        `ITU-R BS.1770`_ is the recommendation specifying methods of filtering,
        windowing and gating for loudness measurements.

    BS 1771
        `ITU-R BS.1771`_ is the recommendation specifying the requirements
        for loudness metering.  It also defines momentary and short-term
        loudness calculations.




.. _ITU-R BS.1770: https://www.itu.int/rec/R-REC-BS.1770/en
.. _ITU-R BS.1771: https://www.itu.int/rec/R-REC-BS.1771/en
.. _EBU Tech 3341: https://tech.ebu.ch/publications/tech3341
.. _EBU Tech 3342: https://tech.ebu.ch/publications/tech3342
