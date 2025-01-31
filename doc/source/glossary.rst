Glossary
========

.. glossary::

    Loudness
        An objective measurement of the perceived loudness of an
        audio signal. The algorithms for this measurement have been
        studied and designed by the `EBU PLOUD Group`_ and described
        in the `EBU R128`_ Loudness Recommendation.

    gating block
        A sliding window of 400 milliseconds in duration with an overlap of 75%
        used to take loudness measurements.  With the overlap factored in,
        a new gating block will occur every 100 milliseconds.

    Integrated Loudness
        :term:`Loudness` measurement of an entire "program" generated using

        - A 2-stage "K-weighting" pre-filter
        - Mean-square summation using per-channel weighting
        - Gating of 400-millisecond blocks with 75% overlap, using an absolute and relative threshold

        This algorithm is described in `ITU-R BS.1770`_.

    Loudness Unit
        "Loudness Unit (``LU``)" is the scale of a :term:`loudness <Loudness>` measuring
        device or meter.

        The value of an audio signal in loudness units represents
        the loss or gain dB that is required to bring the signal to ``0 LU``,
        e.g. a signal that reads ``-10 LU`` will require ``10 dB`` of gain to bring that signal
        up to a reading of ``0 LU``.

        This is defined in `ITU-R BS.1770`_ (Annex 1) and clarified in
        `EBU Tech 3341`_ (section 2.5).

    LUFS
        *Absolute* :term:`Loudness Unit` (with respect to Full Scale).

        This is defined in `EBU Tech 3341`_.

    LU
        :term:`Loudness Unit` *relative* to a target level, typically user-defined.

        If a target of ``-23 LUFS`` is chosen, the measurement of a signal at
        ``-23.0 LUFS`` would be displayed as ``0.0 LU``.

        This is defined in `EBU Tech 3341`_.

    Momentary Loudness
        :term:`Loudness` measurement generated using the algorithm described
        for :term:`Integrated Loudness`, but with a one-sample integration period.

        This algorithm is described in `ITU-R BS.1771`_.

    Short-Term Loudness
        :term:`Loudness` measurement generated using the algorithm described
        for :term:`Integrated Loudness`, but without gating and with a measurement
        window of 3 seconds.

        This algorithm is described in `ITU-R BS.1771`_

    Loudness Range
        The variation of :term:`loudness <Loudness>` throughout an entire "program" using
        a sliding 3-second analysis window and a variation of the gating
        algorithm for :term:`Integrated Loudness`.  The result is then
        computed as the difference between the upper 95th and lower 10th
        percentiles.

        This algorithm is described in `EBU Tech 3342`_

    True Peak
        The maximum (positive or negative) value of a waveform in the continuous
        time domain as opposed to the "sample peaks" typically measured in discrete time.
        This is calculated by oversampling the signal and applying an
        interpolating filter as described in `ITU-R BS.1770`_.

    block size
        The number of audio samples per channel processed within one period.
        This is typically chosen before the audio processing engine starts
        and remains constant throughout the session.

    BS 1770
        `ITU-R BS.1770`_ is the recommendation specifying methods of filtering,
        windowing and gating for loudness measurements.

    BS 1771
        `ITU-R BS.1771`_ is the recommendation specifying the requirements
        for loudness metering.  It also defines momentary and short-term
        loudness calculations.




.. _EBU PLOUD Group: https://tech.ebu.ch/loudness
.. _EBU R128: https://tech.ebu.ch/docs/r/r128.pdf
.. _ITU-R BS.1770: https://www.itu.int/rec/R-REC-BS.1770/en
.. _ITU-R BS.1771: https://www.itu.int/rec/R-REC-BS.1771/en
.. _EBU Tech 3341: https://tech.ebu.ch/publications/tech3341
.. _EBU Tech 3342: https://tech.ebu.ch/publications/tech3342
