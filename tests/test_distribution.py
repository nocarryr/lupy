# A simple test to ensure that the distribution can be installed and imported.


def test_import():
    from lupy import Meter, Sampler, BlockProcessor, TruePeakProcessor
    assert Meter is not None
    assert Sampler is not None
    assert BlockProcessor is not None
    assert TruePeakProcessor is not None

    meter = Meter(block_size=128, num_channels=2, sample_rate=48000)
    assert meter is not None
