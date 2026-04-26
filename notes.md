# lupy - Testing Notes

## Framework
- pytest with pytest-xdist (parallel), pytest-codspeed (benchmarks)
- Python 3.10-3.14 supported

## Test Patterns
- Tests in `tests/` directory; source doctests also run
- One test file per module (test_meter_options.py, test_sampling.py, test_types.py, etc.)
- conftest.py has fixtures: random_samples, inc_samples, block_size, num_channels, sample_rate, etc.
- Compliance test cases in compliance_cases.py
- benchmark tests use `benchmark` fixture from pytest-codspeed

## Coverage Baseline (2026-04-26 after PR)
- meter.py: 98%
- sampling.py: 96%
- typeutils.py: 99%
- total: 95%

## Remaining Gaps
- signalutils/sosfilt.py: 89% (lines 23, 26, 28, 97)
- signalutils/resample.py: 94% (lines 99, 176, 194, 204, 264, 272, 292)
- meter.py: 98% (lines 297-298 current_measurement empty case, line 339)
- processing.py: 94%
