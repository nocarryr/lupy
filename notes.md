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

## CRITICAL Maintainer Priority
- Test Improver MUST read and address code review comments on its own PRs before the maintainer needs to ask
- PR #86 was closed because Test Improver didn't address sourcery-ai reviews itself
- Quote: "make the [Test Improver] agentic workflow do its job and read the code reviews here"

## Coverage Baseline (after this run, 2026-04-27)
- meter.py: 100%
- signalutils/sosfilt.py: 100%
- signalutils/resample.py: 98%
- typeutils.py: 99%
- sampling.py: 96%
- total: 97%

## Remaining Gaps
- signalutils/resample.py: 98% (lines 99, 194, 204)
- sampling.py: 96% (lines 167, 232-235, 276, 279, 378, 403, 408, 475, 562)
- processing.py: 95%
- filters.py: 95%
