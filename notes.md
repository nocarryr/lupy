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

## CRITICAL Maintainer Preferences
- NO section-separator comment blocks (--- style) in test files
- USE pytest fixtures for repetitive test setup (not plain helper functions)
- AVOID type: ignore comments; prefer correct type annotations
- AVOID quoted forward references (e.g. -> 'Meter') when the class is already imported
- Test Improver MUST read and address code review comments on its own PRs before the maintainer needs to ask
- PR #86 was closed because Test Improver didn't address sourcery-ai reviews itself
- Use `ensure_*` functions from lupy.typeutils instead of type: ignore — e.g. `validate_sos(ensure_2d_array(sos))` lets mypy narrow the type without type: ignore
- Keep type: ignore ONLY where the shape/type is intentionally wrong for the test (e.g. passing 1D to a 2D-expecting fn, float32 to float64-expecting fn)
- Use NumChannelsT TypeVar and generic return types: `make_meter(...) -> Meter[NumChannelsT]`

## Coverage Baseline (after 2026-05-03 run)
- meter.py: 100%
- signalutils/sosfilt.py: 100%
- signalutils/resample.py: 99%
- typeutils.py: 99%
- sampling.py: 98%
- filters.py: 98%
- processing.py: 99% (lines 179,185,411,475 — all untestable abstract/dead-code)
- total: 99%

## Remaining Gaps (all low-priority / intentionally skipped)
- sampling.py line 167: defensive guard, dead code
- sampling.py lines 378/403/408: abstract NotImplementedError bodies
- resample.py line 99: n_post_pad += 1 loop — specialized numerical condition
- processing.py lines 179, 185: abstract method raise NotImplementedError bodies (unreachable via subclasses)
- processing.py lines 411, 475: incremental gating fast-path and degenerate LRA branch
- filters.py lines 187, 229: abstract method body and trivial pass

## make_meter Pattern (from nocarryr review on PR #93)
- When a factory helper is only called directly (not injected by pytest), use a plain module-level function, NOT a @pytest.fixture
- Add explicit keyword args for test variants (e.g., `true_peak_enabled: bool = True`) instead of `**kwargs`
- Use NumChannelsT TypeVar: `def make_meter(num_channels: NumChannelsT = 2, ...) -> Meter[NumChannelsT]:`
- Fixtures are appropriate only when pytest needs to inject them as parameters
