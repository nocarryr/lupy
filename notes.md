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
- **random_samples fixture uses fixed seed (seed=42)** since PR #121 merged 2026-05-06 — deterministic tests

## CRITICAL Maintainer Preferences
- NO section-separator comment blocks (--- style) in test files
- USE pytest fixtures for repetitive test setup (not plain helper functions)
- AVOID type: ignore comments; prefer correct type annotations
- AVOID quoted forward references (e.g. -> 'Meter') when the class is already imported
- Test Improver MUST read and address code review comments on its own PRs before the maintainer needs to ask
- PR #86 was closed because Test Improver didn't address sourcery-ai reviews itself
- Use `ensure_*` functions from lupy.typeutils instead of type: ignore
- Keep type: ignore ONLY where the shape/type is intentionally wrong for the test
- Use NumChannelsT TypeVar and generic return types: `make_meter(...) -> Meter[NumChannelsT]`
- **PR #128 merged 2026-05-09**: mypy now type-checks ALL test files (21 source files total). All test functions must have proper type annotations.

## Coverage Baseline (after 2026-05-13 run)
- meter.py: 100%
- signalutils/sosfilt.py: 100%
- signalutils/resample.py: 99% (line 99)
- typeutils.py: 99% (line 14 — typing_extensions import, covered on Python 3.12 in CI)
- sampling.py: 99% (lines 6, 170)
- filters.py: 99% (lines 6, 231)
- processing.py: 99% (lines 6, 426, 490)
- types.py: 99% (line 6)
- total: 99% (10 missed lines, all dead/untestable)

## Notes on Line 426 (processing.py) — CONFIRMED DEAD CODE
- Line 426 `rs += cur_block_wsum` is the fast path in `_calc_gating` when the
  relative threshold hasn't changed between consecutive calls.
- Due to the Sampler's 75% overlapping windowed design, adjacent gate blocks always
  have slightly different LK values (windowed energy varies per gate).
- The running mean therefore changes on every block, causing `rel_threshold_changed=True`
  on virtually every call — even with constant-frequency sine wave input.
- Line 426 can only be triggered by feeding identical constant-energy blocks directly
  via `BlockProcessor.process_block()` (bypassing the Sampler). This is an artificial
  scenario not worth testing.
- Investigated and confirmed 2026-05-13.

## Notes on Version-Conditional Imports
- Lines like `from typing_extensions import Self` (Python < 3.11 branch) show uncovered locally on 3.12
- These ARE covered in multi-version CI (Python 3.10 run) and combined in Codecov
- Not a problem; no action needed

## make_meter Pattern (from nocarryr review on PR #93)
- When a factory helper is only called directly (not injected by pytest), use a plain module-level function, NOT a @pytest.fixture
- Add explicit keyword args for test variants (e.g., `true_peak_enabled: bool = True`) instead of `**kwargs`
- Use NumChannelsT TypeVar: `def make_meter(num_channels: NumChannelsT = 2, ...) -> Meter[NumChannelsT]:`
- Fixtures are appropriate only when pytest needs to inject them as parameters

## Ruff Linting (added PR #122, 2026-05-06)
- `uv run ruff check` must pass; run before committing
- ruff configured in pyproject.toml

## mypy on Tests (added PR #128, 2026-05-09)
- `uv run mypy` now checks 21 source files including all test modules
- All test functions must have return type annotations (-> None)
- All parameters must be annotated

## gate_size Convention (from nocarryr review on PR #130 + comment on #107)
- gate_size is ALWAYS 400ms worth of samples: `int(sample_rate * 0.4)`
- NEVER use arbitrary constants like 128 or int(48000*0.1) for gate_size in tests
- Use `sample_rate` fixture (params: 48000, 44100, 88200, 96000) to vary sample rates
- nocarryr explicitly asked (2026-05-12) to audit ALL Test Improver tests for this convention
- PR #134 (merged 2026-05-12): fixed test_block_processor_momentary_silence and test_true_peak_processor_t_property
