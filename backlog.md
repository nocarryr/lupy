# Testing Backlog

## Completed
- [x] Meter pause functionality (set_paused, can_write/can_process, buffer clear) - PR #86
- [x] ThreadSafeSampler / ThreadSafeTruePeakSampler - PR #86
- [x] typeutils.py type guard helpers - PR #86

## Pending (priority order)
1. **sosfilt.py validate_sos error paths** (89%) - lines 23, 26, 28: three ValueError raises for invalid sos array shape/values; plus line 97 (sosfilt zi shape mismatch). Straightforward validation tests.
2. **meter.py current_measurement empty case** (98%) - lines 297-298: the `if block_data.size == 0` branch (no blocks processed yet). Easy win.
3. **resample.py _UpFIRDn error paths** (94%) - lines 264, 272, 292: ValueError for h not 1D, up/down < 1, dtype mismatch. Also line 176: ResamplePoly.num_input_samples setter no-op early return.
4. **processing.py uncovered branches** (94%) - need deeper analysis

## Backlog Cursor
Next: sosfilt.py validate_sos error paths + meter.py current_measurement empty case
