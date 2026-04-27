# Testing Backlog

## Completed
- [x] Meter pause functionality (set_paused, can_write/can_process, buffer clear) - PR #86 (closed)
- [x] ThreadSafeSampler / ThreadSafeTruePeakSampler basic tests - PR #86 (closed)
- [x] typeutils.py type guard helpers - PR #86 (closed)
- [x] sosfilt.py validate_sos + sosfilt error paths - pending PR (branch: test-assist/sosfilt-resample-error-paths)
- [x] resample.py _UpFIRDn / ResamplePoly error paths - same PR
- [x] meter.py set_paused, current_measurement empty, write_all truncation - same PR
- [x] typeutils.py ensure_true_peak_array wrong-dtype/shape - same PR
- [x] ThreadSafeSampler per-thread RNG in concurrent test - same PR

## Pending (priority order)
1. **processing.py uncovered branches** (95%) - need deeper analysis; lines 86, 90, 93, 96, 99, 102, 105, 125, 139, 179, 185, 321-322, 386, 410, 464, 562
2. **filters.py uncovered branches** (95%) - lines 6, 70, 158-159, 187, 196, 229, 310
3. **sampling.py deeper coverage** (96%) - lines 167, 232-235, 276, 279, 378, 403, 408, 475, 562
4. **resample.py remaining** (98%) - lines 99, 194, 204

## Backlog Cursor
Next: processing.py or filters.py deeper analysis
