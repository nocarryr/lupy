# Testing Backlog

## Completed
- [x] Meter pause functionality (set_paused, can_write/can_process, buffer clear) - PR #86 (closed)
- [x] ThreadSafeSampler / ThreadSafeTruePeakSampler basic tests - PR #86 (closed)
- [x] typeutils.py type guard helpers - PR #86 (closed)
- [x] processing.py RunningSum comparisons/eq/repr, lk_log10/from_lk_log10, BlockProcessor Zij/silence, TruePeakProcessor.t - PR (branch: test-assist/processing-filters-coverage)
- [x] filters.py Coeff.combine error, 1-D input path, FilterGroup assert - same PR
- [x] meter.py set_paused, current_measurement empty, write_all truncation - same PR
- [x] typeutils.py ensure_true_peak_array wrong-dtype/shape - same PR
- [x] ThreadSafeSampler per-thread RNG in concurrent test - same PR

## Pending (priority order)
1. **sampling.py deeper coverage** (96%) - lines 167, 232-235, 276, 279, 378, 403, 408, 475, 562
2. **resample.py remaining** (98%) - lines 99, 194, 204
3. **processing.py remaining** (99%) - lines 386, 464: degenerate branches in _calc_gating/_calc_lra; likely untestable without unusual edge cases
4. **filters.py remaining** (98%) - lines 187, 229: abstract method body and trivial pass; not worth testing

## Backlog Cursor
Next: task5 (comment on testing issues) or task6 (test infrastructure)
