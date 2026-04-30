# Testing Backlog

## Completed
- [x] Meter pause functionality (set_paused, can_write/can_process, buffer clear) - PR #86 (closed)
- [x] ThreadSafeSampler / ThreadSafeTruePeakSampler basic tests - PR #86 (closed)
- [x] typeutils.py type guard helpers - PR #86 (closed)
- [x] processing.py RunningSum comparisons/eq/repr, lk_log10/from_lk_log10, BlockProcessor Zij/silence, TruePeakProcessor.t - PR #93 (merged)
- [x] filters.py Coeff.combine error, 1-D input path, FilterGroup assert - PR #97 (merged)
- [x] meter.py set_paused, current_measurement empty, write_all truncation - PR #97 (merged)
- [x] typeutils.py ensure_true_peak_array wrong-dtype/shape - PR #93 (merged)
- [x] ThreadSafeSampler per-thread RNG in concurrent test - PR #93 (merged)
- [x] sampling.py: Slice.__repr__/__str__, calc_shape non-last-axis, Sampler.write float32, TruePeakSampler buffer doubling - PR test-assist/sampling-resample-coverage-gaps (2026-04-30)
- [x] resample.py: ResamplePoly.num_output_samples, output_shape - same PR

## Pending (low-priority, hard to reach)
1. **sampling.py line 167**: defensive `ix = 0` when start_index negative — dead code under normal usage; skip
2. **sampling.py lines 378, 403, 408**: abstract `raise NotImplementedError` bodies — unreachable via subclasses; skip
3. **resample.py line 99**: `n_post_pad += 1` in `_design_poly_filter` — specialized numerical condition; skip
4. **processing.py lines 390, 454**: degenerate `_calc_gating`/`_calc_lra` branches — likely untestable without unusual edge cases
5. **filters.py lines 187, 229**: abstract method body and trivial pass; not worth testing

## Backlog Cursor
Most valuable remaining gaps have been addressed. Total coverage: 99%.
Next run: task5 (comment on testing issues) or task6 (test infrastructure)
