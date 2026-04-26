# Testing Backlog

## Completed
- [x] Meter pause functionality (set_paused, can_write/can_process, buffer clear) - PR 2026-04-26
- [x] ThreadSafeSampler / ThreadSafeTruePeakSampler - PR 2026-04-26
- [x] typeutils.py type guard helpers - PR 2026-04-26

## Pending (priority order)
1. signalutils/sosfilt.py edge cases (89%)
2. signalutils/resample.py edge cases (94%)
3. meter.py current_measurement early-exit path (line 297-298)
4. processing.py uncovered branches

## Backlog Cursor
Next: signalutils/sosfilt.py
