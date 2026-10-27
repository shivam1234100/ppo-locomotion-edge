# Fixed-rate control loop

- Policy: `models/policy_int8.onnx` in ONNX Runtime (1 thread) driving Hopper-v5 | CPU: Apple M1
- Loop: 1,250 ticks at 8 ms (125 Hz), 9.99 s wall time; absolute `perf_counter` deadlines, sleep then spin for the last 3 ms

| Timing (µs) | p50 | p99 | max |
|---|---|---|---|
| inference | 95.8 | 187.4 | 251.2 |
| work per tick (obs -> infer -> env.step) | 556.4 | 781.0 | 1323.9 |
| tick-start error (lateness) | 0.1 | 0.2 | 19.7 |

- Deadline misses (work > 8 ms): **0 / 1,250**
- Jitter (std of tick-start error): **0.6 µs**
- Worst-case work per tick: 1323.9 µs = 16.55% of the period
- Episodes finished during the run: 1 (mean return 2072.6)
