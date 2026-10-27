# Inference latency benchmark

- CPU: Apple M1 (arm64) | OS: macOS-14.8.7-arm64-arm-64bit
- Python 3.11.18 | torch 2.14.1 | onnxruntime 1.31.1 | onnx 1.23.2 | numpy 2.4.6 | gymnasium 1.4.0 | mujoco 3.17.0
- Setup: batch 1 (one control step), one thread everywhere, 200 warm-up + 10,000 timed calls on real observations, `time.perf_counter_ns()` around each call; ORT graph optimization ORT_ENABLE_ALL
- Policy: `models/best.pt` (baseline, seed 3) | control period 8 ms (`env.unwrapped.dt`) | eval seeds 1000..1009, deterministic actions

| Variant | p50 (µs) | p95 (µs) | p99 (µs) | mean (µs) | inferences/s | size (KB) | eval return (10 ep) | max abs action diff vs PyTorch | p99 % of 8 ms |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 (eager) | 25.2 | 25.5 | 25.9 | 25.3 | 39,546 | 23.5 | 3566.0 ± 7.7 | 0.0e+00 | 0.32% |
| ONNX Runtime FP32 | 6.6 | 6.8 | 6.9 | 6.6 | 150,998 | 22.1 | 3566.5 ± 8.3 | 6.3e-07 | 0.09% |
| ONNX Runtime INT8 | 9.0 | 9.2 | 9.3 | 9.1 | 110,307 | 9.3 | 3494.9 ± 182.2 | 1.4e-01 | 0.12% |
