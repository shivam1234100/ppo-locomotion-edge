# Ablation summary: observation normalization (Hopper-v5)

Final return = mean of each seed's last 10 training episodes, then mean ± std (sample std, ddof=1) across seeds. Raw returns from the stochastic training policy.

| exp | obs_norm | seeds | final return (mean ± std) | per-seed final return | steps |
|---|---|---|---|---|---|
| baseline | on | 3 | 2132.9 ± 123.0 | 1: 2273, 2: 2043, 3: 2082 | 999,424 |
| no_obsnorm | off | 3 | 1912.0 ± 92.9 | 1: 1990, 2: 1809, 3: 1937 | 999,424 |

- no_obsnorm vs baseline: -10.4% final return
