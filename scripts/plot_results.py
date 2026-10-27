"""Learning curves and summary table for the observation-normalization ablation.

    python -m scripts.plot_results [--runs_dir runs] [--out_dir results]

Reads every runs/*/episodes.csv (runs/smoke/ is one level deeper, so it is skipped) and groups
the runs by the exp_name in their saved config.yaml.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to file; never needs a display
import matplotlib.pyplot as plt  # noqa: E402  (must come after the backend is chosen)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

GRID_STEP = 10_000  # common x-grid, so seeds that logged episodes at different steps can be averaged
WINDOW = 10  # rolling mean over the last 10 episodes, as in the training console readout
# Fixed colour per arm (colour follows the entity, never the plot order): categorical slots 1 and 2
# of the validated reference palette, checked for colour-blind separation and contrast.
ARM_COLORS = {"baseline": "#2a78d6", "no_obsnorm": "#eb6834"}
EXTRA_COLORS = ["#1baf7a", "#eda100"]  # slots 3-4, only if more arms are ever added
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(runs_dir):
    """{exp_name: [(config, episodes DataFrame, final global_step), ...]}, baseline first."""
    groups = {}
    for csv_path in sorted(Path(runs_dir).glob("*/episodes.csv")):
        cfg = yaml.safe_load((csv_path.parent / "config.yaml").read_text())
        episodes = pd.read_csv(csv_path)
        metrics = pd.read_csv(csv_path.parent / "train_metrics.csv")
        if episodes.empty or metrics.empty:
            continue  # a run that crashed before finishing an iteration has nothing to plot
        groups.setdefault(cfg["exp_name"], []).append((cfg, episodes, int(metrics["global_step"].iloc[-1])))
    return dict(sorted(groups.items(), key=lambda kv: (kv[0] != "baseline", kv[0])))


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves and write the summary table.")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    groups = load_runs(args.runs_dir)
    assert groups, f"no runs with episodes.csv found under {args.runs_dir}/"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only up to the step every run reached: no extrapolation past the end of a shorter run.
    last_common = min(ep["global_step"].iloc[-1] for runs in groups.values() for _, ep, _ in runs)
    grid = np.arange(GRID_STEP, last_common + 1, GRID_STEP)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    summary = []
    for i, (exp, runs) in enumerate(groups.items()):
        color = ARM_COLORS.get(exp, EXTRA_COLORS[i % len(EXTRA_COLORS)])
        curves, finals = [], []
        for cfg, ep, _ in runs:
            smooth = ep["episodic_return"].rolling(WINDOW, min_periods=1).mean()
            # Linear interpolation of the smoothed curve onto the shared grid.
            curves.append(np.interp(grid, ep["global_step"], smooth))
            finals.append(ep["episodic_return"].tail(WINDOW).mean())  # final return of this seed
        curves = np.array(curves)
        # Sample std (ddof=1): with only 3 seeds it is the less optimistic estimate of the spread.
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        obs_norm = "on" if runs[0][0]["obs_norm"] else "off"
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, mean, color=color, linewidth=1.6, label=f"{exp} (obs norm {obs_norm})")
        # Direct label at the line end, in neutral ink (the line's colour already carries identity).
        # Name only: the final numbers live in summary.md, computed from the raw episodes.
        ax.annotate(exp, (grid[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_MUTED)
        summary.append({
            "exp": exp, "obs_norm": obs_norm, "seeds": sorted(cfg["seed"] for cfg, _, _ in runs),
            "finals": finals, "steps": max(steps for _, _, steps in runs),
        })

    n_seeds = max(len(runs) for runs in groups.values())
    # pad lifts the title clear of the subtitle line placed just above the axes.
    ax.set_title(f"Hopper-v5 PPO: observation-normalization ablation ({n_seeds} seeds per arm)",
                 loc="left", fontsize=11, color=INK, pad=20)
    ax.text(0, 1.02, f"Mean ± 1 std across seeds of the rolling mean of the last {WINDOW} episodes "
                     "(raw, unnormalized returns)",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED, va="bottom")
    ax.set_xlabel("Environment steps", color=INK_MUTED)
    ax.set_ylabel("Episodic return", color=INK_MUTED)
    ax.xaxis.set_major_formatter(lambda x, _: "0" if x == 0 else (f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"))
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(bottom=0)
    # Recessive scaffolding: light horizontal grid only, no top/right frame.
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    png = out_dir / "learning_curves.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE)

    lines = [
        "# Ablation summary: observation normalization (Hopper-v5)",
        "",
        f"Final return = mean of each seed's last {WINDOW} training episodes, then mean ± std (sample std, "
        "ddof=1) across seeds. Raw returns from the stochastic training policy.",
        "",
        "| exp | obs_norm | seeds | final return (mean ± std) | per-seed final return | steps |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        finals = np.array(row["finals"])
        per_seed = ", ".join(f"{s}: {f:.0f}" for s, f in zip(row["seeds"], finals))
        lines.append(f"| {row['exp']} | {row['obs_norm']} | {len(finals)} | {finals.mean():.1f} ± "
                     f"{finals.std(ddof=1):.1f} | {per_seed} | {row['steps']:,} |")
    if "baseline" in groups:
        base = np.mean(next(r["finals"] for r in summary if r["exp"] == "baseline"))
        lines.append("")
        for row in summary:
            if row["exp"] != "baseline":
                change = 100 * (np.mean(row["finals"]) - base) / base
                lines.append(f"- {row['exp']} vs baseline: {change:+.1f}% final return")
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report)
    print(report)
    print(f"wrote {png} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
