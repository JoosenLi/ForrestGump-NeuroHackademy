#!/usr/bin/env python3
"""Plot LOSO and within-subject performance across FIR-style BOLD delays."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()
    frames = []
    for lag_dir in sorted(args.results_root.glob("lag-*tr")):
        lag = int(lag_dir.name.removeprefix("lag-").removesuffix("tr"))
        for mode in ("loso", "within"):
            path = lag_dir / f"{mode}.csv"
            if path.exists():
                frame = pd.read_csv(path); frame["lag_trs"] = lag; frame["lag_seconds"] = 2 * lag
                frames.append(frame)
    if not frames: raise FileNotFoundError("No lag evaluation CSV files found")
    data = pd.concat(frames, ignore_index=True)
    summary = data.groupby(["mode", "target", "lag_trs", "lag_seconds"]).pearson_r.agg(
        count="count", mean="mean", median="median", std="std").reset_index()
    summary["sem"] = summary["std"] / summary["count"].pow(.5)
    args.summary.parent.mkdir(parents=True, exist_ok=True); summary.to_csv(args.summary, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    colors = {"loso": "#4c78a8", "within": "#f58518"}
    labels = {"loso": "LOSO", "within": "Within (runs 1–6 → 7–8)"}
    for ax, target in zip(axes, ("valence", "arousal")):
        for mode in ("loso", "within"):
            s = summary[(summary["target"] == target) & (summary["mode"] == mode)].sort_values("lag_seconds")
            ax.errorbar(s["lag_seconds"].to_numpy(), s["mean"].to_numpy(), yerr=s["sem"].to_numpy(), marker="o", capsize=3,
                        color=colors[mode], label=labels[mode])
        ax.axhline(0, color="0.35", linewidth=.8); ax.set_xlabel("BOLD delay (s)"); ax.set_title(target.capitalize())
    axes[0].set_ylabel("Mean held-out Pearson r"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("StudyForrest emotion decoding: FIR-style delay sweep", y=1.02)
    fig.tight_layout(); args.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(summary.to_string(index=False)); print(f"saved {args.output}")


if __name__ == "__main__": main()
