#!/usr/bin/env python3
"""Visualize StudyForrest's curated human emotion annotations over movie time."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sample_events(events: pd.DataFrame, columns: list[str], duration: int) -> np.ndarray:
    """One-second overlap-weighted mean; missing means no human annotation."""
    start = events["onset"].to_numpy(float)
    end = start + events["duration"].to_numpy(float)
    values = events[columns].to_numpy(float)
    output = np.full((duration, len(columns)), np.nan)
    for second in range(duration):
        overlap = np.maximum(0, np.minimum(end, second + 1) - np.maximum(start, second))
        if overlap.sum() > 0:
            output[second] = (values * overlap[:, None]).sum(axis=0) / overlap.sum()
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, required=True,
                    help=".../curated/segments/avmovie")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    chunks, boundaries, run_labels = [], [0], []
    emotion_columns = None
    for run in range(1, 9):
        events = pd.read_csv(args.annotations / f"emotions_av_1s_events_run-{run}_events.tsv", sep="\t")
        if emotion_columns is None:
            emotion_columns = [c for c in events if c.startswith("e_")]
        events["valence"] = events["valence_positive"] - events["valence_negative"]
        columns = ["valence", "arousal", *emotion_columns]
        duration = int(np.ceil((events["onset"] + events["duration"]).max()))
        chunks.append(sample_events(events, columns, duration))
        boundaries.append(boundaries[-1] + duration)
        run_labels.append((boundaries[-2] + boundaries[-1]) / 2)
    data = np.concatenate(chunks, axis=0)
    time_min = np.arange(len(data)) / 60
    fig, (ax_dim, ax_heat) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 3]})
    ax_dim.plot(time_min, data[:, 0], lw=.7, color="#3572a5", label="Valence (positive − negative)")
    ax_dim.plot(time_min, data[:, 1], lw=.7, color="#e6812d", label="Arousal")
    ax_dim.axhline(0, color="0.4", linewidth=.7); ax_dim.set_ylabel("Human rating")
    ax_dim.legend(loc="upper right", ncol=2, frameon=False)
    heat = np.ma.masked_invalid(data[:, 2:].T)
    cmap = plt.colormaps["magma"].copy(); cmap.set_bad("white")
    im = ax_heat.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
                        extent=[0, len(data) / 60, len(emotion_columns) - .5, -.5])
    ax_heat.set_yticks(range(len(emotion_columns)))
    ax_heat.set_yticklabels([x.removeprefix("e_") for x in emotion_columns], fontsize=8)
    ax_heat.set_ylabel("Discrete emotion intensity")
    ax_heat.set_xlabel("Concatenated movie time (minutes)")
    for ax in (ax_dim, ax_heat):
        for boundary in boundaries[1:-1]: ax.axvline(boundary / 60, color="white" if ax is ax_heat else "0.4", lw=.8)
    for x, label in zip(run_labels, range(1, 9)):
        ax_dim.text(x / 60, 1.03, f"run {label}", ha="center", va="bottom", fontsize=8,
                    transform=ax_dim.get_xaxis_transform())
    colorbar = fig.colorbar(im, ax=ax_heat, pad=.01, fraction=.025)
    colorbar.set_label("Human-rated emotion intensity")
    fig.suptitle("StudyForrest: curated human emotion annotations\nwhite heatmap regions = unannotated movie time", y=.98)
    fig.tight_layout(rect=[0, 0, 1, .95]); args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved {args.output}; {len(data)} one-second movie samples; {len(emotion_columns)} emotion dimensions")


if __name__ == "__main__":
    main()
