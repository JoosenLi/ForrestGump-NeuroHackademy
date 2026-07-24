#!/usr/bin/env python3
"""ROI inter-subject correlation for synchronized StudyForrest movie cuts.

The input is a lag-specific ``features.npz`` written by merge_feature_sets.py.
It contains one row per labelled TR, subject and run.  Within each run, rows
are aligned by ordinal TR after a fixed BOLD delay has been applied upstream.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--delay-seconds", type=float, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    raw = np.load(args.data, allow_pickle=False)
    x = raw["X"].astype(float)
    y = raw["y"].astype(float)
    subject = raw["subject"].astype(str)
    run = raw["run"].astype(int)
    rois = raw["feature_names"].astype(str).tolist()
    arousal = y[:, raw["target_columns"].astype(str).tolist().index("arousal")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loo_rows, pair_rows = [], []
    subjects = sorted(np.unique(subject))
    for run_id in sorted(np.unique(run)):
        # The merged files preserve TR order, so ordinal position aligns a cut.
        rows = {s: np.flatnonzero((subject == s) & (run == run_id)) for s in subjects}
        n = min(map(len, rows.values()))
        if n < 3:
            continue
        values = {s: x[idx[:n]] for s, idx in rows.items()}
        labels = np.mean([arousal[idx[:n]] for idx in rows.values()], axis=0)
        for held_out in subjects:
            others = [s for s in subjects if s != held_out]
            group = np.mean([values[s] for s in others], axis=0)
            for roi_i, roi in enumerate(rois):
                loo_rows.append({"subject": held_out, "run": run_id, "roi": roi,
                                 "loo_isc_r": corr(values[held_out][:, roi_i], group[:, roi_i]),
                                 "bold_delay_seconds": args.delay_seconds})
        high, low = labels >= np.median(labels), labels < np.median(labels)
        for a, b in combinations(subjects, 2):
            for roi_i, roi in enumerate(rois):
                pair_rows.append({"subject_a": a, "subject_b": b, "roi": roi,
                                  "high_arousal_isc_r": corr(values[a][high, roi_i], values[b][high, roi_i]),
                                  "low_arousal_isc_r": corr(values[a][low, roi_i], values[b][low, roi_i]),
                                  "bold_delay_seconds": args.delay_seconds})

    def write(name: str, rows: list[dict]) -> None:
        with (args.output_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    write("isc_loo_by_subject_run_roi.csv", loo_rows)
    write("isc_arousal_high_vs_low_by_pair_roi.csv", pair_rows)

    summary = []
    for roi in rois:
        loo = np.array([r["loo_isc_r"] for r in loo_rows if r["roi"] == roi], float)
        pairs = [r for r in pair_rows if r["roi"] == roi]
        diff = np.array([r["high_arousal_isc_r"] - r["low_arousal_isc_r"] for r in pairs], float)
        summary.append({"roi": roi, "loo_isc_mean": np.nanmean(loo),
                        "high_minus_low_mean": np.nanmean(diff),
                        "high_minus_low_sem": np.nanstd(diff, ddof=1) / np.sqrt(np.sum(~np.isnan(diff)))})
    write("isc_summary.csv", summary)

    ordered = sorted(summary, key=lambda row: row["loo_isc_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([r["roi"] for r in ordered][::-1], [r["loo_isc_mean"] for r in ordered][::-1])
    ax.set(xlabel="leave-one-out ISC (Pearson r)", title=f"Same movie cut ISC ({args.delay_seconds:g}-s BOLD delay)")
    fig.tight_layout(); fig.savefig(args.output_dir / "isc_same_movie_cut.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
