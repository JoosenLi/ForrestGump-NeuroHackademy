#!/usr/bin/env python3
"""Convert StudyForrest's 1 Hz emotion events to fMRI-TR targets.

Only TRs overlapping a human-annotated event are retained.  This avoids
silently treating unannotated parts of the film as "neutral".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def interval_mean(frame: pd.DataFrame, start: float, end: float, columns: list[str]):
    overlap = np.maximum(
        0.0,
        np.minimum(frame["onset"].to_numpy() + frame["duration"].to_numpy(), end)
        - np.maximum(frame["onset"].to_numpy(), start),
    )
    if overlap.sum() == 0:
        return None
    values = frame[columns].to_numpy(dtype=np.float32)
    return (values * overlap[:, None]).sum(axis=0) / overlap.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--n-trs", type=int, required=True)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--lag-trs", type=int, default=2, help="BOLD delay; default=4 s")
    ap.add_argument("--presentation-events", type=Path,
                    help="raw movie presentation log; maps scanner seconds to actual movie videotime")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    source = args.annotations / f"emotions_av_1s_events_run-{args.run}_events.tsv"
    events = pd.read_csv(source, sep="\t")
    emotion_cols = [c for c in events if c.startswith("e_")]
    # Signed valence is more interpretable than separate positive/negative ratings.
    events["valence"] = events["valence_positive"] - events["valence_negative"]
    columns = ["valence", "arousal", *emotion_cols]
    if args.presentation_events:
        presentation = pd.read_csv(args.presentation_events, sep="\t").dropna(subset=["onset", "videotime"])
        presentation = presentation.sort_values("onset").drop_duplicates("onset")
        scanner_time = presentation["onset"].to_numpy(dtype=float)
        movie_time = presentation["videotime"].to_numpy(dtype=float)

        def to_movie_time(times: np.ndarray) -> np.ndarray:
            # The presentation log records the true display time of every movie
            # frame relative to scanner start. Linear interpolation captures
            # the observed sub-TR frame timing/drift; linear extrapolation only
            # applies outside the logged range.
            return np.interp(times, scanner_time, movie_time)
    else:
        def to_movie_time(times: np.ndarray) -> np.ndarray:
            return times
    labels = np.full((args.n_trs, len(columns)), np.nan, dtype=np.float32)
    for bold_tr in range(args.n_trs):
        stimulus_start = (bold_tr - args.lag_trs) * args.tr
        stimulus_end = (bold_tr + 1 - args.lag_trs) * args.tr
        if stimulus_start >= 0:
            start, end = to_movie_time(np.asarray([stimulus_start, stimulus_end]))
            value = interval_mean(events, start, end, columns)
            if value is not None:
                labels[bold_tr] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=labels, valid=np.isfinite(labels).all(axis=1),
                        columns=np.asarray(columns), tr=args.tr, lag_trs=args.lag_trs,
                        source=str(source), presentation_events=str(args.presentation_events or ""))
    print(f"{args.output}: {np.isfinite(labels).all(axis=1).sum()}/{args.n_trs} labelled TRs")


if __name__ == "__main__":
    main()
