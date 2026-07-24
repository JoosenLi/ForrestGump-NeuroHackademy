#!/usr/bin/env python3
"""Merge per-subject feature archives without changing their sample order."""
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(args.input_dir.glob("sub-*.npz"))
    if not files:
        raise FileNotFoundError(f"No sub-*.npz files in {args.input_dir}")
    sets = [np.load(p, allow_pickle=False) for p in files]
    names = sets[0]["feature_names"]
    targets = sets[0]["target_columns"]
    if any(not np.array_equal(s["feature_names"], names) for s in sets):
        raise ValueError("Feature names differ across subjects")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        X=np.concatenate([s["X"] for s in sets]),
                        y=np.concatenate([s["y"] for s in sets]),
                        subject=np.concatenate([s["subject"] for s in sets]),
                        run=np.concatenate([s["run"] for s in sets]),
                        feature_names=names, target_columns=targets)
    print(f"saved {args.output}: {len(files)} subjects, X={np.concatenate([s['X'] for s in sets]).shape}")


if __name__ == "__main__":
    main()
