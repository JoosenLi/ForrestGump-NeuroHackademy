#!/usr/bin/env python3
"""GPU-only LOSO and within-subject evaluation for ROI emotion decoding."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or y.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def predict(train_x, train_y, test_x, *, epochs: int, seed: int, checkpoint: Path | None = None) -> np.ndarray:
    """Fit a small nonlinear regression model exclusively on CUDA."""
    torch.manual_seed(seed)
    x_mean, x_std = train_x.mean(0), train_x.std(0)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    train_x = (train_x - x_mean) / np.maximum(x_std, 1e-6)
    test_x = (test_x - x_mean) / np.maximum(x_std, 1e-6)
    train_y = (train_y - y_mean) / np.maximum(y_std, 1e-6)
    device = torch.device("cuda")
    x = torch.as_tensor(train_x, device=device)
    y = torch.as_tensor(train_y, device=device)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, y.shape[1])).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    for _ in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward(); optimiser.step()
    with torch.no_grad():
        output = model(torch.as_tensor(test_x, device=device)).cpu().numpy()
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                    "architecture": "Linear(roi,64)->GELU->Linear(64,target)"}, checkpoint)
    return output * y_std + y_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mode", choices=("loso", "within"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", default=["valence", "arousal"])
    ap.add_argument("--train-runs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="directory for one GPU-trained checkpoint per held-out subject")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: CPU training is intentionally disabled.")
    torch.backends.cudnn.benchmark = False
    raw = np.load(args.data, allow_pickle=False)
    target_names = raw["target_columns"].astype(str).tolist()
    missing = set(args.targets).difference(target_names)
    if missing:
        raise ValueError(f"Unknown targets: {sorted(missing)}")
    target_idx = [target_names.index(t) for t in args.targets]
    x, y = raw["X"].astype(np.float32), raw["y"][:, target_idx].astype(np.float32)
    subject, runs = raw["subject"].astype(str), raw["run"].astype(int)
    rows = []
    for fold, held_out in enumerate(sorted(np.unique(subject))):
        if args.mode == "loso":
            test = subject == held_out
        else:
            is_subject = subject == held_out
            test = is_subject & ~np.isin(runs, args.train_runs)
        train = ~test if args.mode == "loso" else ((subject == held_out) & np.isin(runs, args.train_runs))
        if not train.any() or not test.any():
            raise ValueError(f"Empty split for subject {held_out}")
        prediction = predict(x[train], y[train], x[test], epochs=args.epochs, seed=20260724 + fold,
                             checkpoint=args.model_dir / f"sub-{held_out}.pt")
        for j, target in enumerate(args.targets):
            rows.append({"mode": args.mode, "subject": held_out, "target": target,
                         "n_train": int(train.sum()), "n_test": int(test.sum()),
                         "pearson_r": pearson(y[test, j], prediction[:, j]),
                         "mse": float(np.mean((y[test, j] - prediction[:, j]) ** 2))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"saved {args.output}; checkpoints={args.model_dir}; CUDA={torch.cuda.get_device_name(0)}; {len(rows)} scores")


if __name__ == "__main__":
    main()
