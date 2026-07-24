#!/usr/bin/env python3
"""Build comparable cortical-ROI movie features from FreeSurfer segmentations.

Each movie run is affinely registered (mutual information) to its subject's
FreeSurfer T1.  `aparc+aseg` is then resampled back to that EPI grid.  Features
are per-run z-scored BOLD means in homologous FreeSurfer regions, so train and
test subjects do not require invalid native-voxel correspondence.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import signal


ROIS = {
    "caudal_ACC": (1002, 2002), "rostral_ACC": (1026, 2026),
    "medial_OFC": (1014, 2014), "lateral_OFC": (1012, 2012),
    "rostral_middle_frontal": (1027, 2027), "insula": (1035, 2035),
    "temporal_pole": (1033, 2033), "inferior_parietal": (1008, 2008),
    "amygdala": (18, 54), "hippocampus": (17, 53), "thalamus": (10, 49),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    # ANTs emits thousands of per-iteration diagnostics.  Suppressing normal
    # stdout keeps a batch/remote execution channel from truncating and
    # terminating an otherwise healthy registration; errors remain visible.
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def save_mgz_as_nifti(source: Path, target: Path) -> None:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        image = nib.load(str(source))
        nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine, image.header), str(target))


def prepare_segmentation(bold_file: Path, fs_subject: Path, work: Path) -> Path:
    """Return aparc+aseg in this run's EPI space, creating cached files."""
    output = work / "aparc+aseg_in_bold.nii.gz"
    if output.exists():
        return output
    work.mkdir(parents=True, exist_ok=True)
    t1 = work.parent / "fs_brain.nii.gz"
    seg = work.parent / "fs_aparc+aseg.nii.gz"
    save_mgz_as_nifti(fs_subject / "mri" / "brain.mgz", t1)
    save_mgz_as_nifti(fs_subject / "mri" / "aparc+aseg.mgz", seg)
    bold = nib.load(str(bold_file))
    mean = np.asanyarray(bold.dataobj).mean(axis=3, dtype=np.float32)
    mean_file = work / "bold_mean.nii.gz"
    nib.save(nib.Nifti1Image(mean, bold.affine, bold.header), str(mean_file))
    prefix = str(work / "bold_to_fs_")
    # Affine MI registration is intentionally conservative for EPI-to-T1.
    run(["antsRegistrationSyNQuick.sh", "-d", "3", "-f", str(t1), "-m", str(mean_file),
         "-o", prefix, "-t", "a", "-n", "1"])
    affine = work / "bold_to_fs_0GenericAffine.mat"
    run(["antsApplyTransforms", "-d", "3", "-i", str(seg), "-r", str(mean_file),
         "-o", str(output), "-n", "NearestNeighbor", "-t", f"[{affine},1]"])
    return output


def roi_timeseries(bold_file: Path, segmentation: Path) -> np.ndarray:
    bold = nib.load(str(bold_file))
    data = np.asanyarray(bold.dataobj, dtype=np.float32)
    seg = np.asanyarray(nib.load(str(segmentation)).dataobj).astype(np.int32)
    if data.shape[:3] != seg.shape:
        raise ValueError(f"grid mismatch: {bold_file} {data.shape[:3]} vs {seg.shape}")
    flat = data.reshape(-1, data.shape[3])
    flat = signal.detrend(flat, axis=1, type="linear")
    flat /= np.maximum(flat.std(axis=1, keepdims=True), 1e-6)
    labels = seg.ravel()
    features = []
    for name, ids in ROIS.items():
        mask = np.isin(labels, ids)
        if mask.sum() < 3:
            raise ValueError(f"{bold_file}: ROI {name} has only {mask.sum()} voxels")
        features.append(flat[mask].mean(axis=0))
    return np.stack(features, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", type=int, default=range(1, 9))
    args = ap.parse_args()
    fs_root = args.dataset / "derivatives" / "freesurfer"
    all_x, all_y, all_subject, all_run = [], [], [], []
    target_columns = None
    for subject in (s.zfill(2) for s in args.subjects):
        for number in args.runs:
            pattern = f"sub-{subject}_ses-movie_task-movie_run-{number}_bold.nii.gz"
            bold = args.dataset / f"sub-{subject}" / "ses-movie" / "func" / pattern
            label_file = args.labels_dir / f"run-{number:02d}.npz"
            if not bold.exists() or not label_file.exists():
                raise FileNotFoundError(f"missing {bold} or {label_file}")
            segmentation = prepare_segmentation(bold, fs_root / f"sub-{subject}",
                                                args.work_dir / f"sub-{subject}" / f"run-{number:02d}")
            x = roi_timeseries(bold, segmentation)
            label_data = np.load(label_file, allow_pickle=False)
            valid, y = label_data["valid"], label_data["labels"]
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"TR mismatch in {bold}: {x.shape[0]} vs {y.shape[0]}")
            all_x.append(x[valid]); all_y.append(y[valid])
            all_subject.extend([subject] * valid.sum()); all_run.extend([number] * valid.sum())
            target_columns = label_data["columns"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=np.concatenate(all_x), y=np.concatenate(all_y),
                        subject=np.asarray(all_subject), run=np.asarray(all_run),
                        feature_names=np.asarray(list(ROIS)), target_columns=target_columns)
    print(f"saved {args.output}: X={np.concatenate(all_x).shape}, subjects={sorted(set(all_subject))}")


if __name__ == "__main__":
    main()
