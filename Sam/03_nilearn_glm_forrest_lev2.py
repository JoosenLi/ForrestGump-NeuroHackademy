#!/usr/bin/env python

# 2nd-level fMRI: one sample test w permutation testing + alpha thresh visuals.
# S Torrisi + Claude, for NeuroHackademy '26, Study Forrest project.
# Final script for group analysis of localizers. See Sengupta et al 2015.

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_stat_map, show

def set_mni_space(img):
    #Explicitly flag nibabel outputs as MNI space
    img.header['sform_code'] = 4
    img.header['qform_code'] = 4
    return img

def get_union_bounds(img, z_coords_mm, pad=5, epsilon=1e-6):
    """Single bounding box (xmin, xmax, ymin, ymax) in mm that covers the
    nonzero content across ALL given axial slices (world z coords, mm).

    Using one shared box (instead of a separate box per slice) means every
    panel gets the same crop, so the figure's aspect ratio can just match
    the crop's real proportions — no need to force square panels."""
    data = img.get_fdata()
    affine = img.affine
    inv_affine = np.linalg.inv(affine)

    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    for z_mm in z_coords_mm:
        vox_pt = inv_affine @ np.array([0, 0, z_mm, 1])
        z_idx = int(np.clip(round(vox_pt[2]), 0, data.shape[2] - 1))
        sl = data[:, :, z_idx]
        mask = np.abs(sl) > epsilon
        if not mask.any():
            continue
        x_idx, y_idx = np.where(mask)
        corners_vox = np.array([
            [x_idx.min(), y_idx.min(), z_idx, 1],
            [x_idx.max(), y_idx.max(), z_idx, 1],
        ])
        corners_world = corners_vox @ affine.T
        xmin = min(xmin, corners_world[:, 0].min())
        xmax = max(xmax, corners_world[:, 0].max())
        ymin = min(ymin, corners_world[:, 1].min())
        ymax = max(ymax, corners_world[:, 1].max())

    return xmin - pad, xmax + pad, ymin - pad, ymax + pad

subjects = ['01', '02', '03', '04', '05', '06', '09', '10',
            '14', '15', '16', '17', '18', '19', '20']

contrasts = ['body', 'face', 'house', 'object', 'scene', 'scramble']
mni_head_full = "/home/jovyan/analysis/mni152_head_full.nii.gz"
results_dir = Path('results_es')
smoothing_fwhm = 5.0

# loop through 6 contrasts
for contrast_name in contrasts:
    effect_maps = [
        results_dir / f'sub-{sub}_forrest_objectcat_{contrast_name}_es.nii.gz'
        for sub in subjects
    ]

    design_matrix = pd.DataFrame([1] * len(effect_maps), columns=['intercept'])

    second_level_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
    second_level_model = second_level_model.fit(effect_maps, design_matrix=design_matrix)

    z_map = second_level_model.compute_contrast(
        second_level_contrast='intercept',
        output_type='z_score'
    )
    z_map = set_mni_space(z_map)
    z_map.to_filename(results_dir / f'forrest_{contrast_name}activ_zmap.nii.gz')

    # permutation testing
    out_dict = non_parametric_inference(
        effect_maps,
        design_matrix=design_matrix,
        model_intercept=True,
        n_perm=10000,
        two_sided_test=True,
        smoothing_fwhm=smoothing_fwhm,
        n_jobs=14,
        threshold=0.001,
        verbose=1,
    )

    for key, img in out_dict.items():
        img = set_mni_space(img)
        img.to_filename(results_dir / f'forrest_{contrast_name}activ_{key}.nii.gz')

    # Voxel-level FWE threshold: logp_max_t is -log10(p) from the max-|t| null
    # (two-sided), which is monotonic with |z|, so the smallest |z| among
    # voxels clearing alpha IS the equivalent FWE-corrected z cutoff.
    alpha = 0.05
    neg_log_p_thresh = -np.log10(alpha)
    logp_max_t = np.squeeze(out_dict['logp_max_t'].get_fdata())
    z_data = np.squeeze(z_map.get_fdata())
    sig_voxels = logp_max_t >= neg_log_p_thresh
    if sig_voxels.any():
        threshold = np.abs(z_data)[sig_voxels].min()
        print(f'{contrast_name}: FWE alpha={alpha} -> z threshold = {threshold:.3f}')
    else:
        threshold = 3
        print(f'{contrast_name}: no voxels survive FWE alpha={alpha}; falling back to threshold={threshold}')

    # use transparent thresholding! (see Taylor et al, "Go Figure..." in press, 2026)
    bg_img = nib.load(mni_head_full)

    plotting_config = {
            "bg_img": bg_img,
            "draw_cross": False,
            "cmap": "cold_white_hot",
            "vmax": 12,
    }
    vmin = threshold * 0.5

    n_cols, n_rows = 5, 3
    n_slices = n_cols * n_rows
    z_coords = np.linspace(-40, 0, n_slices)

    # one crop box shared by every panel, sized off the same brain mask
    xmin, xmax, ymin, ymax = get_union_bounds(z_map, z_coords, pad=5)
    ymin -= 10  # extra room for the posterior edge
    cell_w = 2.0
    cell_h = cell_w * (ymax - ymin) / (xmax - xmin)  # match panel shape to crop shape

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * cell_w, n_rows * cell_h))
    axes = axes.ravel()

    for coord, ax in zip(z_coords, axes):
        display = plot_stat_map(
            z_map,
            transparency=z_map,
            transparency_range=[vmin, threshold],
            annotate=False,
            display_mode="z",
            cut_coords=[coord],
            axes=ax,
            colorbar=False,
            **plotting_config,
        )
        display.add_contours(
            z_map, filled=False, levels=[-threshold, threshold], colors=["k"], linewidths=0.5, linestyles="solid",
        )
        for cut_ax in display.axes.values():
            cut_ax.ax.set_xlim(xmin, xmax)
            cut_ax.ax.set_ylim(ymin, ymax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(f"montage_{contrast_name}_fweZ{threshold:.2f}.jpg", dpi=200, bbox_inches="tight", pad_inches=0)

print("done")
