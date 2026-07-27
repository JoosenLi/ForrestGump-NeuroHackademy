#!/usr/bin/env python

# S Torrisi + Claude, for NeuroHackademy '26, Study Forrest project.
# Third script for single-subj GLM analysis of localizers. See Sengupta et al 2015.

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import view_img
from nilearn import image

subjects = ['01', '02', '03', '04', '05', '06', '09', '10',
            '14', '15', '16', '17', '18', '19', '20']

for sub in subjects:

    print(f"=== Running subject {sub} ===")

    epi_paths = [
        f'analysis/sub-{sub}_run-1_bold_in_mni.nii.gz',
        f'analysis/sub-{sub}_run-2_bold_in_mni.nii.gz',
        f'analysis/sub-{sub}_run-3_bold_in_mni.nii.gz',
        f'analysis/sub-{sub}_run-4_bold_in_mni.nii.gz',
    ]

    motion_confounds_paths = [
        f'analysis/sub-{sub}_task-objectcategories_run-1_bold_mcparams.txt',
        f'analysis/sub-{sub}_task-objectcategories_run-2_bold_mcparams.txt',
        f'analysis/sub-{sub}_task-objectcategories_run-3_bold_mcparams.txt',
        f'analysis/sub-{sub}_task-objectcategories_run-4_bold_mcparams.txt',
    ]

    # motion confounds (already provided by the open dataset)
    confounds_list = [
        pd.read_csv(p, sep=r'\s+', header=None,
                    names=['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])
        for p in motion_confounds_paths
    ]

    # sanity check row counts match volumes
    for img_path, conf in zip(epi_paths, confounds_list):
        n_vols = nib.load(img_path).shape[-1]
        print(img_path, n_vols, len(conf))

    # stimuli events
    events = pd.read_table('studyForrest_localizer_timings.tsv')
    events_list = [events.copy() for _ in range(4)]

    # GLM model
    fmri_glm = FirstLevelModel(
        t_r=2,
        noise_model="ar1",
        standardize=False,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        verbose=1,
    )

    # fit GLM
    fmri_glm = fmri_glm.fit(epi_paths, events=events_list, confounds=confounds_list)

    events = pd.read_table('studyForrest_localizer_timings.tsv')
    events_list = [events.copy() for _ in range(4)]
    design_matrix = fmri_glm.design_matrices_[0]

    output_dir = Path.cwd() / "results_es"
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for i, (dm, ax) in enumerate(zip(fmri_glm.design_matrices_, axes)):
        plot_design_matrix(dm, axes=ax)
        ax.set_title(f"Run {i+1}")
    plt.tight_layout()
    fig.savefig(output_dir / f'sub-{sub}_forrest_objcat_designmtx.png', dpi=150, bbox_inches='tight')
    plt.show()

    # single condition contrasts. some hard-coding below for making this fast and clear
    n_regressors = design_matrix.shape[1]
    # allactiv = np.zeros(n_regressors)
    # allactiv[:6] = 1
    bodyactiv = np.zeros(n_regressors)
    bodyactiv[0] = 1
    faceactiv = np.zeros(n_regressors)
    faceactiv[1] = 1
    houseactiv = np.zeros(n_regressors)
    houseactiv[2] = 1
    objectactiv = np.zeros(n_regressors)
    objectactiv[3] = 1
    sceneactiv = np.zeros(n_regressors)
    sceneactiv[4] = 1
    scrambleactiv = np.zeros(n_regressors)
    scrambleactiv[5] = 1

    # look at single condition contrasts
    from nilearn.plotting import plot_contrast_matrix
    plot_contrast_matrix(contrast_def=bodyactiv, design_matrix=design_matrix, colorbar=False)
    plot_contrast_matrix(contrast_def=faceactiv, design_matrix=design_matrix, colorbar=False)
    plot_contrast_matrix(contrast_def=houseactiv, design_matrix=design_matrix, colorbar=False)
    plot_contrast_matrix(contrast_def=objectactiv, design_matrix=design_matrix, colorbar=False)
    plot_contrast_matrix(contrast_def=sceneactiv, design_matrix=design_matrix, colorbar=False)
    plot_contrast_matrix(contrast_def=scrambleactiv, design_matrix=design_matrix, colorbar=False)

    eff_map_body = fmri_glm.compute_contrast(bodyactiv, output_type="effect_size")
    eff_map_face = fmri_glm.compute_contrast(faceactiv, output_type="effect_size")
    eff_map_house = fmri_glm.compute_contrast(houseactiv, output_type="effect_size")
    eff_map_object = fmri_glm.compute_contrast(objectactiv, output_type="effect_size")
    eff_map_scene = fmri_glm.compute_contrast(sceneactiv, output_type="effect_size")
    eff_map_scramble = fmri_glm.compute_contrast(scrambleactiv, output_type="effect_size")

    # save effect sizes, aka betas, aka copes
    eff_map_body.to_filename(f'results_es/sub-{sub}_forrest_objectcat_body_es.nii.gz')
    eff_map_face.to_filename(f'results_es/sub-{sub}_forrest_objectcat_face_es.nii.gz')
    eff_map_house.to_filename(f'results_es/sub-{sub}_forrest_objectcat_house_es.nii.gz')
    eff_map_object.to_filename(f'results_es/sub-{sub}_forrest_objectcat_object_es.nii.gz')
    eff_map_scene.to_filename(f'results_es/sub-{sub}_forrest_objectcat_scene_es.nii.gz')
    eff_map_scramble.to_filename(f'results_es/sub-{sub}_forrest_objectcat_scramble_es.nii.gz')

    z_map_body = fmri_glm.compute_contrast(bodyactiv, output_type="z_score")
    z_map_face = fmri_glm.compute_contrast(faceactiv, output_type="z_score")
    z_map_house = fmri_glm.compute_contrast(houseactiv, output_type="z_score")
    z_map_object = fmri_glm.compute_contrast(objectactiv, output_type="z_score")
    z_map_scene = fmri_glm.compute_contrast(sceneactiv, output_type="z_score")
    z_map_scramble = fmri_glm.compute_contrast(scrambleactiv, output_type="z_score")

    # glance at body contrast to check
    run1 = epi_paths[1]
    mean_epi = image.mean_img(run1)
    thresh = 4

    view = view_img(
        z_map_body,
        bg_img=mean_epi,
        threshold=thresh,
        black_bg=True,
        title=f's-{sub} FG body loc (|Z|>{thresh})',
    )
    view

    # glance again but clusterized
    from nilearn.glm import threshold_stats_img
    from nilearn.plotting import view_img

    clean_map, threshold = threshold_stats_img(
        z_map_body,
        alpha=0.01,
        height_control="fdr",
        cluster_threshold=20,
        two_sided=True,
    )

    view = view_img(
        clean_map,
        bg_img=mean_epi,
        threshold=threshold,
        black_bg=True,
        title=(
            f"s-{sub} FG body localizer "
            f"(p<0.01 FDR-corrected; threshold: {threshold:.3f}; "
            "clusters > 40 voxels)"
        ),
    )
    view

    print(f"done with subject {sub}")

print("all subjects done")
