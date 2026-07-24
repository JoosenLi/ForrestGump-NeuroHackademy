# Emotion analysis — Qiaoxin

Initial analysis of the StudyForrest **3 T `ses-movie` audiovisual** experiment: 15 participants watched the same eight movie runs. This is distinct from the dataset's 7 T audio-description session.

## What was done

- Converted the curated human segment annotations into TR-level targets using run-specific presentation logs, then swept BOLD delays from 2 to 12 s.
- Registered each participant's movie BOLD data to their FreeSurfer anatomy and extracted mean signals from 11 homologous cortical/subcortical ROIs.
- Evaluated valence and arousal regression with leave-one-subject-out (train 14 people, test the held-out person) and within-subject testing (train runs 1–6, test runs 7–8). Fitting is CUDA-only.
- Calculated 12-s delay inter-subject correlation (ISC) for matching movie cuts. Inferior parietal cortex showed the strongest mean leave-one-out ISC (r = 0.289); rostral ACC followed (r = 0.217).

The lag sweep is an exploratory baseline, not a claim of robust emotion decoding. At 12 s, mean LOSO correlation was r = 0.056 for arousal and r = 0.034 for valence; statistical testing and independent confirmation remain necessary.

## Contents

- `code/`: label construction, FreeSurfer-ROI extraction, feature merging, GPU evaluation, and figure scripts.
- `findings/`: three lightweight figures: human annotations, 12-s ISC, and decoding lag sweep.

## Data and reproducibility

Data are not stored here. Obtain the BIDS dataset from [OpenNeuro ds000113](https://openneuro.org/datasets/ds000113/versions/1.3.0), then obtain the StudyForrest FreeSurfer derivative separately. The scripts document their command-line interfaces and use NumPy, SciPy, pandas, nibabel, matplotlib, PyTorch/CUDA, and ANTs.

The human labels include segment-level continuous valence, arousal, and 22 discrete emotion dimensions; the current regression baseline reports only valence and arousal.
