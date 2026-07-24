# Forrest Gump NeuroHackademy

Collaborative research repository for complementary analyses of the StudyForrest dataset. Each contributor owns a clearly named top-level directory and should commit code, concise documentation, and lightweight findings only. Raw data, derivatives, feature caches, and trained weights stay outside this repository.

## Research directions

| Direction | Lead | Current scope |
| --- | --- | --- |
| [Emotion analysis — Qiaoxin](Emotion%20analysis-Qiaoxin/README.md) | Qiaoxin | Human emotion annotations, cross-subject ROI decoding, and inter-subject correlation during the movie. |
| Localization of regions via Encoding Analysis | Pushpita | Using the available annotations in the dataset, this work uses encoding model-based analyses to localize high- and low-level audio and visual information processing in one subject. |
| Speech vs Non Speech Vocalizations at the Studyforrest dataset | Konstantinos | Using the available annotations in the audio dataset, this work tries to find the brain areas that are being activated during the Speech vs Non Speech Vocalizations, uncorrected maps are being created and an RDF analysis shows the real brain areas that are being activated during these two conditions. |
| Positive-negative contact classification from ROI | Tamar | Using body contact annotations, this work trie sto see if one functionally defined ROI (EBA) represents negative and positive contact diffeentailly for one subject (using classification). |
| The effect of annotaions on V1 activity prediction | Junru | Planned to build enocders that take annotations as auxiliary input to predict V1 activity. Failed due to the struggle in alignning MRI spaces. |
| Group localizer task (GLM) analysis | Sam | Learned nilearn through group analysis, but had some alignment and preprocessing to do too |
Please add new directions as their own top-level directories and add one row here.
66
